// SPDX-License-Identifier: Apache-2.0
//
// M4 tests: ModelDeploymentCard construction from a local HF-format
// directory, JSON round-trips (externally-tagged artifacts, RFC3339
// last_published), slug/mdcsum, publish/list over discovery kv with lease
// binding, and the from_mdc factories.

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <unistd.h>

#include "discovery/in_process.h"
#include "llm/backend.h"
#include "llm/model_card.h"
#include "llm/preprocessor.h"
#include "runtime/coro/sync_wait.h"
#include "runtime/runtime.h"

using namespace dynamo::llm;
namespace fs = std::filesystem;
namespace coro = dynamo::coro;
using dynamo::Runtime;
namespace discovery = dynamo::discovery;

namespace {

/// Creates a temp HF-style model directory; removed on destruction.
struct ModelDir {
  fs::path root;

  explicit ModelDir(const std::string& name) {
    root = fs::temp_directory_path() / ("dynamo_mdc_test_" + name + "_" +
                                        std::to_string(::getpid()));
    fs::create_directories(root);
    write("config.json", R"({
      "model_type": "llama",
      "bos_token_id": 1,
      "eos_token_id": [2, 3],
      "max_position_embeddings": 4096,
      "num_hidden_layers": 32,
      "num_attention_heads": 32,
      "vocab_size": 32000,
      "architectures": ["LlamaForCausalLM"]
    })");
    write("tokenizer.json", "{}");
    write("tokenizer_config.json", R"({
      "bos_token": {"content": "<s>"},
      "eos_token": "</s>",
      "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}{% if add_generation_prompt %}>{% endif %}"
    })");
  }

  void write(const char* name, const std::string& content) {
    std::ofstream out(root / name);
    out << content;
  }

  ~ModelDir() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

}  // namespace

TEST_CASE("mdc from local hf directory", "[llm][mdc]") {
  ModelDir dir("local");
  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "test-model");

  CHECK(card.display_name == "test-model");
  CHECK(card.service_name == "test-model");
  CHECK(card.model_info.kind == kModelInfoHfConfigJson);
  CHECK(card.tokenizer.kind == kTokenizerHfJson);
  REQUIRE(card.prompt_formatter.has_value());
  CHECK(card.prompt_formatter->kind == kPromptFormatterHfTokenizerConfigJson);
  CHECK(card.requires_preprocessing);

  auto info = card.load_model_info();
  CHECK(info.model_type == "llama");
  CHECK(info.bos_token_id == 1u);
  CHECK(info.eos_token_ids == std::vector<TokenIdType>{2, 3});
  CHECK(info.max_position_embeddings == 4096u);
  CHECK(info.vocab_size == 32000u);

  auto config = HfTokenizerConfig::from_file(card.prompt_formatter->path);
  CHECK(config.bos_token == "<s>");   // AddedToken object form
  CHECK(config.eos_token == "</s>");  // plain string form
  REQUIRE(config.chat_template.has_value());

  CHECK_THROWS_AS(ModelDeploymentCard::from_local_path("/nonexistent/path"),
                  std::runtime_error);
}

TEST_CASE("mdc json round-trip", "[llm][mdc]") {
  ModelDir dir("json");
  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "Meta-Llama/Model.1");
  card.last_published = 1700000000;
  card.revision = 7;

  auto j = nlohmann::json(card);
  // Externally-tagged artifacts, as serde encodes them.
  CHECK(j.at("model_info").contains("hf_config_json"));
  CHECK(j.at("tokenizer").contains("hf_tokenizer_json"));
  CHECK(j.at("last_published") == "2023-11-14T22:13:20Z");
  CHECK_FALSE(j.contains("revision"));  // parity: skip_serializing

  auto back = ModelDeploymentCard::load_from_json_str(j.dump());
  CHECK(back.service_name == "Meta-Llama/Model.1");
  CHECK(back.model_info.path == card.model_info.path);
  CHECK(back.last_published == 1700000000);
  CHECK(back.revision == 0);  // not carried through JSON

  // load_from_json_file clears requires_preprocessing (parity).
  auto file = dir.root / "card.json";
  card.save_to_json_file(file.string());
  auto loaded = ModelDeploymentCard::load_from_json_file(file.string());
  CHECK_FALSE(loaded.requires_preprocessing);
}

TEST_CASE("slug and mdcsum", "[llm][mdc]") {
  std::string slug = slugify("meta-llama/Meta-Llama-3.1-8B-Instruct");
  // lowercased, non [a-z0-9_] replaced, 8-hex hash suffix
  CHECK(slug.rfind("meta_llama_meta_llama_3_1_8b_instruct_", 0) == 0);
  CHECK(slug.size() == std::string("meta_llama_meta_llama_3_1_8b_instruct_").size() + 8);
  CHECK(slug == slugify("meta-llama/Meta-Llama-3.1-8B-Instruct"));  // deterministic
  CHECK(slug != slugify("meta-llama/Meta-Llama-3.1-70B-Instruct"));

  ModelDir dir("sum");
  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "m");
  auto sum1 = card.mdcsum();
  CHECK(sum1.size() == 32);
  card.display_name = "changed";
  CHECK(card.mdcsum() != sum1);

  CHECK(card.kv_key().rfind("mdc/", 0) == 0);
  CHECK_FALSE(card.is_expired());  // never published
  card.last_published = 1;         // 1970: long expired
  CHECK(card.is_expired());
}

TEST_CASE("mdc publish and list over discovery kv", "[llm][mdc]") {
  dynamo::RuntimeConfig config;
  config.num_worker_threads = 4;
  config.num_background_threads = 2;
  auto rt = Runtime::create(config);
  auto store = discovery::InProcessDiscovery::new_store();
  auto disco = std::make_shared<discovery::InProcessDiscovery>(rt, store);

  ModelDir dir("publish");
  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "published-model");

  coro::sync_wait([&]() -> coro::Task<void> {
    auto lease = co_await disco->create_lease(std::chrono::seconds(10));
    co_await publish_model_card(*disco, card, lease.id);

    auto cards = co_await list_model_cards(*disco);
    REQUIRE(cards.size() == 1);
    CHECK(cards[0].service_name == "published-model");
    CHECK(cards[0].last_published.has_value());

    // Malformed entries are skipped, not fatal.
    co_await disco->kv_put(std::string(kMdcBucketName) + "/garbage", "not json", std::nullopt);
    cards = co_await list_model_cards(*disco);
    CHECK(cards.size() == 1);

    // The card dies with its lease.
    lease.revoke();
    cards = co_await list_model_cards(*disco);
    CHECK(cards.empty());
  }());

  disco->shutdown();
  rt.shutdown();
}

TEST_CASE("named-template list keeps default and tool_use", "[llm][mdc]") {
  ModelDir dir("named_tmpl");
  dir.write("tokenizer_config.json", R"({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "chat_template": [
      {"name": "default", "template": "D:{% for m in messages %}{{ m.content }}{% endfor %}"},
      {"name": "tool_use", "template": "T:{% for t in tools %}{{ t.function.name }}{% endfor %}"}
    ]
  })");

  auto config = HfTokenizerConfig::from_file((dir.root / "tokenizer_config.json").string());
  REQUIRE(config.chat_template.has_value());
  CHECK(config.chat_template->rfind("D:", 0) == 0);
  REQUIRE(config.tool_use_chat_template.has_value());
  CHECK(config.tool_use_chat_template->rfind("T:", 0) == 0);

  // End to end: the MDC formatter routes tool-carrying requests to tool_use.
  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "named-model");
  auto formatter = formatter_from_mdc(card);

  ChatTemplateInput input;
  input.messages.push_back({{"role", "user"}, {"content", "hi"}});
  CHECK(formatter->render(input) == "D:hi");

  input.tools = nlohmann::json::parse(
      R"([{"type": "function", "function": {"name": "get_weather"}}])");
  CHECK(formatter->render(input) == "T:get_weather");
}

TEST_CASE("prompt_context mixins flow from the card into the formatter", "[llm][mdc]") {
  ModelDir dir("mixins");
  dir.write("tokenizer_config.json", R"({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "chat_template": "{% if datetime %}dated{% else %}undated{% endif %}"
  })");

  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "dated-model");
  CHECK(formatter_from_mdc(card)->render(ChatTemplateInput{}) == "undated");

  card.prompt_context = std::vector<std::string>{"llama3_datetime"};
  CHECK(formatter_from_mdc(card)->render(ChatTemplateInput{}) == "dated");
}

TEST_CASE("preprocessor and backend from_mdc with byte-level tokenizer", "[llm][mdc]") {
  ModelDir dir("factory");
  // Byte-level model config: eos = the byte tokenizer's </s>.
  dir.write("config.json", R"({
    "model_type": "byte",
    "bos_token_id": 256,
    "eos_token_id": 257,
    "max_position_embeddings": 4096,
    "num_hidden_layers": 0,
    "num_attention_heads": 0,
    "vocab_size": 258,
    "architectures": []
  })");

  auto card = ModelDeploymentCard::from_local_path(dir.root.string(), "byte-model");
  card.tokenizer = FileRef{kTokenizerByteLevel, ""};

  auto preprocessor = OpenAIPreprocessor::from_mdc(card);
  auto backend = Backend::from_mdc(card);
  CHECK(backend != nullptr);

  openai::NvCreateChatCompletionRequest request;
  request.model = "byte-model";
  request.messages.push_back({.role = "user", .content = "abc"});
  auto [input, annotations] = preprocessor->preprocess(request);

  // Template "{{ content }}...>" renders "abc>" and eos 257 is merged in.
  CHECK(input.token_ids == std::vector<TokenIdType>{'a', 'b', 'c', '>'});
  REQUIRE(input.stop_conditions.stop_token_ids_hidden.has_value());
  CHECK(*input.stop_conditions.stop_token_ids_hidden == std::vector<TokenIdType>{257});
  CHECK(input.mdc_sum == card.mdcsum());

  // HF tokenizer backend is explicitly unsupported for now.
  card.tokenizer = FileRef{kTokenizerHfJson, "/tmp/tokenizer.json"};
  CHECK_THROWS_AS(OpenAIPreprocessor::from_mdc(card), std::runtime_error);
}
