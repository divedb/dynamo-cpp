#include <dynamo/llm/protocols.h>

#include <chrono>

namespace dynamo::llm {

void to_json(nlohmann::json& j, const GenerationParams& p) {
    j = nlohmann::json{
        {"max_tokens", p.max_tokens},
        {"temperature", p.temperature},
        {"top_p", p.top_p},
        {"top_k", p.top_k},
        {"stream", p.stream},
        {"min_tokens", p.min_tokens},
    };
    if (p.seed) j["seed"] = *p.seed;
    if (!p.stop_token_ids.empty()) j["stop_token_ids"] = p.stop_token_ids;
}

void from_json(const nlohmann::json& j, GenerationParams& p) {
    p.max_tokens = j.value("max_tokens", 256);
    p.temperature = j.value("temperature", 1.0);
    p.top_p = j.value("top_p", 1.0);
    p.top_k = j.value("top_k", 50);
    p.stream = j.value("stream", true);
    p.min_tokens = j.value("min_tokens", 0);
    if (j.contains("seed") && !j["seed"].is_null())
        p.seed = j["seed"].get<int>();
    if (j.contains("stop_token_ids"))
        p.stop_token_ids = j["stop_token_ids"].get<std::vector<int32_t>>();
}

void to_json(nlohmann::json& j, const CompletionRequest& r) {
    j = nlohmann::json{
        {"model", r.model},
        {"prompt", r.prompt},
        {"params", r.params},
    };
    if (!r.input_ids.empty()) j["input_ids"] = r.input_ids;
}

void from_json(const nlohmann::json& j, CompletionRequest& r) {
    r.model = j.value("model", "");
    r.prompt = j.value("prompt", "");
    r.params = j.value("params", GenerationParams{});
    if (j.contains("input_ids"))
        r.input_ids = j["input_ids"].get<std::vector<int32_t>>();
}

void to_json(nlohmann::json& j, const CompletionResponse& r) {
    j = nlohmann::json{
        {"request_id", r.request_id},
        {"token_id", r.token_id},
        {"logprob", r.logprob},
        {"finished", r.finished},
        {"generated_tokens", r.generated_tokens},
    };
    if (!r.finish_reason.empty()) j["finish_reason"] = r.finish_reason;
}

void to_json(nlohmann::json& j, const ChatMessage& m) {
    j = nlohmann::json{
        {"role", m.role},
        {"content", m.content},
    };
}

void to_json(nlohmann::json& j, const ChatCompletionRequest& r) {
    j = nlohmann::json{
        {"model", r.model},
        {"messages", r.messages},
        {"params", r.params},
    };
}

} // namespace dynamo::llm
