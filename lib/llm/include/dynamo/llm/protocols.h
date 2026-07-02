#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace dynamo::llm {

// ---------------------------------------------------------------------------
// Request / Response types matching OpenAI API + Dynamo extensions
// ---------------------------------------------------------------------------

struct GenerationParams {
    int max_tokens = 256;
    double temperature = 1.0;
    double top_p = 1.0;
    int top_k = 50;
    std::optional<int> seed;
    bool stream = true;
    std::vector<int32_t> stop_token_ids;
    int min_tokens = 0;
};

struct CompletionRequest {
    std::string model;
    std::string prompt;
    std::vector<int32_t> input_ids;
    GenerationParams params;
};

struct CompletionResponse {
    std::string request_id;
    int32_t token_id = -1;
    double logprob = 0.0;
    bool finished = false;
    std::string finish_reason;
    int generated_tokens = 0;
};

struct ChatMessage {
    std::string role;   // "system", "user", "assistant"
    std::string content;
};

struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    GenerationParams params;
};

struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created;
    std::string model;
    std::vector<ChatMessage> choices;
    std::string finish_reason;
};

// ---------------------------------------------------------------------------
// Annotated<T> — universal response envelope
// ---------------------------------------------------------------------------

template <typename T>
struct Annotated {
    std::optional<T> data;
    std::optional<std::string> event;
    std::optional<std::string> comment;

    static Annotated make_data(T d) {
        return Annotated{std::move(d), std::nullopt, std::nullopt};
    }

    static Annotated make_event(std::string e) {
        return Annotated{std::nullopt, std::move(e), std::nullopt};
    }

    bool has_data() const noexcept { return data.has_value(); }
    bool is_event() const noexcept { return event.has_value(); }
};

// ---------------------------------------------------------------------------
// JSON conversion
// ---------------------------------------------------------------------------

void to_json(nlohmann::json& j, const GenerationParams& p);
void from_json(const nlohmann::json& j, GenerationParams& p);
void to_json(nlohmann::json& j, const CompletionRequest& r);
void from_json(const nlohmann::json& j, CompletionRequest& r);
void to_json(nlohmann::json& j, const CompletionResponse& r);
void to_json(nlohmann::json& j, const ChatMessage& m);
void to_json(nlohmann::json& j, const ChatCompletionRequest& r);

} // namespace dynamo::llm
