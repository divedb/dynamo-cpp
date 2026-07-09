// SPDX-License-Identifier: Apache-2.0

#include "llm/tokenizers.h"

#include <algorithm>
#include <stdexcept>

namespace dynamo::llm {

namespace {

constexpr std::string_view kReplacementChar = "\xEF\xBF\xBD";  // U+FFFD

bool is_char_boundary(const std::string& s, size_t index) {
  if (index == 0 || index >= s.size()) return true;
  return (static_cast<unsigned char>(s[index]) & 0xC0) != 0x80;
}

bool ends_with_replacement(const std::string& s) {
  return s.size() >= 3 && std::string_view(s).substr(s.size() - 3) == kReplacementChar;
}

std::string remove_replacement_chars(std::string s) {
  size_t pos;
  while ((pos = s.find(kReplacementChar)) != std::string::npos) {
    s.erase(pos, kReplacementChar.size());
  }
  return s;
}

/// Expected length of a UTF-8 sequence from its lead byte; 0 for invalid.
size_t utf8_seq_len(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead & 0xE0) == 0xC0) return 2;
  if ((lead & 0xF0) == 0xE0) return 3;
  if ((lead & 0xF8) == 0xF0) return 4;
  return 0;
}

/// Lossy UTF-8: valid sequences pass through, invalid bytes and the
/// incomplete trailing sequence each become one U+FFFD.
std::string utf8_lossy(const std::string& bytes) {
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  while (i < bytes.size()) {
    auto lead = static_cast<unsigned char>(bytes[i]);
    size_t len = utf8_seq_len(lead);
    if (len == 0) {
      out += kReplacementChar;
      ++i;
      continue;
    }
    if (i + len > bytes.size()) {
      // Incomplete trailing sequence.
      out += kReplacementChar;
      break;
    }
    bool valid = true;
    for (size_t k = 1; k < len; ++k) {
      if ((static_cast<unsigned char>(bytes[i + k]) & 0xC0) != 0x80) {
        valid = false;
        break;
      }
    }
    if (!valid) {
      out += kReplacementChar;
      ++i;
      continue;
    }
    out.append(bytes, i, len);
    i += len;
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// ByteLevelTokenizer

Encoding ByteLevelTokenizer::encode(const std::string& input) const {
  Encoding encoding;
  encoding.token_ids.reserve(input.size());
  encoding.tokens.reserve(input.size());
  encoding.spans.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    encoding.token_ids.push_back(static_cast<unsigned char>(input[i]));
    encoding.tokens.push_back(input.substr(i, 1));
    encoding.spans.emplace_back(i, i + 1);
  }
  return encoding;
}

std::string ByteLevelTokenizer::decode(const std::vector<TokenIdType>& token_ids,
                                       bool skip_special_tokens) const {
  std::string bytes;
  std::string out;
  auto flush_bytes = [&] {
    out += utf8_lossy(bytes);
    bytes.clear();
  };
  for (TokenIdType id : token_ids) {
    if (id < 256) {
      bytes.push_back(static_cast<char>(id));
      continue;
    }
    flush_bytes();
    if (skip_special_tokens) continue;
    if (id == kBosId) {
      out += "<s>";
    } else if (id == kEosId) {
      out += "</s>";
    } else {
      throw std::runtime_error("ByteLevelTokenizer: unknown token id " + std::to_string(id));
    }
  }
  flush_bytes();
  return out;
}

// ---------------------------------------------------------------------------
// Sequence

void Sequence::append_text(const std::string& input) {
  auto encoding = tokenizer_->encode(input);
  token_ids_.insert(token_ids_.end(), encoding.token_ids.begin(), encoding.token_ids.end());
}

std::string Sequence::append_token_id(TokenIdType token_id) {
  token_ids_.push_back(token_id);

  std::vector<TokenIdType> prefix_ids(token_ids_.begin() + static_cast<ptrdiff_t>(prefix_offset_),
                                      token_ids_.begin() + static_cast<ptrdiff_t>(read_offset_));
  std::vector<TokenIdType> all_ids(token_ids_.begin() + static_cast<ptrdiff_t>(prefix_offset_),
                                   token_ids_.end());

  std::string prefix_text = tokenizer_->decode(prefix_ids, false);
  std::string new_text = tokenizer_->decode(all_ids, false);

  size_t prefix_len = prefix_text.size();
  while (prefix_len > 0 && !is_char_boundary(new_text, prefix_len)) --prefix_len;

  if (new_text.size() > prefix_text.size()) {
    if (ends_with_replacement(new_text)) {
      return "";  // incomplete character at the tail; hold
    }
    std::string emitted = remove_replacement_chars(new_text.substr(prefix_len));
    prefix_offset_ = read_offset_;
    read_offset_ = token_ids_.size();
    return emitted;
  }
  return "";
}

// ---------------------------------------------------------------------------
// StopSequenceDecoder

SequenceDecoderOutput StopSequenceDecoder::append_token_id(TokenIdType token_id) {
  if (stopped_) throw std::runtime_error("Decoder is stopped");

  state_ += sequence_.append_token_id(token_id);

  bool stop = false;
  bool visible = false;
  if (std::find(config_.stop_token_ids_visible.begin(), config_.stop_token_ids_visible.end(),
                token_id) != config_.stop_token_ids_visible.end()) {
    stop = true;
    visible = true;
  }
  if (std::find(config_.stop_token_ids_hidden.begin(), config_.stop_token_ids_hidden.end(),
                token_id) != config_.stop_token_ids_hidden.end()) {
    stop = true;
    visible = false;
  }

  if (stop) {
    stopped_ = true;
    SequenceDecoderOutput out;
    if (visible) {
      out.kind = SequenceDecoderOutput::Kind::stopped_with_text;
      out.text = std::move(state_);
    } else {
      out.kind = SequenceDecoderOutput::Kind::stopped;
    }
    state_.clear();
    return out;
  }

  for (const auto& stop_sequence : config_.stop_sequences_hidden) {
    if (stop_sequence.rfind(state_, 0) == 0) {  // state_ is a prefix of the stop sequence
      if (stop_sequence == state_) {
        stopped_ = true;
        return {SequenceDecoderOutput::Kind::stopped, {}};
      }
      return {SequenceDecoderOutput::Kind::held, {}};
    }
  }

  SequenceDecoderOutput out;
  out.kind = SequenceDecoderOutput::Kind::text;
  out.text = std::move(state_);
  state_.clear();
  return out;
}

}  // namespace dynamo::llm
