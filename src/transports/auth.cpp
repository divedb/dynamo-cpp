// SPDX-License-Identifier: Apache-2.0

#include "transports/auth.h"

#include <cstdlib>
#include <mutex>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "transports/socket.h"

namespace dynamo::transports::auth {

namespace {

struct TokenState {
  std::mutex mutex;
  std::optional<std::string> token;
  bool env_loaded = false;

  std::optional<std::string> get() {
    std::lock_guard lock(mutex);
    if (!env_loaded) {
      env_loaded = true;
      if (const char* env = std::getenv("DYN_AUTH_TOKEN"); env && *env) token = env;
    }
    return token;
  }
};

TokenState& state() {
  static TokenState instance;
  return instance;
}

bool constant_time_equal(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  unsigned char acc = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    acc = static_cast<unsigned char>(acc | (a[i] ^ b[i]));
  }
  return acc == 0;
}

}  // namespace

void set_token(std::optional<std::string> token) {
  auto& s = state();
  std::lock_guard lock(s.mutex);
  s.env_loaded = true;  // explicit config wins over env
  s.token = std::move(token);
}

bool required() { return state().get().has_value(); }

bool send(Socket& sock) {
  auto token = state().get();
  if (!token) return true;
  nlohmann::json header{{"auth_token", *token}};
  return sock.write_frame(TwoPartMessage::from_header(header.dump()));
}

bool expect(Socket& sock) {
  auto token = state().get();
  if (!token) return true;
  sock.set_recv_timeout(std::chrono::seconds(5));
  try {
    auto frame = sock.read_frame();
    if (!frame || !frame->has_header()) return false;
    auto header = nlohmann::json::parse(frame->header);
    auto presented = header.value("auth_token", std::string());
    if (!constant_time_equal(presented, *token)) return false;
  } catch (const std::exception&) {
    return false;  // malformed frame, timeout, or non-JSON first frame
  }
  sock.set_recv_timeout(std::chrono::milliseconds::zero());
  return true;
}

}  // namespace dynamo::transports::auth
