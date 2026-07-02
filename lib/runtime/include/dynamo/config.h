#pragma once

#include <chrono>
#include <string>
#include <string_view>
#include <toml.hpp>
#include <optional>

namespace dynamo {

struct RuntimeConfig {
    std::string runtime_name = "dynamo";
    int num_worker_threads = 16;
    int max_blocking_threads = 512;
    std::chrono::milliseconds thread_timeout{60'000};

    static RuntimeConfig load();
    static RuntimeConfig load_from_env();
    static RuntimeConfig load_from_file(std::string_view path);
};

struct WorkerConfig {
    std::chrono::milliseconds graceful_shutdown_timeout{30'000};
    bool json_logging = false;
    bool disable_ansi = false;

    static WorkerConfig load();
};

struct EtcdConfig {
    std::string endpoints = "http://127.0.0.1:2379";
    std::chrono::seconds lease_ttl{30};
    std::chrono::milliseconds dial_timeout{5'000};

    static EtcdConfig load();
};

struct NatsConfig {
    std::string server = "nats://127.0.0.1:4222";
    std::optional<std::string> username;
    std::optional<std::string> password;
    std::optional<std::string> token;

    static NatsConfig load();
};

} // namespace dynamo
