#include <dynamo/config.h>

#include <cstdlib>
#include <algorithm>
#include <spdlog/spdlog.h>

namespace dynamo {
namespace {

std::optional<std::string> env(const char* name) {
    auto* val = std::getenv(name);
    if (!val) return std::nullopt;
    return std::string(val);
}

int env_int(const char* name, int default_val) {
    auto v = env(name);
    if (!v) return default_val;
    try {
        return std::stoi(*v);
    } catch (...) {
        return default_val;
    }
}

bool env_bool(const char* name, bool default_val) {
    auto v = env(name);
    if (!v) return default_val;
    std::string lower = *v;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return lower == "true" || lower == "1" || lower == "yes";
}

} // anonymous namespace

RuntimeConfig RuntimeConfig::load() {
    auto cfg = load_from_env();
    // Try loading from default file locations
    auto file_paths = {
        "/opt/dynamo/etc/runtime.toml",
        "/opt/dynamo/defaults/runtime.toml",
        "./dynamo.toml",
        "./runtime.toml",
    };
    for (auto path : file_paths) {
        try {
            auto file_cfg = load_from_file(path);
            // Merge: file overrides env defaults
#define MERGE(field) if (file_cfg.field != decltype(file_cfg.field){}) cfg.field = file_cfg.field
            MERGE(runtime_name);
            MERGE(num_worker_threads);
            MERGE(max_blocking_threads);
            MERGE(thread_timeout);
#undef MERGE
            break;
        } catch (const std::exception& e) {
            spdlog::debug("Config: could not load {}: {}", path, e.what());
        }
    }
    return cfg;
}

RuntimeConfig RuntimeConfig::load_from_env() {
    RuntimeConfig cfg;
    cfg.runtime_name = env("DYN_RUNTIME_NAME").value_or("dynamo");
    cfg.num_worker_threads = env_int("DYN_RUNTIME_NUM_WORKER_THREADS", 16);
    cfg.max_blocking_threads = env_int("DYN_RUNTIME_MAX_BLOCKING_THREADS", 512);
    cfg.thread_timeout = std::chrono::milliseconds(
        env_int("DYN_RUNTIME_THREAD_TIMEOUT_MS", 60000));
    return cfg;
}

RuntimeConfig RuntimeConfig::load_from_file(std::string_view path) {
    RuntimeConfig cfg;
    try {
        auto data = toml::parse(std::string(path));
        if (auto* runtime = data.as_table().find("runtime"); runtime != nullptr) {
            auto tbl = runtime->second.as_table();
            if (auto v = tbl.find("name"); v != nullptr)
                cfg.runtime_name = v->second.as_string();
            if (auto v = tbl.find("num_worker_threads"); v != nullptr)
                cfg.num_worker_threads = static_cast<int>(v->second.as_integer());
            if (auto v = tbl.find("max_blocking_threads"); v != nullptr)
                cfg.max_blocking_threads = static_cast<int>(v->second.as_integer());
        }
    } catch (const std::exception&) {
        throw;
    }
    return cfg;
}

WorkerConfig WorkerConfig::load() {
    WorkerConfig cfg;
    auto timeout_str = env("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT_MS");
    if (timeout_str) {
        try {
            cfg.graceful_shutdown_timeout = std::chrono::milliseconds(
                std::stoi(*timeout_str));
        } catch (...) {}
    }
    cfg.json_logging = env_bool("DYN_LOGGING_JSONL", false);
    cfg.disable_ansi = env_bool("DYN_SDK_DISABLE_ANSI_LOGGING", false);
    return cfg;
}

EtcdConfig EtcdConfig::load() {
    EtcdConfig cfg;
    cfg.endpoints = env("DYN_ETCD_ENDPOINTS").value_or("http://127.0.0.1:2379");
    cfg.lease_ttl = std::chrono::seconds(
        env_int("DYN_ETCD_LEASE_TTL", 30));
    cfg.dial_timeout = std::chrono::milliseconds(
        env_int("DYN_ETCD_DIAL_TIMEOUT_MS", 5000));
    return cfg;
}

NatsConfig NatsConfig::load() {
    NatsConfig cfg;
    cfg.server = env("DYN_NATS_SERVER").value_or("nats://127.0.0.1:4222");
    cfg.username = env("DYN_NATS_USERNAME");
    cfg.password = env("DYN_NATS_PASSWORD");
    cfg.token = env("DYN_NATS_TOKEN");
    return cfg;
}

} // namespace dynamo
