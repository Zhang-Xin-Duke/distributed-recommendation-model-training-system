#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <brpc/redis.h>
#include <google/protobuf/service.h>

namespace rsdk {
class Status {
 public:
  enum Code { success = 0, invalid_argument = 1, unavailable = 2, server_error = 3 };
  Status() = default;
  Status(Code code, std::string message) : code_(code), message_(std::move(message)) {}
  bool ok() const { return code_ == success; }
  Code code() const { return code_; }
  std::string to_string() const { return ok() ? "OK" : message_; }
 private:
  Code code_ = success;
  std::string message_;
};
namespace client {
struct AccessOptions {
  std::string _cname;
  std::int32_t timeout_ms = 1000;
  bool _merge_batch_as_array = false;
};
struct Record {
  std::string _key, _data;
  std::int64_t _ttl_s = 0;
};
using RecordList = std::vector<Record>;
struct Endpoint {
  std::string address, password;
  int database = 0;
};
class Client {
 public:
  Client(Endpoint, AccessOptions);
  ~Client();
  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;
  // Completion callbacks run synchronously before return. Original callers use
  // nullptr. Request and response ownership always stays with the caller.
  Status exec(const brpc::RedisRequest&, brpc::RedisResponse*, google::protobuf::Closure* = nullptr);
  Status mget(const RecordList&, std::vector<brpc::RedisResponse>*, google::protobuf::Closure* = nullptr);
  Status mset(const RecordList&, std::vector<brpc::RedisResponse>*, google::protobuf::Closure* = nullptr);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Status Batch(const RecordList&, std::vector<brpc::RedisResponse>*, bool);
};
}  // namespace client
}  // namespace rsdk
