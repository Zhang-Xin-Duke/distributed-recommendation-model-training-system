#pragma once
#include <memory>
#include <string>
#include "redis_sdk/client.h"
namespace prometheus { class Registry; }
namespace bilibili::redis {
class RClientMgr {
 public:
  RClientMgr();
  ~RClientMgr();
  RClientMgr(const RClientMgr&) = delete;
  RClientMgr& operator=(const RClientMgr&) = delete;
  rsdk::Status init(const std::string& appid, const std::string& zone, const std::string& environment);
  rsdk::Status init(prometheus::Registry*, const std::string& appid, const std::string& zone, const std::string& environment);
  rsdk::Status add_cluster(const std::string&);
  std::shared_ptr<rsdk::client::Client> new_client(const rsdk::client::AccessOptions&);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
std::shared_ptr<RClientMgr> new_rc_mgr();
}  // namespace bilibili::redis
