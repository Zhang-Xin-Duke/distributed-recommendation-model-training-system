#include "bili_redis/bili_redis_client.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <brpc/channel.h>
#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <brpc/policy/redis_authenticator.h>
#include <nlohmann/json.hpp>

namespace {
constexpr std::size_t kMaxRecords = 4096, kMaxRequestBytes = 8ULL << 20;
using rsdk::Status;
void Parse(butil::IOBuf buffer, brpc::RedisResponse* response) {
  response->Clear();
  if (response->ConsumePartialIOBuf(buffer, 1) != brpc::PARSE_OK || !buffer.empty())
    throw std::runtime_error("invalid Redis response conversion");
}
void NormalizeNil(brpc::RedisReply& reply) {
  if (reply.type()==brpc::REDIS_REPLY_NIL) reply.SetNullString();
  else if (reply.is_array() && !reply.is_nil())
    for (std::size_t i=0;i<reply.size();++i) NormalizeNil(reply[i]);
}
void Append(const brpc::RedisReply& reply,butil::IOBufAppender* append) {
  butil::Arena arena; brpc::RedisReply copy(&arena); copy.CopyFromDifferentArena(reply);
  NormalizeNil(copy);
  if (!copy.SerializeTo(append)) throw std::runtime_error("Redis reply conversion failed");
}
void Single(const brpc::RedisReply& reply, brpc::RedisResponse* response) {
  butil::IOBufAppender append;
  Append(reply,&append);
  Parse(append.buf(), response);
}
Status Failure(brpc::RedisResponse* response, Status::Code code, const std::string& text) {
  if (response) { butil::IOBuf buffer; buffer.append("-EPS_REDIS " + text + "\r\n"); Parse(std::move(buffer), response); }
  return {code, "EPS_REDIS " + text};
}
Status Replies(const brpc::RedisResponse& response) {
  for (int i=0;i<response.reply_size();++i)
    if (response.reply(i).is_error()) return {Status::server_error,"Redis command returned an error"};
  return {};
}
}
namespace rsdk::client {
struct Client::Impl {
  // Authenticator outlives the channel that borrows it.
  brpc::policy::RedisAuthenticator auth;
  brpc::Channel channel;
  AccessOptions options;
  Impl(Endpoint endpoint, AccessOptions o) : auth(endpoint.password,endpoint.database), options(std::move(o)) {
    if (options._cname.empty() || options.timeout_ms<=0 || options.timeout_ms>300000 || endpoint.address.empty() ||
        endpoint.database<0 || endpoint.database>65535) throw std::invalid_argument("invalid Redis endpoint/options");
    brpc::ChannelOptions settings; settings.protocol="redis"; settings.connection_type="pooled";
    settings.timeout_ms=options.timeout_ms; settings.connect_timeout_ms=std::min(options.timeout_ms,1000);
    settings.max_retry=0; settings.auth=&auth;
    if (channel.Init(endpoint.address.c_str(),&settings)!=0) throw std::invalid_argument("cannot initialize Redis channel");
  }
  Status Call(const brpc::RedisRequest& request, brpc::RedisResponse* response) {
    if (!response) return {Status::invalid_argument,"null Redis response"};
    response->Clear();
    if (request.has_error() || request.command_size()<=0 || request.command_size()>static_cast<int>(kMaxRecords) ||
        request.ByteSizeLong()>kMaxRequestBytes)
      return Failure(response,Status::invalid_argument,"invalid or oversized request");
    brpc::Controller controller;
    channel.CallMethod(nullptr,&controller,&request,response,nullptr);
    if (controller.Failed()) return Failure(response,Status::unavailable,"transport failed; execution may be ambiguous");
    if (response->reply_size()!=request.command_size()) return Failure(response,Status::unavailable,"reply count mismatch");
    return Replies(*response);
  }
};
Client::Client(Endpoint endpoint, AccessOptions options) : impl_(new Impl(std::move(endpoint),std::move(options))) {}
Client::~Client() = default;
Status Client::exec(const brpc::RedisRequest& request, brpc::RedisResponse* response, google::protobuf::Closure* done) {
  brpc::ClosureGuard complete(done);
  auto status=impl_->Call(request,response);
  if (response && status.code()!=Status::unavailable && status.code()!=Status::invalid_argument &&
      impl_->options._merge_batch_as_array && response->reply_size()>1) {
    butil::IOBufAppender append; append.append("*"+std::to_string(response->reply_size())+"\r\n");
    {
      for (int i=0;i<response->reply_size();++i) {
        Append(response->reply(i),&append);
      }
    }
    Parse(append.buf(),response);
  }
  return status;
}
Status Client::Batch(const RecordList& records, std::vector<brpc::RedisResponse>* responses, bool write) {
  if (!responses) return {Status::invalid_argument,"null Redis batch response"};
  responses->clear();
  if (records.size()>kMaxRecords) return {Status::invalid_argument,"Redis batch record limit exceeded"};
  if (records.empty()) return {};
  // Validate the entire batch before constructing or sending any write command.
  std::size_t bytes=0;
  for (const auto& r:records) {
    if (r._ttl_s<0 || r._key.size()>kMaxRequestBytes || r._data.size()>kMaxRequestBytes ||
        r._key.size()+r._data.size()+128>kMaxRequestBytes-bytes) {
      responses->resize(records.size());
      for (auto& response:*responses) Failure(&response,Status::invalid_argument,"invalid or oversized batch");
      return {Status::invalid_argument,"invalid or oversized Redis batch"};
    }
    bytes+=r._key.size()+r._data.size()+128;
  }
  brpc::RedisRequest request;
  for (const auto& r:records) {
    const std::string ttl=std::to_string(r._ttl_s);
    std::vector<butil::StringPiece> parts{write?"SET":"GET",butil::StringPiece(r._key)};
    if (write) { parts.emplace_back(r._data); if (r._ttl_s) {parts.emplace_back("EX");parts.emplace_back(ttl);} }
    if (!request.AddCommandByComponents(parts.data(),parts.size()))
      return {Status::invalid_argument,"cannot encode Redis batch"};
  }
  brpc::RedisResponse pipeline;auto status=impl_->Call(request,&pipeline);
  responses->resize(records.size());
  if (status.code()==Status::unavailable || status.code()==Status::invalid_argument) {
    for (auto& response:*responses) Failure(&response,status.code(),"batch transport failed; execution may be ambiguous");
  } else for (std::size_t i=0;i<records.size();++i) Single(pipeline.reply(i),&responses->at(i));
  return status;
}
Status Client::mget(const RecordList& r,std::vector<brpc::RedisResponse>* out,google::protobuf::Closure* done) {
  brpc::ClosureGuard complete(done);return Batch(r,out,false);
}
Status Client::mset(const RecordList& r,std::vector<brpc::RedisResponse>* out,google::protobuf::Closure* done) {
  brpc::ClosureGuard complete(done);return Batch(r,out,true);
}
}  // namespace rsdk::client

namespace bilibili::redis {
struct RClientMgr::Impl {
  std::mutex mutex;
  bool initialized=false;
  std::string appid,zone,environment;
  std::map<std::string,rsdk::client::Endpoint> configured;
  std::set<std::string> enabled;
};
RClientMgr::RClientMgr() : impl_(new Impl) {}
RClientMgr::~RClientMgr() = default;
rsdk::Status RClientMgr::init(const std::string& appid,const std::string& zone,const std::string& environment) {
  std::lock_guard lock(impl_->mutex);
  if (appid.empty() || zone.empty() || environment.empty()) return {Status::invalid_argument,"Redis manager identity is empty"};
  if (impl_->initialized) return impl_->appid==appid && impl_->zone==zone && impl_->environment==environment ?
    Status{} : Status{Status::invalid_argument,"Redis manager is already initialized for another identity"};
  const char* path=std::getenv("EPS_REDIS_CONFIG");
  if (!path || !*path) return {Status::invalid_argument,"EPS_REDIS_CONFIG is required"};
  std::map<std::string,rsdk::client::Endpoint> endpoints;
  try {
    std::ifstream file(path,std::ios::binary); if (!file) throw std::runtime_error("configuration unavailable");
    file.seekg(0,std::ios::end); const auto bytes=file.tellg();
    if (bytes<=0 || bytes>(1<<20)) throw std::runtime_error("invalid configuration size");
    file.seekg(0);const auto config=nlohmann::json::parse(file);
    const auto& clusters=config.at("clusters");
    if (!clusters.is_object() || clusters.empty() || clusters.size()>256) throw std::runtime_error("invalid clusters");
    for (auto it=clusters.begin();it!=clusters.end();++it) {
      if (it.key().empty() || it.key().size()>256) throw std::runtime_error("invalid name");
      const auto& c=it.value();rsdk::client::Endpoint endpoint{c.at("endpoint").get<std::string>(),{},c.value("database",0)};
      if (endpoint.address.empty() || endpoint.address.size()>1024 || endpoint.database<0 || endpoint.database>65535)
        throw std::runtime_error("invalid endpoint");
      if (c.contains("password_env")) {
        const auto name=c.at("password_env").get<std::string>();const char* value=name.empty()?nullptr:std::getenv(name.c_str());
        if (!value || !*value) throw std::runtime_error("configured password is unavailable");
        endpoint.password=value;
      }
      endpoints.emplace(it.key(),std::move(endpoint));
    }
  } catch (const std::exception&) {
    return {Status::invalid_argument,"invalid or unreadable Redis configuration or missing password environment"};
  }
  impl_->configured=std::move(endpoints);impl_->appid=appid;impl_->zone=zone;impl_->environment=environment;
  impl_->initialized=true;return {};
}
rsdk::Status RClientMgr::init(prometheus::Registry*,const std::string& a,const std::string& z,const std::string& e) {
  // Registration belongs to the retained wrapper; this overload preserves its
  // source-level call shape. No private SDK metric names are synthesized.
  return init(a,z,e);
}
rsdk::Status RClientMgr::add_cluster(const std::string& name) {
  std::lock_guard lock(impl_->mutex);
  if (!impl_->initialized || impl_->configured.find(name)==impl_->configured.end()) return {Status::invalid_argument,"unknown Redis cluster or uninitialized manager"};
  impl_->enabled.insert(name);return {};
}
std::shared_ptr<rsdk::client::Client> RClientMgr::new_client(const rsdk::client::AccessOptions& options) {
  std::lock_guard lock(impl_->mutex);
  if (impl_->enabled.find(options._cname)==impl_->enabled.end()) return {};
  try {return std::make_shared<rsdk::client::Client>(impl_->configured.at(options._cname),options);}
  catch (const std::invalid_argument&) {return {};}
}
std::shared_ptr<RClientMgr> new_rc_mgr() { return std::make_shared<RClientMgr>(); }
}  // namespace bilibili::redis
