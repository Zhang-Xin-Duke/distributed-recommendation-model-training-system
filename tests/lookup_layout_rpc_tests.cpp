#include "eps/brpc_client.hpp"
#include "transport/brpc_server.hpp"
#include "transport/codec.hpp"
#include <brpc/server.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/endpoint.h>
#include <atomic>
#include <iostream>
#include <thread>
using namespace eps;
using namespace std::chrono_literals;
namespace {
void Check(bool ok,const char* why) { if(!ok) throw std::runtime_error(why); }
template<class T> T Take(std::future<Result<T>> f) {
  Check(f.wait_for(8s)==std::future_status::ready,"layout RPC did not complete");
  auto r=f.get(); if(auto* e=std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
struct Harness {
  CpuParameterService parameters{[] { RuntimeOptions o;o.run_id="layouts";return o;}()};
  LookupLayoutCacheOptions options;
  std::unique_ptr<transport::BrpcServer> rpc;
  std::unique_ptr<brpc::Server> server;
  std::string address;
  explicit Harness(LookupLayoutCacheOptions cache = {}) : options(cache) {
    Take(parameters.CreateTable({1,2,1,1,{.02f,.9f,.999f,1e-8f,.9f,.999f},0,0,42}));
    Start();
  }
  void Start() {
    rpc=std::make_unique<transport::BrpcServer>(parameters,0,options); server=std::make_unique<brpc::Server>();
    Check(server->AddService(rpc.get(),brpc::SERVER_DOESNT_OWN_SERVICE)==0,"register layout service");
    Check(server->Start(address.empty()?"127.0.0.1:0":address.c_str(),nullptr)==0,"start layout service");
    address=butil::endpoint2str(server->listen_address()).c_str();
  }
  void RestartRegistry() { server->Stop(0);server->Join();server.reset();rpc.reset();Start(); }
  ~Harness() {server->Stop(0);server->Join();}
};
LookupRequest Query(std::uint32_t learner,std::uint64_t sequence,Key key=1,bool read_only=true) {
  return {{"layouts",learner,sequence,Phase::lookup},1,read_only,
    {{1,0,Pooling::sequence,{key,key,3},{0,3},{3}}}};
}
void Release(BrpcParameterService& c,const LookupRequest& q,const LookupReply& r) {
  auto id=q.id;id.phase=Phase::release;Take(c.Release({id,r.ticket}));
}
void FreshValuesAndRestart() {
  Harness h;BrpcParameterService c(h.address);c.SetLookupLayoutCache(true);
  auto q=Query(0,0);auto before=Take(c.Lookup(q));Release(c,q,before);
  BrpcParameterService writer(h.address);
  auto update=Query(1,0,1,false);auto t=Take(writer.Lookup(update));
  auto id=update.id;id.phase=Phase::push;
  Take(writer.Push({id,t.ticket,{{0,{3,2},{1,1,1,1,1,1}}}}));
  q=Query(0,1);auto after=Take(c.Lookup(q));Release(c,q,after);
  auto baseline=Query(2,0);auto expected=Take(writer.Lookup(baseline));Release(writer,baseline,expected);
  Check(after.cpu_contributions[0].values==expected.cpu_contributions[0].values &&
    after.cpu_contributions[0].values!=before.cpu_contributions[0].values,"layout cache returned stale parameter values");
  auto stats=c.LookupTransport();Check(stats.full_requests==1 && stats.token_requests==1,"second lookup did not reuse layout");
  Check(Take(c.LookupLayoutStats()).hits==1,"server did not resolve cached layout");
  h.RestartRegistry();
  const auto deadline=std::chrono::steady_clock::now()+6s;
  while (std::holds_alternative<Error>(c.Stats().get())) {
    Check(std::chrono::steady_clock::now()<deadline,"same endpoint did not reconnect after registry restart");
    std::this_thread::sleep_for(10ms);
  }
  q=Query(0,2);auto restarted=Take(c.Lookup(q));Release(c,q,restarted);
  Check(restarted.cpu_contributions[0].values==expected.cpu_contributions[0].values,"registry restart changed model state");
  stats=c.LookupTransport();Check(stats.fallback_requests==1 && stats.full_requests==2 && stats.token_requests==2,
    "registry restart did not perform one same-request fallback");
  Check(Take(c.Stats()).live_tickets==0 && Take(c.Stats()).lookups==5,"metadata miss entered parameter actor or leaked ticket");
}
void ExpiryEvictionAndScope() {
  Harness h({1,1<<20,200ms});BrpcParameterService c(h.address);c.SetLookupLayoutCache(true);
  for(unsigned i=0;i<3;++i) {auto q=Query(0,i,i==1?2:1);auto r=Take(c.Lookup(q));Release(c,q,r);}
  Check(c.LookupTransport().fallback_requests==1,"LRU miss did not fall back");
  std::this_thread::sleep_for(250ms);
  auto q=Query(0,3);auto r=Take(c.Lookup(q));Release(c,q,r);
  Check(c.LookupTransport().fallback_requests==2,"expired server layout did not fall back");
  Check(Take(c.LookupLayoutStats()).expirations>=1,"server TTL did not expire layout");
  brpc::Channel channel;Check(channel.Init(h.address.c_str(),nullptr)==0,"init raw layout channel");
  v1::EmbeddingService_Stub stub(&channel);brpc::Controller register_call;
  v1::LookupRequest full;v1::LookupReply registered;
  auto raw=Query(4,0);transport::Encode(raw.id,full.mutable_identity());full.set_partition_epoch(1);full.set_read_only(true);
  full.set_register_layout(true);
  for(const auto& f:raw.features)transport::Encode(f,full.add_features(),register_call.request_attachment());
  stub.Lookup(&register_call,&full,&registered,nullptr);
  Check(!register_call.Failed() && !transport::GetError(registered.status()) && !registered.layout_token().empty(),"raw registration failed");
  Release(c,raw,LookupReply{registered.ticket(),{}, {}});
  for(unsigned field=0;field<3;++field) {
    brpc::Controller ctl;v1::LookupRequest token;v1::LookupReply reply;
    auto identity=raw.id;++identity.sequence;
    if(field==0)identity.run="other";if(field==1)++identity.learner;
    transport::Encode(identity,token.mutable_identity());token.set_partition_epoch(field==2?2:1);
    token.set_read_only(true);token.set_layout_token(registered.layout_token());
    stub.Lookup(&ctl,&token,&reply,nullptr);
    const auto error=transport::GetError(reply.status());
    Check(!ctl.Failed() && error && error->code==Error::Code::stale_generation,"cached token crossed ownership scope");
  }
  Check(Take(c.Stats()).live_tickets==0,"scope rejection allocated tickets");
}
void ConcurrentQueries() {
  Harness h;BrpcParameterService c(h.address);c.SetLookupLayoutCache(true);
  std::atomic<unsigned> failures=0;std::vector<std::jthread> threads;
  for(unsigned w=0;w<8;++w)threads.emplace_back([&,w] {
    try {for(unsigned i=0;i<20;++i) {auto q=Query(w,i,w+10);auto r=Take(c.Lookup(q));Release(c,q,r);}}
    catch(...) {++failures;}
  });
  threads.clear();Check(!failures && Take(c.Stats()).lookups==160 && !Take(c.Stats()).live_tickets,
                       "concurrent cached lookup changed request/ticket semantics");
  Check(c.LookupTransport().token_requests>=100,"concurrent lookups did not reuse registered metadata");
}
void DisabledRegistrationAndLegacyPeer() {
  Harness disabled({0,0,1000ms}); BrpcParameterService c(disabled.address); c.SetLookupLayoutCache(true);
  for (unsigned i=0;i<3;++i) {auto q=Query(0,i);auto r=Take(c.Lookup(q));Release(c,q,r);}
  Check(c.LookupTransport().full_requests==3 && c.LookupTransport().token_requests==0 &&
    Take(c.LookupLayoutStats()).rejected==3,"disabled server cache affected normal reads");
  struct Legacy final : v1::EmbeddingService {
    transport::BrpcServer& delegate;
    explicit Legacy(transport::BrpcServer& d):delegate(d){}
    void Lookup(google::protobuf::RpcController* c,const v1::LookupRequest* q,v1::LookupReply* r,
                google::protobuf::Closure* done) override {
      auto old=*q;old.clear_register_layout();old.clear_layout_token();delegate.Lookup(c,&old,r,done);
    }
    void Release(google::protobuf::RpcController* c,const v1::ReleaseRequest* q,v1::Ack* r,
                 google::protobuf::Closure* done) override {delegate.Release(c,q,r,done);}
  } legacy(*disabled.rpc);
  brpc::Server server;Check(server.AddService(&legacy,brpc::SERVER_DOESNT_OWN_SERVICE)==0,"register old protocol shim");
  Check(server.Start("127.0.0.1:0",nullptr)==0,"start old protocol shim");
  BrpcParameterService compatible(butil::endpoint2str(server.listen_address()).c_str());compatible.SetLookupLayoutCache(true);
  for(unsigned i=0;i<3;++i) {auto q=Query(10,i);auto r=Take(compatible.Lookup(q));Release(compatible,q,r);}
  Check(compatible.LookupTransport().full_requests==3 && compatible.LookupTransport().token_requests==0,
        "server without metadata-token support received compressed requests");
  server.Stop(0);server.Join();
}
}
int main() {
  try {FreshValuesAndRestart();ExpiryEvictionAndScope();ConcurrentQueries();DisabledRegistrationAndLegacyPeer();
    std::cout<<"PASS reusable layout RPC: fresh weights, volatile registry restart, one fallback, TTL/LRU, scope and concurrent tickets\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
