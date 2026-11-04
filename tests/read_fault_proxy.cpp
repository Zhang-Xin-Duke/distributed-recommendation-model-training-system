// Test-only BRPC bridge: forward real parameter/gradient RPCs and inject an
// uncertain read result before/after execution, or delay the original Lookup.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include <brpc/server.h>
#include <brpc/channel.h>
#include <brpc/controller.h>
#include <butil/endpoint.h>
#include <nlohmann/json.hpp>
#include "embedding_service.pb.h"
namespace wire=eps::v1;
struct State {
  brpc::Channel channel;std::string method,stage,report;unsigned learner=0,count=1,delay_ms=0;std::uint64_t sequence=3;
  std::mutex mutex;std::vector<std::thread> timers;
  unsigned injected=0,completed_faults=0,release_calls=0,cancellations=0;int delayed_status=-1;
  void Write(){if(report.empty())return;nlohmann::json j={{"injected",injected},{"completed_faults",completed_faults},
    {"target_release_calls",release_calls},{"identity_cancellations",cancellations},{"delayed_upstream_status",delayed_status}};
    auto tmp=report+".tmp";{std::ofstream f(tmp);f<<j.dump(2)<<'\n';}std::filesystem::rename(tmp,report);}
  ~State(){for(auto& timer:timers)if(timer.joinable())timer.join();}
  struct Call final:google::protobuf::Closure {
    State& owner;const google::protobuf::MethodDescriptor* method;const google::protobuf::Message* request;
    google::protobuf::Message* response;brpc::Controller* incoming;google::protobuf::Closure* done;
    brpc::Controller forwarded;bool fault;
    Call(State& s,const google::protobuf::MethodDescriptor* m,const google::protobuf::Message* q,google::protobuf::Message* r,
        brpc::Controller* c,google::protobuf::Closure* d,bool f):owner(s),method(m),request(q),response(r),incoming(c),done(d),fault(f){}
    void Send(){forwarded.set_timeout_ms(15000);forwarded.request_attachment().append(incoming->request_attachment());
      owner.channel.CallMethod(method,&forwarded,request,response,this);}
    wire::Status* Status(){if(method->name()=="Lookup")return static_cast<wire::LookupReply*>(response)->mutable_status();
      if(method->name()=="Release")return static_cast<wire::Ack*>(response)->mutable_status();return nullptr;}
    void Run()override{
      std::unique_ptr<Call> self(this);brpc::ClosureGuard guard(done);
      if(forwarded.Failed()){incoming->SetFailed(forwarded.ErrorCode(),"%s",forwarded.ErrorText().c_str());return;}
      bool discard=false;
      if(fault){std::lock_guard lock(owner.mutex);++owner.completed_faults;
        if(owner.stage=="delay")owner.delayed_status=Status()->code();
        if(owner.stage=="after" && Status()->code()==wire::Status::OK){discard=true;response->Clear();auto* s=Status();
          s->set_code(wire::Status::UNAVAILABLE);s->set_message("injected lost successful reply");}
        owner.Write();}
      if(!discard)incoming->response_attachment().append(forwarded.response_attachment());
    }
  };
  void Forward(const google::protobuf::MethodDescriptor* m,google::protobuf::RpcController* raw,const google::protobuf::Message* q,
               google::protobuf::Message* r,google::protobuf::Closure* done){
    auto* c=static_cast<brpc::Controller*>(raw);bool fault=false;
    const wire::RequestIdentity* id=nullptr;
    if(m->name()=="Lookup")id=&static_cast<const wire::LookupRequest*>(q)->identity();
    if(m->name()=="Release")id=&static_cast<const wire::ReleaseRequest*>(q)->identity();
    if(id && id->learner()==learner && id->sequence()==sequence){std::lock_guard lock(mutex);
      if(m->name()=="Release"){++release_calls;if(static_cast<const wire::ReleaseRequest*>(q)->cancel_unseen())++cancellations;}
      if(m->name()==method && injected<count){++injected;fault=true;}Write();}
    auto call=std::make_unique<Call>(*this,m,q,r,c,done,fault);
    if(fault && stage=="before"){
      auto* status=call->Status();status->set_code(wire::Status::UNAVAILABLE);status->set_message("injected pre-dispatch read failure");
      {std::lock_guard lock(mutex);++completed_faults;Write();}done->Run();return;
    }
    if(fault && stage=="delay"){
      auto* pending=call.release();std::lock_guard lock(mutex);
      timers.emplace_back([pending,delay=delay_ms]{std::this_thread::sleep_for(std::chrono::milliseconds(delay));pending->Send();});return;
    }
    call.release()->Send();
  }
};
template<class Base>class Proxy final:public Base {
  State& state;
 public:explicit Proxy(State& s):state(s){}
  void CallMethod(const google::protobuf::MethodDescriptor* m,google::protobuf::RpcController* c,
                  const google::protobuf::Message* q,google::protobuf::Message* r,google::protobuf::Closure* done)override{
    state.Forward(m,c,q,r,done);
  }
};
int main(int argc,char** argv){try{
 State state;std::string upstream;
 for(int i=1;i<argc;++i){std::string arg=argv[i];if(i+1==argc)throw std::runtime_error("missing proxy option value");auto value=std::string(argv[++i]);
  if(arg=="--upstream")upstream=value;else if(arg=="--method")state.method=value;else if(arg=="--stage")state.stage=value;
  else if(arg=="--report")state.report=value;else if(arg=="--count")state.count=std::stoul(value);
  else if(arg=="--learner")state.learner=std::stoul(value);else if(arg=="--sequence")state.sequence=std::stoull(value);
  else if(arg=="--delay-ms")state.delay_ms=std::stoul(value);else throw std::runtime_error("unknown proxy option");}
 if(upstream.empty() || (state.method!="Lookup" && state.method!="Release") ||
    (state.stage!="before" && state.stage!="after" && state.stage!="delay") || state.count>3 || state.delay_ms>10000)
  throw std::runtime_error("invalid read fault proxy options");
 brpc::ChannelOptions options;options.protocol="baidu_std";options.max_retry=0;options.timeout_ms=15000;
 if(state.channel.Init(upstream.c_str(),&options))throw std::runtime_error("proxy upstream unavailable");
 Proxy<wire::EmbeddingService> parameters(state);Proxy<wire::SynchronousGradientService> gradients(state);brpc::Server server;
 if(server.AddService(&parameters,brpc::SERVER_DOESNT_OWN_SERVICE) || server.AddService(&gradients,brpc::SERVER_DOESNT_OWN_SERVICE))throw std::runtime_error("proxy service registration failed");
 brpc::ServerOptions settings;settings.max_concurrency=0;if(server.Start("127.0.0.1:0",&settings))throw std::runtime_error("proxy start failed");
 std::cout<<"READY "<<butil::endpoint2str(server.listen_address()).c_str()<<std::endl;
 while(!brpc::IsAskedToQuit())std::this_thread::sleep_for(std::chrono::milliseconds(50));server.Stop(0);server.Join();
 {std::lock_guard lock(state.mutex);state.Write();}
}catch(const std::exception& e){std::cerr<<"FAIL read fault proxy: "<<e.what()<<'\n';return 1;}}
