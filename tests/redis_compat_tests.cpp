#include "bili_redis/bili_redis_client.h"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
using rsdk::client::Client;
using rsdk::client::RecordList;
namespace {
void Check(bool value,const char* why) {if(!value)throw std::runtime_error(why);}
std::shared_ptr<Client> Open(const std::shared_ptr<bilibili::redis::RClientMgr>& manager,const std::string& name,bool merge=false,int timeout=1000) {
  Check(manager->add_cluster(name).ok(),"cluster initialization failed");
  auto client=manager->new_client({name,timeout,merge});Check(bool(client),"client initialization failed");return client;
}
brpc::RedisResponse Call(Client& client,const std::vector<std::string>& parts,bool success=true) {
  std::vector<butil::StringPiece> args;for(const auto& p:parts)args.emplace_back(p);
  brpc::RedisRequest request;Check(request.AddCommandByComponents(args.data(),args.size()),"command encoding failed");
  brpc::RedisResponse response;Check(client.exec(request,&response).ok()==success,"unexpected Redis command status");return response;
}
struct Done : google::protobuf::Closure {
  unsigned calls=0;
  void Run()override{++calls;}
};
void Basic(const std::shared_ptr<bilibili::redis::RClientMgr>& manager) {
  Check(!manager->new_client({"unknown",1000,false}),"unknown cluster became a client");
  Check(!manager->add_cluster("unknown").ok(),"unknown cluster enabled");
  auto client=Open(manager,"plain");
  std::string key("key\0\r\n ",8),value("value\0\r\n '%",11);
  auto set=Call(*client,{"SET",key,value});Check(set.reply(0).data()=="OK","binary SET failed");
  Check(Call(*client,{"GET",key}).reply(0).data().as_string()==value,"binary value changed");
  std::string large(65537,'x');large[32768]='\0';
  std::vector<brpc::RedisResponse> large_replies;
  Check(client->mset({{"large",large,0}},&large_replies).ok(),"large binary write failed");
  Check(client->mget({{"large","",0},{key,"",0}},&large_replies).ok() &&
        large_replies[0].reply(0).data().as_string()==large && large_replies[1].reply(0).data().as_string()==value,
        "multi-buffer reply conversion lost binary data");
  Call(*client,{"SET","empty",""});
  RecordList records{{key,"",0},{"absent","",0},{"empty","",0},{key,"",0}};
  std::vector<brpc::RedisResponse> responses;Check(client->mget(records,&responses).ok(),"batch read failed");
  Check(responses.size()==4 && responses[0].reply(0).data().as_string()==value && responses[1].reply(0).is_nil() &&
        responses[2].reply(0).is_string() && responses[2].reply(0).size()==0 && responses[3].reply(0).data().as_string()==value,
        "batch order, binary, nil or empty semantics changed");
  RecordList writes{{"expires","short",1},{"persistent","long",0},{"ordered","first",0},{"ordered","last",0}};
  Check(client->mset(writes,&responses).ok() && responses.size()==4,"batch writes failed");
  Check(Call(*client,{"GET","ordered"}).reply(0).data()=="last","duplicate-Key writes lost order");
  const auto ttl=Call(*client,{"TTL","expires"}).reply(0).integer();Check(ttl>=0 && ttl<=1,"per-record TTL missing");
  Check(Call(*client,{"TTL","persistent"}).reply(0).integer()==-1,"zero TTL set expiration");
  std::this_thread::sleep_for(std::chrono::milliseconds(1150));
  Check(Call(*client,{"GET","expires"}).reply(0).is_nil(),"TTL did not expire");
  RecordList invalid{{"must-not-write","v",0},{"bad-ttl","v",-1}};
  Check(!client->mset(invalid,&responses).ok() && responses.size()==2 && responses[0].reply(0).is_error(),"invalid batch accepted");
  Check(Call(*client,{"GET","must-not-write"}).reply(0).is_nil(),"invalid batch partially executed");
  Check(!client->mget(RecordList(4097),&responses).ok(),"batch count limit ignored");
  Check(!client->mset({{"oversized",std::string((8<<20)+1,'x'),0}},&responses).ok(),"batch byte limit ignored");
  Call(*client,{"LPUSH","list","v"});
  Check(!client->mget({{"persistent","",0},{"list","",0},{"absent","",0}},&responses).ok(),"batch type error was hidden");
  Check(responses.size()==3 && responses[0].reply(0).data()=="long" && responses[1].reply(0).is_error() &&
        responses[2].reply(0).is_nil(),"batch error changed another slot");
  brpc::RedisRequest pipeline;pipeline.AddCommand("GET persistent");pipeline.AddCommand("INCR counter");pipeline.AddCommand("MGET persistent absent");
  brpc::RedisResponse raw;Check(client->exec(pipeline,&raw).ok() && raw.reply_size()==3 && raw.reply(1).integer()==1 &&
        raw.reply(2).is_array() && raw.reply(2)[1].is_nil(),"pipeline replies changed");
  auto merged=Open(manager,"plain",true);brpc::RedisResponse array;
  Check(merged->exec(pipeline,&array).ok() && array.reply_size()==1 && array.reply(0).is_array() && array.reply(0).size()==3 &&
        array.reply(0)[1].integer()==2 && array.reply(0)[2][1].is_nil(),"merged pipeline changed replies");
  brpc::RedisRequest mixed;mixed.AddCommand("GET persistent");mixed.AddCommand("GET list");
  Check(!merged->exec(mixed,&array).ok() && array.reply(0).is_array() && array.reply(0)[0].data()=="long" &&
        array.reply(0)[1].is_error(),"merged pipeline concealed per-command error");
  Check(Call(*client,{"UNKNOWN_TEST_COMMAND"},false).reply(0).is_error(),"server error was hidden");
  brpc::RedisRequest ping;ping.AddCommand("PING");Done done;
  Check(client->exec(ping,&raw,&done).ok() && done.calls==1,"completion callback count changed");
  brpc::RedisRequest invalid_request;Done invalid_done;
  Check(!client->exec(invalid_request,&raw,&invalid_done).ok() && raw.reply(0).is_error() && invalid_done.calls==1,
        "invalid request completion failed");
  Done empty;Check(client->mget({},&responses,&empty).ok() && responses.empty() && empty.calls==1,"empty batch callback failed");
  Check(!client->exec(ping,nullptr).ok(),"null response accepted");
  auto other=Open(manager,"database-one");Call(*other,{"SET","database-only","one"});
  Check(Call(*client,{"GET","database-only"}).reply(0).is_nil(),"database selection leaked across channels");
  Check(Call(*other,{"GET","database-only"}).reply(0).data()=="one","database SELECT lost");
  auto auth=Open(manager,"authenticated");Call(*auth,{"SET",key,value});
  Check(Call(*auth,{"GET",key}).reply(0).data().as_string()==value,"authenticated binary roundtrip failed");
  auto wrong=Open(manager,"wrong-password");Check(Call(*wrong,{"GET",key},false).reply(0).is_error(),"bad authentication hidden");
  std::atomic<unsigned> failures=0;
  std::vector<std::thread> threads;
  for(unsigned w=0;w<8;++w)threads.emplace_back([&,w]{try{
    for(unsigned i=0;i<24;++i) {const auto k="parallel-"+std::to_string(w),v=std::to_string(i);
      Call(*client,{"SET",k,v});Check(Call(*client,{"GET",k}).reply(0).data()==v,"concurrent value changed");}
  }catch(...){++failures;}});
  for(auto& t:threads)t.join();Check(failures==0,"concurrent shared client failed");
  Call(*client,{"CLIENT","PAUSE","350"});
  auto short_timeout=Open(manager,"plain",false,50);
  Check(Call(*short_timeout,{"GET","persistent"},false).reply(0).is_error(),"timeout became success or nil");
  std::this_thread::sleep_for(std::chrono::milliseconds(450));
  Check(Call(*client,{"GET","persistent"}).reply(0).data()=="long","client did not recover after pause");
  Call(*client,{"SET","lost-reply-counter","0"});
}
}
int main(int argc,char** argv) {
  try {
    Check(argc==2,"expected mode");auto manager=bilibili::redis::new_rc_mgr();
    const std::string mode=argv[1];
    if(mode=="config-rejected") {
      Check(!manager->init("training","local","test").ok(),"invalid configuration accepted");
      std::cout<<nlohmann::json{{"test","public_redis_sdk"},{"mode",mode},{"passed",true}}<<'\n';return 0;
    }
    Check(!manager->init("","local","test").ok(),"empty identity accepted");
    Check(manager->init("training","local","test").ok(),"manager configuration failed");
    Check(manager->init(static_cast<prometheus::Registry*>(nullptr),"training","local","test").ok(),"idempotent init failed");
    Check(!manager->init("other","local","test").ok(),"manager identity changed");
    if(mode=="basic")Basic(manager);
    else if(mode=="lost-reply") {
      auto client=Open(manager,"fault",false,500);
      Check(Call(*client,{"INCR","lost-reply-counter"},false).reply(0).is_error(),"lost reply became success");
    } else if(mode=="inspect-counter") {
      auto client=Open(manager,"plain");Check(Call(*client,{"GET","lost-reply-counter"}).reply(0).data()=="1","ambiguous non-idempotent write was replayed");
    } else if(mode=="unavailable") {
      auto client=Open(manager,"plain",false,100);
      std::vector<brpc::RedisResponse> responses;
      Check(!client->mget({{"a","",0},{"b","",0}},&responses).ok() && responses.size()==2 &&
            responses[0].reply(0).is_error() && responses[1].reply(0).is_error(),"connection failure became nil or success");
    } else throw std::runtime_error("unknown mode");
    std::cout<<nlohmann::json{{"test","public_redis_sdk"},{"mode",mode},{"passed",true}}<<'\n';
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
