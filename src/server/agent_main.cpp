#include <iostream>
#include <string>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include "eps/agent_runtime.hpp"
#include "transport/brpc_agent_server.hpp"
#ifdef EPS_HAS_CUDA
#include "eps/cuda_cache_backend.hpp"
#endif

int main(int argc,char** argv){
  try{
    std::string run,listen="127.0.0.1:9200",backend="host";
    std::uint64_t bytes=64ULL<<20,rows=4096;std::uint32_t agent=0;int device=0;
    for(int i=1;i<argc;++i){
      std::string arg=argv[i];if(i+1>=argc)throw std::runtime_error("missing option value");
      if(arg=="--run")run=argv[++i];else if(arg=="--listen")listen=argv[++i];
      else if(arg=="--agent")agent=std::stoul(argv[++i]);
      else if(arg=="--state-bytes")bytes=std::stoull(argv[++i]);
      else if(arg=="--rows-per-table")rows=std::stoull(argv[++i]);
      else if(arg=="--device")device=std::stoi(argv[++i]);
      else if(arg=="--backend")backend=argv[++i];else throw std::runtime_error("unknown agent option");
    }
    std::shared_ptr<eps::CacheBackend> cache;
    if(backend=="host")cache=std::make_shared<eps::HostCacheBackend>(agent,bytes);
#ifdef EPS_HAS_CUDA
    else if(backend=="cuda"){
      eps::CudaCacheOptions options;options.agent=agent;options.device=device;
      options.state_bytes=bytes;options.rows_per_table=rows;
      cache=std::make_shared<eps::CudaCacheBackend>(options);
    }
#endif
    else throw std::runtime_error("requested backend is not available in this build");
    std::cerr<<"CACHE_BACKEND="<<backend<<'\n';
    eps::AgentRuntime runtime(run,cache);eps::transport::BrpcAgentServer service(runtime);brpc::Server server;
    if(server.AddService(&service,brpc::SERVER_DOESNT_OWN_SERVICE)!=0)throw std::runtime_error("cannot register cache agent");
    brpc::ServerOptions options;options.max_concurrency=128;
    if(server.Start(listen.c_str(),&options)!=0)throw std::runtime_error("cannot start cache agent");
    std::cout<<"READY "<<butil::endpoint2str(server.listen_address()).c_str()<<std::endl;
    server.RunUntilAskedToQuit();return 0;
  }catch(const std::exception& e){std::cerr<<"eps_cache_agent: "<<e.what()<<'\n';return 1;}
}
