#include "eps/runtime.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace eps;
namespace {
void Check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class T>T Take(std::future<Result<T>> f){auto r=f.get();if(auto* e=std::get_if<Error>(&r))throw std::runtime_error(e->message);return std::get<T>(std::move(r));}
template<class T>void Reject(std::future<Result<T>> f){Check(std::holds_alternative<Error>(f.get()),"expected restore rejection");}
struct Temp{std::filesystem::path path;Temp(){char p[]="/tmp/eps-provenance-XXXXXX";auto* v=mkdtemp(p);Check(v,"mkdtemp");path=v;}~Temp(){std::error_code e;std::filesystem::remove_all(path,e);}};
void Lifecycle(){
 Temp temp;RuntimeOptions o;o.run_id="source";o.snapshot_directory=temp.path;
 CpuParameterService original(o);TableDefinition table{1,2,1,1,{.01f,.9f,.999f,1e-8f,.9f,.999f},.25f,.25f,42};
 Take(original.CreateTable(table));auto q=Take(original.Lookup({{"source",0,0,Phase::lookup},1,false,{{1,0,Pooling::sum,{1},{0,1},{1}}}}));
 Take(original.Push({{"source",0,0,Phase::push},q.ticket,{{0,{1,2},{.1f,-.2f}}}}));
 auto saved=Take(original.DrainAndFlush("saved"));const auto file=saved.state_files.at(0);
 o.run_id="restarted";CpuParameterService loaded(o);auto fresh=Take(loaded.RestoredFrom());
 Check(!fresh.loaded && !fresh.pristine && fresh.run=="restarted" && fresh.shard_count==1,"fresh restore provenance invalid");
 Take(loaded.Restore(file));auto identity=Take(loaded.RestoredFrom());
 Check(identity.loaded && identity.pristine && identity.run=="restarted" && identity.source_run=="source" &&
   identity.bytes==std::filesystem::file_size(file) && identity.shard==0 && identity.shard_count==1 && identity.partition_epoch==1,"loaded provenance invalid");
 std::ifstream stream(file,std::ios::binary);stream.seekg(-8,std::ios::end);std::uint64_t checksum=0;for(unsigned i=0;i<8;++i)checksum|=std::uint64_t(static_cast<unsigned char>(stream.get()))<<(8*i);
 Check(identity.checksum==checksum,"provenance is not the validated snapshot checksum");
 Take(loaded.CreateTable(table));Check(Take(loaded.RestoredFrom()).pristine,"duplicate definition dirtied restored state");
 Reject(loaded.Restore(file));
 auto read=Take(loaded.Lookup({{"restarted",0,0,Phase::lookup},1,true,{{1,0,Pooling::sum,{1},{0,1},{1}}}}));
 Take(loaded.Release({{"restarted",0,0,Phase::release},read.ticket}));
 Check(!Take(loaded.RestoredFrom()).pristine,"parameter admission did not invalidate restore boundary");
 o.run_id="schema-change";CpuParameterService schema(o);Take(schema.Restore(file));table.id=2;Take(schema.CreateTable(table));
 Check(!Take(schema.RestoredFrom()).pristine,"new table did not invalidate restored schema");
 auto corrupt=temp.path/"corrupt";std::filesystem::copy_file(file,corrupt);{std::fstream s(corrupt,std::ios::in|std::ios::out|std::ios::binary);s.seekg(-1,std::ios::end);auto byte=s.get();s.seekp(-1,std::ios::end);s.put(char(byte^1));}
 o.run_id="failed";CpuParameterService failed(o);Reject(failed.Restore(corrupt));Check(!Take(failed.RestoredFrom()).loaded,"failed load published provenance");
 Take(failed.Restore(file));Check(Take(failed.RestoredFrom()).pristine,"failed attempt poisoned fresh restore");
 o.run_id="empty-source";CpuParameterService empty(o);auto empty_file=Take(empty.DrainAndFlush("empty")).state_files.at(0);
 o.run_id="empty-restored";CpuParameterService empty_loaded(o);Take(empty_loaded.Restore(empty_file));
 Check(Take(empty_loaded.RestoredFrom()).loaded,"empty snapshot lost provenance");Reject(empty_loaded.Restore(empty_file));
}
}
int main(){try{Lifecycle();std::cout<<"PASS validated restore provenance, pristine admission boundary and failed-load publication\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
