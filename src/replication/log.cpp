#include "eps/replica_log.hpp"
#include "common/error.hpp"
#include "store/snapshot.hpp"
#include <array>
#include <charconv>
#include <cerrno>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace eps {
namespace {
namespace fs=std::filesystem;
using detail::Require;
std::uint64_t Hash(std::string_view bytes) {
  std::uint64_t hash=14695981039346656037ULL;
  for(unsigned char c:bytes)hash=(hash^c)*1099511628211ULL;return hash;
}
void U(std::string& out,std::uint64_t n) {for(unsigned i=0;i<8;++i)out.push_back(char(n>>(8*i)));}
std::uint64_t U(std::string_view data) {
  Require(data.size()>=8,"truncated replica metadata");std::uint64_t n=0;
  for(unsigned i=0;i<8;++i)n|=std::uint64_t(static_cast<unsigned char>(data[i]))<<(8*i);return n;
}
struct File {
  int fd;
  File(const fs::path& p,int flags):fd(::open(p.c_str(),flags,0600)) {Require(fd>=0,"cannot open replica file",Error::Code::unavailable);}
  ~File(){::close(fd);}
  std::uint64_t Size()const {struct stat s{};Require(::fstat(fd,&s)==0 && s.st_size>=0,"cannot stat replica file");return s.st_size;}
  void Sync(){Require(::fsync(fd)==0,"cannot fsync replica file",Error::Code::unavailable);}
  void Write(std::uint64_t offset,std::string_view bytes) {
    while(!bytes.empty()) {
      const auto n=::pwrite(fd,bytes.data(),bytes.size(),offset);
      if(n<0 && errno==EINTR)continue;
      Require(n>0,"replica write failed",Error::Code::unavailable);offset+=n;bytes.remove_prefix(n);
    }
  }
  std::string Read(std::uint64_t offset,std::size_t count)const {
    std::string data(count,'\0');std::size_t at=0;
    while(at<count) {const auto n=::pread(fd,data.data()+at,count-at,offset+at);
      if(n<0 && errno==EINTR)continue;Require(n>0,"truncated replica file");at+=n;}
    return data;
  }
};
void SyncDirectory(const fs::path& p){File directory(p,O_RDONLY);directory.Sync();}
void Atomic(const fs::path& path,std::string bytes) {
  U(bytes,Hash(bytes));const auto pending=path.string()+".pending";
  {File file(pending,O_WRONLY|O_CREAT|O_TRUNC);file.Write(0,bytes);file.Sync();}
  Require(::rename(pending.c_str(),path.c_str())==0,"cannot publish replica metadata",Error::Code::unavailable);
  SyncDirectory(path.parent_path());
}
std::string Metadata(const fs::path& path) {
  File f(path,O_RDONLY);Require(f.Size()>=8 && f.Size()<=(2ULL<<20),"invalid replica metadata size");
  auto bytes=f.Read(0,f.Size());const auto body=bytes.size()-8;
  Require(Hash(std::string_view(bytes).substr(0,body))==U(std::string_view(bytes).substr(body)),"replica metadata checksum mismatch");
  bytes.resize(body);return bytes;
}
void Checksum(const fs::path& path) {
  File file(path,O_RDONLY);const auto size=file.Size();Require(size>=8,"truncated replica blob");
  std::uint64_t hash=14695981039346656037ULL;
  for(std::uint64_t offset=0;offset<size-8;) {
    const auto count=std::min<std::uint64_t>(1<<20,size-8-offset);const auto data=file.Read(offset,count);
    for(unsigned char c:data)hash=(hash^c)*1099511628211ULL;offset+=count;
  }
  Require(hash==U(file.Read(size-8,8)),"replica blob checksum mismatch");
}
}
struct ReplicaLog::Impl {
  ReplicaLogOptions options;
  std::unique_ptr<File> directory_lock;
  mutable std::mutex mutex;
  ReplicaHead head;
  explicit Impl(ReplicaLogOptions o):options(std::move(o)) {
    Require(!options.directory.empty() && !options.run.empty() && options.run.size()<=256 &&
      options.binding.size()<=(1ULL<<20) && options.shards && options.shards<=1024 && options.shard<options.shards &&
      options.epoch && options.base_round!=UINT64_MAX && options.record_bytes>=4096 && options.record_bytes<=(256ULL<<20) &&
      options.disk_bytes>=options.record_bytes,"invalid replica log options");
    options.directory=fs::absolute(options.directory);
    fs::create_directories(options.directory);SyncDirectory(options.directory.parent_path());
    directory_lock=std::make_unique<File>(options.directory/"lock",O_RDWR|O_CREAT|O_CLOEXEC);
    Require(::flock(directory_lock->fd,LOCK_EX|LOCK_NB)==0,
      "replica directory is already in use",Error::Code::unavailable);
    std::string config="EPS-REPLICA-1";U(config,options.run.size());config+=options.run;
    U(config,options.shard);U(config,options.shards);U(config,options.epoch);U(config,options.base_round);config+=options.binding;
    const auto metadata=options.directory/"configuration";
    if(fs::exists(metadata))Require(Metadata(metadata)==config,"replica log configuration differs");
    else {
      Require(!fs::exists(options.directory/"baseline.eps"),"replica baseline exists without configuration");
      for(const auto& e:fs::directory_iterator(options.directory))Require(!e.path().filename().string().starts_with("round-"),"replica records exist without configuration");
      Atomic(metadata,config);
    }
    const auto lease=options.directory/"lease";
    if(fs::exists(lease)){const auto data=Metadata(lease);head.lease={U(data),data.substr(8)};
      Require(head.lease.term && !head.lease.primary.empty() && head.lease.primary.size()<=256,"invalid persisted replica lease");}
    Refresh(true);
  }
  fs::path Path(bool baseline,std::uint64_t round)const {
    return options.directory/(baseline?"baseline.eps":"round-"+std::to_string(round)+".epsr");
  }
  void Refresh(bool verify=false) {
    head.base_round=options.base_round;head.next_round=options.base_round;head.bytes=head.baseline_bytes=0;
    head.baseline=fs::exists(Path(true,options.base_round));
    if(head.baseline){head.bytes=head.baseline_bytes=fs::file_size(Path(true,options.base_round));if(verify)Checksum(Path(true,options.base_round));}
    Require(head.bytes<=options.disk_bytes,"replica baseline exceeds disk budget");
    while(fs::exists(Path(false,head.next_round))) {
      Require(head.baseline && head.next_round!=UINT64_MAX,"replica records lack a baseline or exceed clock");
      const auto path=Path(false,head.next_round);const auto bytes=fs::file_size(path);
      Require(bytes>=8 && bytes<=options.record_bytes && bytes<=options.disk_bytes-head.bytes,"replica log exceeds budget");
      if(verify)Checksum(path);head.bytes+=bytes;++head.next_round;
    }
    for(const auto& e:fs::directory_iterator(options.directory)) {
      const auto name=e.path().filename().string();if(!name.starts_with("round-") || !name.ends_with(".epsr"))continue;
      const auto number=std::string_view(name).substr(6,name.size()-11);std::uint64_t round=0;
      const auto parsed=std::from_chars(number.data(),number.data()+number.size(),round);
      Require(parsed.ec==std::errc{} && parsed.ptr==number.data()+number.size() && round>=head.base_round && round<head.next_round,
              "replica log has a gap or invalid record name");
    }
    Require(head.bytes<=options.disk_bytes,"replica disk budget exceeded");
  }
  void Check(const ReplicaLease& lease)const {
    Require(lease.term && lease==head.lease,"replica primary term is fenced",Error::Code::stale_generation);
  }
  void CatchUp() {
    const auto baseline=Path(true,options.base_round);
    if(!head.baseline && fs::exists(baseline)) {
      head.baseline=true;head.baseline_bytes=fs::file_size(baseline);head.bytes+=head.baseline_bytes;
    }
    Require(head.bytes<=options.disk_bytes,"replica payload disk budget exceeded");
    while(fs::exists(Path(false,head.next_round))) {
      Require(head.baseline && head.next_round!=UINT64_MAX,"invalid replica frontier");
      const auto bytes=fs::file_size(Path(false,head.next_round));
      Require(bytes>=8 && bytes<=options.record_bytes && bytes<=options.disk_bytes-head.bytes,"replica record budget exceeded");
      head.bytes+=bytes;++head.next_round;
    }
  }
};
ReplicaLog::ReplicaLog(ReplicaLogOptions o):impl_(std::make_unique<Impl>(std::move(o))){}
ReplicaLog::~ReplicaLog()=default;
ReplicaHead ReplicaLog::Head()const {std::lock_guard lock(impl_->mutex);impl_->CatchUp();return impl_->head;}
void ReplicaLog::Check(const ReplicaLease& lease)const {std::lock_guard lock(impl_->mutex);impl_->Check(lease);}
ReplicaLease ReplicaLog::Fence(std::uint64_t expected,std::string primary) {
  std::lock_guard lock(impl_->mutex);Require(!primary.empty() && primary.size()<=256 && expected!=UINT64_MAX,"invalid replica promotion");
  auto& lease=impl_->head.lease;
  if(lease.term==expected+1 && lease.primary==primary)return lease;
  Require(lease.term==expected,"replica promotion compare-and-swap failed",Error::Code::stale_generation);
  std::string data;U(data,expected+1);data+=primary;Atomic(impl_->options.directory/"lease",data);lease={expected+1,std::move(primary)};
  for(const auto& e:fs::directory_iterator(impl_->options.directory)) {
    const auto name=e.path().filename().string();if(name.starts_with("upload-") && (name.ends_with(".part") || name.ends_with(".size")))fs::remove(e.path());
  }
  SyncDirectory(impl_->options.directory);impl_->CatchUp();return lease;
}
void ReplicaLog::Write(const ReplicaLease& lease,bool baseline,std::uint64_t round,std::uint64_t total,
                       std::uint64_t offset,const std::string& data) {
  std::lock_guard lock(impl_->mutex);impl_->Check(lease);
  Require(total>=8 && total<=impl_->options.disk_bytes && (baseline || total<=impl_->options.record_bytes) &&
    offset<=total && data.size()<=total-offset && data.size()<=(1<<20) && (!data.empty() || offset==total),"invalid replica chunk");
  if(baseline)Require(round==impl_->options.base_round,"replica baseline round differs");
  const auto target=impl_->Path(baseline,round);
  if(fs::exists(target)) {
    File file(target,O_RDONLY);Require(file.Size()==total && file.Read(offset,data.size())==data,"replicated retry changed committed data");
    file.Sync();SyncDirectory(impl_->options.directory);impl_->CatchUp();return;
  }
  Require(baseline || (impl_->head.baseline && round==impl_->head.next_round && round!=UINT64_MAX),"replica round is not contiguous");
  Require(total<=impl_->options.disk_bytes-impl_->head.bytes,"replica disk budget exhausted",Error::Code::capacity_exceeded);
  const auto prefix=impl_->options.directory/("upload-"+std::to_string(lease.term)+"-"+(baseline?"baseline":std::to_string(round)));
  const auto partial=prefix.string()+".part",size=prefix.string()+".size";
  std::string amount;U(amount,total);
  if(fs::exists(size))Require(Metadata(size)==amount,"replica retry changed upload length");else Atomic(size,amount);
  File file(partial,O_RDWR|O_CREAT);const auto written=file.Size();
  Require(written<=total && offset<=written,"replica upload has a gap");
  const auto overlap=std::min<std::uint64_t>(written-offset,data.size());
  Require(file.Read(offset,overlap)==data.substr(0,overlap),"replica retry changed pending data");
  if(data.size()>overlap)file.Write(offset+overlap,std::string_view(data).substr(overlap));
  if(file.Size()!=total)return;
  file.Sync();Checksum(partial);
  if(!baseline) {
    const auto image=DecodeReplicatedUpdate(file.Read(0,total),impl_->options.record_bytes);
    Require(image.run==impl_->options.run && image.shard==impl_->options.shard && image.shards==impl_->options.shards &&
      image.epoch==impl_->options.epoch && image.round==round,"replica image identity differs from log");
    if(impl_->options.before_publish)impl_->options.before_publish(round);
  }
  Require(::link(partial.c_str(),target.c_str())==0,"cannot publish replica record",Error::Code::unavailable);
  ::unlink(partial.c_str());::unlink(size.c_str());SyncDirectory(impl_->options.directory);impl_->CatchUp();
}
SnapshotChunk ReplicaLog::Read(const ReplicaLease& lease,bool baseline,std::uint64_t round,std::uint64_t offset,std::uint32_t count)const {
  std::lock_guard lock(impl_->mutex);impl_->Check(lease);
  Require((!baseline || round==impl_->options.base_round) && count>0 && count<=(1<<20),"invalid replica read");
  return detail::ReadSnapshotChunk(impl_->Path(baseline,round),offset,count);
}
}  // namespace eps
