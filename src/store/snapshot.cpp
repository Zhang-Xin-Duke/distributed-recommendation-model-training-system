#include "store/snapshot.hpp"

#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>

#include "common/error.hpp"

namespace eps::detail {
namespace {
constexpr std::uint64_t kOffset = 14695981039346656037ULL;
constexpr std::uint64_t kPrime = 1099511628211ULL;
constexpr std::uint64_t kMagic = 0x3153504541524853ULL;
constexpr std::uint64_t kMagicV2 = 0x3253504541524853ULL;
struct File {
  int fd;
  ~File() { if (fd >= 0) ::close(fd); }
};
class Binary {
 public:
  explicit Binary(int fd) : fd_(fd) {}
  void Bytes(void* data, std::size_t size, bool write, bool checksum = true) {
    auto* p = static_cast<unsigned char*>(data);
    for (std::size_t pos = 0; pos < size;) {
      if (write) {
        const auto n = std::min(size - pos, buffer_.size() - filled_);
        std::memcpy(buffer_.data() + filled_, p + pos, n);
        filled_ += n; pos += n;
        if (filled_ == buffer_.size()) Flush();
      } else {
        if (cursor_ == filled_) {
          ssize_t n;
          do { n = ::read(fd_, buffer_.data(), buffer_.size()); } while (n < 0 && errno == EINTR);
          Require(n > 0, "snapshot truncated", Error::Code::unavailable);
          cursor_ = 0; filled_ = n;
        }
        const auto n = std::min(size - pos, filled_ - cursor_);
        std::memcpy(p + pos, buffer_.data() + cursor_, n);
        cursor_ += n; pos += n;
      }
    }
    position_+=size;
    if (checksum) for (std::size_t i = 0; i < size; ++i) hash_ = (hash_ ^ p[i]) * kPrime;
  }
  void U64(std::uint64_t& value, bool write, bool checksum = true) {
    std::array<unsigned char, 8> b{};
    if (write) for (unsigned i = 0; i < 8; ++i) b[i] = (value >> (i * 8)) & 255;
    Bytes(b.data(), b.size(), write, checksum);
    if (!write) { value = 0; for (unsigned i = 0; i < 8; ++i) value |= std::uint64_t(b[i]) << (i * 8); }
  }
  void Float(float& value, bool write) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    std::uint64_t bits = write ? std::bit_cast<std::uint32_t>(value) : 0;
    // Store a float as four little-endian bytes, independent of host endianness.
    std::array<unsigned char, 4> b{};
    if (write) for (unsigned i = 0; i < 4; ++i) b[i] = (bits >> (8 * i)) & 255;
    Bytes(b.data(), b.size(), write);
    if (!write) {
      std::uint32_t u = 0;
      for (unsigned i = 0; i < 4; ++i) u |= std::uint32_t(b[i]) << (8 * i);
      value = std::bit_cast<float>(u);
      Require(std::isfinite(value), "snapshot contains non-finite state");
    }
  }
  void String(std::string& value, bool write) {
    std::uint64_t n = value.size(); U64(n, write);
    Require(n <= 1024, "snapshot string too long");
    if (!write) value.resize(n);
    Bytes(value.data(), n, write);
  }
  void Definition(TableDefinition& d, bool write, bool with_bounds = true) {
    std::uint64_t id = d.id, dim = d.dimension, count = d.shard_count;
    U64(id, write); U64(dim, write); U64(count, write);
    Require(id <= UINT32_MAX && dim <= 65536 && count <= UINT32_MAX,
            "invalid snapshot table definition");
    d.id = id; d.dimension = dim; d.shard_count = count;
    U64(d.partition_epoch, write);
    auto& o = d.optimizer;
    Float(o.learning_rate, write); Float(o.beta1, write); Float(o.beta2, write);
    Float(o.epsilon, write); Float(o.initial_beta1_power, write);
    Float(o.initial_beta2_power, write);
    if(with_bounds) {
      std::uint64_t enabled = o.weight_bounds.has_value(); U64(enabled, write);
      Require(enabled <= 1, "invalid snapshot weight-bounds flag");
      if(enabled) {
        if(!write)o.weight_bounds=WeightBounds{};
        Float(o.weight_bounds->minimum,write); Float(o.weight_bounds->maximum,write);
      } else o.weight_bounds.reset();
    }
    Float(d.initialization_min, write); Float(d.initialization_max, write);
    U64(d.initialization_seed, write);
  }
  std::uint64_t hash() const { return hash_; }
  std::uint64_t position() const { return position_; }
  void Flush() {
    std::size_t pos = 0;
    while (pos < filled_) {
      auto n = ::write(fd_, buffer_.data() + pos, filled_ - pos);
      if (n < 0 && errno == EINTR) continue;
      Require(n > 0, "snapshot write failed", Error::Code::unavailable);
      pos += n;
    }
    filled_ = 0;
  }
  bool AtEnd() const {
    if (cursor_ != filled_) return false;
    char c; ssize_t n;
    do { n = ::read(fd_, &c, 1); } while (n < 0 && errno == EINTR);
    return n == 0;
  }
 private:
  int fd_;
  std::uint64_t hash_ = kOffset;
  std::uint64_t position_ = 0;
  std::array<unsigned char, 65536> buffer_{};
  std::size_t cursor_ = 0, filled_ = 0;
};
void Sync(int fd) {
  Require(::fsync(fd) == 0, "snapshot fsync failed", Error::Code::unavailable);
}
}

SnapshotChunk ReadSnapshotChunk(const std::filesystem::path& path,
    std::uint64_t offset, std::uint32_t max_bytes) {
  Require(max_bytes>0 && max_bytes<=(1U<<20), "checkpoint chunk exceeds 1 MiB limit");
  File file{::open(path.c_str(),O_RDONLY)};
  Require(file.fd>=0,"cannot open checkpoint",Error::Code::unavailable);
  struct stat info{};
  Require(::fstat(file.fd,&info)==0 && S_ISREG(info.st_mode) && info.st_size>=8,
          "invalid checkpoint file",Error::Code::unavailable);
  const auto size=std::uint64_t(info.st_size);
  Require(offset<=size,"checkpoint offset exceeds file size");
  const auto count=std::min<std::uint64_t>(max_bytes,size-offset);
  SnapshotChunk result{offset,size,std::string(count,'\0'),offset+count==size};
  for(std::size_t done=0;done<count;) {
    auto n=::pread(file.fd,result.data.data()+done,count-done,off_t(offset+done));
    if(n<0 && errno==EINTR)continue;
    Require(n>0,"checkpoint file changed during export",Error::Code::unavailable);
    done+=std::size_t(n);
  }
  return result;
}

void WriteSnapshot(const std::filesystem::path& path, const std::string& run,
                   const ShardStore& store) {
  std::filesystem::create_directories(path.parent_path());
  auto pending = path.string() + ".pending";
  File file{::open(pending.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600)};
  Require(file.fd >= 0, "cannot create snapshot staging file", Error::Code::unavailable);
  try {
    Binary b(file.fd);
    auto magic = kMagicV2; b.U64(magic, true);
    auto run_copy = run; b.String(run_copy, true);
    std::uint64_t shard = store.shard(), count = store.shard_count(), epoch = store.epoch();
    b.U64(shard, true); b.U64(count, true); b.U64(epoch, true);
    std::uint64_t tables = store.tables().size(); b.U64(tables, true);
    for (const auto& [id, t] : store.tables()) {
      auto d = t.definition; b.Definition(d, true);
      auto seq = t.optimizer_sequence; auto beta1 = t.beta1_power, beta2 = t.beta2_power;
      b.U64(seq, true); b.Float(beta1, true); b.Float(beta2, true);
      std::uint64_t rows = t.keys.size(); b.U64(rows, true);
      for (auto key : t.keys) {
        b.U64(key, true);
        const auto row = t.At(t.index.at(key));
        auto updates = *row.updates; b.U64(updates, true);
        for (std::uint64_t i = 0; i < std::uint64_t(t.definition.dimension) * 3; ++i)
          b.Float(row.weights[i], true);
      }
    }
    auto checksum = b.hash(); b.U64(checksum, true, false); b.Flush(); Sync(file.fd);
    // link is an atomic no-overwrite publication on the same filesystem.
    Require(::link(pending.c_str(), path.c_str()) == 0,
            "snapshot already exists or cannot be published", Error::Code::unavailable);
    ::unlink(pending.c_str());
    File directory{::open(path.parent_path().c_str(), O_RDONLY)};
    Require(directory.fd >= 0, "cannot open snapshot directory", Error::Code::unavailable);
    Sync(directory.fd);
  } catch (...) { ::unlink(pending.c_str()); throw; }
}

std::unique_ptr<ShardStore> ReadSnapshot(const std::filesystem::path& path,
    const std::string& new_run, ShardId shard, std::uint32_t count,
    std::uint64_t epoch, std::uint64_t byte_limit, RestoredSnapshot* provenance,bool replica_same_run) {
  File file{::open(path.c_str(), O_RDONLY)};
  Require(file.fd >= 0, "cannot open snapshot", Error::Code::unavailable);
  Binary b(file.fd);
  std::uint64_t magic = 0; b.U64(magic, false);
  Require(magic == kMagic || magic == kMagicV2, "unsupported snapshot format");
  std::string old_run; b.String(old_run, false);
  Require(replica_same_run ? old_run==new_run : old_run!=new_run,
    replica_same_run ? "replica baseline run differs" : "restore requires a new run ID");
  std::uint64_t old_shard = 0, old_count = 0, old_epoch = 0;
  b.U64(old_shard, false); b.U64(old_count, false); b.U64(old_epoch, false);
  Require(old_shard == shard && old_count == count && old_epoch == epoch,
          "snapshot partition does not match server");
  auto store = std::make_unique<ShardStore>(shard, count, epoch, byte_limit);
  std::uint64_t tables = 0; b.U64(tables, false);
  Require(tables <= 1024, "snapshot table count exceeds limit");
  for (std::uint64_t i = 0; i < tables; ++i) {
    TableDefinition d{}; b.Definition(d, false, magic == kMagicV2); store->Create(d);
    auto& t = store->Table(d.id);
    b.U64(t.optimizer_sequence, false); b.Float(t.beta1_power, false); b.Float(t.beta2_power, false);
    Require(t.beta1_power >= 0 && t.beta1_power < 1 &&
            t.beta2_power >= 0 && t.beta2_power < 1, "invalid snapshot optimizer state");
    std::uint64_t rows = 0; b.U64(rows, false);
    Require(rows <= byte_limit / (std::uint64_t(d.dimension) * 12 + 8),
            "snapshot exceeds configured row capacity", Error::Code::capacity_exceeded);
    RowState state; state.row.table = d.id;
    state.weights.resize(d.dimension); state.first_moment.resize(d.dimension);
    state.second_moment.resize(d.dimension);
    for (std::uint64_t j = 0; j < rows; ++j) {
      b.U64(state.row.key, false); b.U64(state.row_updates, false);
      for (auto* v : {&state.weights, &state.first_moment, &state.second_moment}) {
        v->resize(d.dimension);
        for (auto& x : *v) b.Float(x, false);
      }
      for (auto x : state.second_moment) Require(x >= 0, "negative second moment in snapshot");
      store->Import(state);
    }
  }
  const auto actual = b.hash(); std::uint64_t expected = 0; b.U64(expected, false, false);
  Require(actual == expected, "snapshot checksum mismatch");
  Require(b.AtEnd(), "snapshot contains trailing data");
  if(provenance)*provenance={true,true,new_run,old_run,b.position(),expected,shard,count,epoch};
  return store;
}
}  // namespace eps::detail
