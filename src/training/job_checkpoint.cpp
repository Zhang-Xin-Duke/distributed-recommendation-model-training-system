#include "job_checkpoint.hpp"
#include "checkpoint_json.hpp"
#include "eps/brpc_client.hpp"
#include "eps/brpc_synchronous_gradients.hpp"
#include "eps/brpc_training.hpp"
#include <openssl/evp.h>
#include <array>
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <unistd.h>

namespace eps::training {
namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
template<class T> T Take(std::future<Result<T>> f) {
  auto r = f.get();
  if (const auto* e = std::get_if<Error>(&r)) throw std::runtime_error(e->message);
  return std::get<T>(std::move(r));
}
void Require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Digest {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{EVP_MD_CTX_new(), EVP_MD_CTX_free};
  Digest() { Require(ctx && EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr) == 1, "initialize SHA256"); }
  void Add(const char* p, std::size_t n) {
    Require(EVP_DigestUpdate(ctx.get(), p, n) == 1, "update SHA256");
  }
  std::string End() {
    unsigned char bytes[EVP_MAX_MD_SIZE]; unsigned n = 0;
    Require(EVP_DigestFinal_ex(ctx.get(), bytes, &n) == 1, "finalize SHA256");
    std::ostringstream out;
    for (unsigned i = 0; i < n; ++i) out << std::hex << std::setw(2) << std::setfill('0') << unsigned(bytes[i]);
    return out.str();
  }
};
struct File {
  int fd;
  explicit File(const fs::path& path, int flags) : fd(::open(path.c_str(), flags, 0600)) {
    Require(fd >= 0, "cannot open checkpoint file");
  }
  ~File() { ::close(fd); }
  void Write(const char* p, std::size_t n) {
    while (n) {
      const auto done = ::write(fd, p, n);
      if (done < 0 && errno == EINTR) continue;
      Require(done > 0, "checkpoint write failed"); p += done; n -= done;
    }
  }
  void Sync() { Require(::fsync(fd) == 0, "checkpoint fsync failed"); }
};
void SyncDirectory(const fs::path& path) { File directory(path, O_RDONLY); directory.Sync(); }
Json ReadJson(const fs::path& path) {
  Require(fs::file_size(path) <= (32ULL << 20), "checkpoint metadata exceeds limit");
  std::ifstream input(path); Require(bool(input), "cannot read checkpoint metadata");
  Json j; input >> j; return j;
}
// Retain only eight trailing bytes; hash multi-GiB files without loading them.
struct Fingerprint {
  Digest sha;
  std::array<unsigned char, 8> tail{};
  std::uint64_t bytes = 0, checksum = 14695981039346656037ULL;
  void Add(const char* data, std::size_t n) {
    sha.Add(data, n);
    for (std::size_t i = 0; i < n; ++i) {
      if (bytes >= 8) { checksum ^= tail[bytes % 8]; checksum *= 1099511628211ULL; }
      tail[bytes++ % 8] = static_cast<unsigned char>(data[i]);
    }
  }
  Json End() {
    Require(bytes >= 8, "truncated parameter snapshot");
    std::uint64_t stored = 0;
    for (unsigned i = 0; i < 8; ++i) stored |= std::uint64_t(tail[(bytes - 8 + i) % 8]) << (i * 8);
    Require(stored == checksum, "parameter snapshot native checksum mismatch");
    return {{"bytes", bytes}, {"checksum", checksum}, {"sha256", sha.End()}};
  }
};
Json FingerprintFile(const fs::path& path) {
  std::ifstream input(path, std::ios::binary); Require(bool(input), "missing parameter snapshot");
  Fingerprint f; std::array<char, 1 << 20> block;
  while (input) { input.read(block.data(), block.size()); f.Add(block.data(), input.gcount()); }
  Require(input.eof(), "read parameter snapshot failed"); return f.End();
}
Json Progress(const TrainingProgressCheckpoint& c) {
  return {{"source_run", c.source_run}, {"epoch", c.epoch}, {"consistency", static_cast<unsigned>(c.consistency)},
    {"max_delay", c.max_delay}, {"completed", c.completed}};
}
void Workers(const Json& workers, std::uint64_t round, std::size_t count, const Json& model) {
  Require(workers.is_array() && workers.size() == count && count > 0, "worker input group mismatch");
  std::set<unsigned> seen;
  for (const auto& worker : workers) {
    const auto id = worker.at("worker").get<unsigned>();
    Require(id < count && seen.insert(id).second, "invalid or duplicate input worker");
    const auto& state = worker.at("input_state");
    Require(state.at("next_iteration") == round,
            "worker input iteration mismatch");
    Require(state.at("model") == model, "worker checkpoint model mismatch");
    const auto data = state.at("dataset_json").get<std::string>();
    Digest hash; hash.Add(data.data(), data.size());
    Require(hash.End() == state.at("dataset_sha256").get<std::string>(), "worker input dataset fingerprint mismatch");
    const auto input = Json::parse(data);
    if (state.at("kind") == "indexed-recommendation-v1") {
      const auto batches = input.at("train_batches").get<std::uint64_t>();
      Require(input.at("kind") == state.at("kind") && input.at("worker") == id &&
              batches > 0 && batches <= 65536 && state.at("next_batch") == round % batches &&
              input.at("validation_batches").get<std::uint64_t>() > 0,
              "indexed input cursor mismatch");
      for (const auto* key : {"version", "train_sha256", "validation_sha256"}) {
        const auto value = input.at(key).get<std::string>();
        Require(value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string::npos,
                "invalid indexed input digest");
      }
    } else {
      Require(state.at("kind") == "cyclic-json-batches-v1" && input.is_array() && !input.empty() &&
              state.at("next_batch") == round % input.size(), "worker input cursor mismatch");
      for (const auto& batch : input)
        Require(batch.is_array() && !batch.empty(), "empty checkpoint input batch");
    }
  }
}
std::uint64_t Round(const Json& progress) {
  const auto c = progress.at("completed").get<std::vector<std::uint64_t>>();
  Require(!c.empty(), "empty progress group");
  for (auto n : c) Require(n == c.front(), "training workers are not at a joint round boundary");
  return c.front();
}
void ValidateManifest(const Json& m, const fs::path& directory) {
  Require(m.at("schema_version") == 1 && m.at("kind") == "eps-synchronous-training",
          "unsupported joint checkpoint");
  const auto round = Round(m.at("progress"));
  const auto count = m.at("parameters").size(), workers = m.at("workers").size();
  Require(count > 0 && count <= 1024 && m.at("groups").size() == count, "checkpoint shard group mismatch");
  Require(m.at("progress").at("source_run") == m.at("source_run") &&
          m.at("progress").at("epoch") == m.at("epoch"), "checkpoint progress identity mismatch");
  Workers(m.at("workers"), round, workers, m.at("model"));
  Require(m.at("progress").at("completed").size() == workers, "checkpoint worker membership mismatch");
  for (std::size_t shard = 0; shard < count; ++shard) {
    const auto& p = m.at("parameters").at(shard);
    const auto& g = m.at("groups").at(shard);
    Require(g.at("source_run") == m.at("source_run") && g.at("epoch") == m.at("epoch") &&
      g.at("shard") == shard && g.at("shards") == count && g.at("workers") == workers &&
      g.at("next_round") == round, "checkpoint gradient group mismatch");
    Require(p.at("source_run") == m.at("source_run") && p.at("epoch") == m.at("epoch") &&
      p.at("shard") == shard && p.at("shards") == count, "checkpoint parameter identity mismatch");
    const auto name = p.at("file").get<std::string>();
    Require(name == "shard-" + std::to_string(shard) + ".eps", "invalid parameter checkpoint filename");
    const auto fingerprint = FingerprintFile(directory / name);
    for (const auto* key : {"bytes", "checksum", "sha256"})
      Require(fingerprint.at(key) == p.at(key), "parameter checkpoint fingerprint mismatch");
    const auto schema = DecodeGroup(g);
    Require(schema.tables.size() == m.at("tables").size(), "checkpoint table membership mismatch");
    std::set<TableId> ids;
    for (const auto& t : m.at("tables")) {
      const auto definition = DecodeTable(t, count, m.at("epoch"));
      Require(ids.insert(definition.id).second, "duplicate checkpoint table");
      Require(std::any_of(schema.tables.begin(), schema.tables.end(), [&](const auto& item) {
        return item.table == definition.id && item.dimension == definition.dimension;
      }), "checkpoint table schema mismatch");
    }
  }
}
Json Save(const Json& j) {
  const auto run = j.at("run").get<std::string>(); const auto epoch = j.value("epoch", std::uint64_t(1));
  const auto addresses = j.at("shards").get<std::vector<std::string>>();
  Require(!addresses.empty() && addresses.size() <= 1024, "invalid checkpoint shard count");
  const auto directory = fs::absolute(j.at("checkpoint_directory").get<std::string>());
  // A unique directory makes partial publication visibly uncommitted and immutable.
  Require(fs::create_directory(directory), "checkpoint directory already exists or parent is absent");
  BrpcTrainingProgress coordinator(j.at("coordinator"), run, epoch, 120000);
  Take(coordinator.Pause());
  const auto progress = Progress(Take(coordinator.Checkpoint())); const auto round = Round(progress);
  Workers(j.at("workers"), round, progress.at("completed").size(), j.at("model"));
  Json manifest{{"schema_version", 1}, {"kind", "eps-synchronous-training"}, {"source_run", run},
    {"epoch", epoch}, {"progress", progress}, {"tables", j.at("tables")}, {"model", j.at("model")},
    {"workers", j.at("workers")}, {"groups", Json::array()}, {"parameters", Json::array()}};
  for (const auto& address : addresses) {
    BrpcSynchronousGradients group(address, run, epoch, 120000); Take(group.Pause());
    const auto c = Take(group.Checkpoint());
    Require(c.next_round == round && c.workers == progress.at("completed").size(), "parameter and worker round mismatch");
    manifest["groups"].push_back(Encode(c));
  }
  const auto directory_name = directory.string();
  Digest identity; identity.Add(directory_name.data(), directory_name.size());
  const auto id = "joint-" + identity.End().substr(0, 24);
  manifest["snapshot_id"] = id;
  for (std::size_t shard = 0; shard < addresses.size(); ++shard) {
    BrpcParameterService parameters(addresses[shard], 120000);
    for (const auto& t : j.at("tables")) Take(parameters.CreateTable(DecodeTable(t, addresses.size(), epoch)));
    const auto saved = Take(parameters.DrainAndFlush(id));
    Require(saved.shard == shard && saved.partition_epoch == epoch, "snapshot shard mismatch");
    for (const auto& context : saved.optimizer_contexts)
      Require(context.optimizer_sequence == round, "snapshot optimizer round mismatch");
    const auto name = "shard-" + std::to_string(shard) + ".eps";
    File file(directory / name, O_WRONLY | O_CREAT | O_EXCL); Fingerprint fingerprint;
    std::uint64_t offset = 0, total = 0;
    for (;;) {
      const auto chunk = Take(parameters.ReadCheckpoint({run, epoch, id}, offset, 1 << 20));
      if (!offset) total = chunk.total_bytes;
      Require(chunk.offset == offset && chunk.total_bytes == total && offset <= total &&
        chunk.data.size() <= total - offset && chunk.data.size() <= (1 << 20) &&
        (!chunk.data.empty() || chunk.eof), "invalid checkpoint chunk");
      file.Write(chunk.data.data(), chunk.data.size()); fingerprint.Add(chunk.data.data(), chunk.data.size());
      offset += chunk.data.size();
      Require(chunk.eof == (offset == total), "inconsistent checkpoint EOF");
      if (chunk.eof) break;
    }
    file.Sync();
    auto metadata = fingerprint.End();
    metadata.update({{"source_run", run}, {"shard", shard}, {"shards", addresses.size()}, {"epoch", epoch}, {"file", name}});
    manifest["parameters"].push_back(metadata);
  }
  ValidateManifest(manifest, directory);
  const auto text = manifest.dump(2) + "\n";
  { File pending(directory / "manifest.pending", O_WRONLY | O_CREAT | O_EXCL);
    pending.Write(text.data(), text.size()); pending.Sync(); }
  Require(::link((directory / "manifest.pending").c_str(), (directory / "manifest.json").c_str()) == 0,
          "cannot publish immutable joint manifest");
  Require(::unlink((directory / "manifest.pending").c_str()) == 0, "cannot remove manifest staging file");
  SyncDirectory(directory); SyncDirectory(directory.parent_path());
  return {{"manifest", (directory / "manifest.json").string()}, {"next_round", round}, {"paused", true}};
}
Json RestoreConfig(const Json& j, const Json& m, const fs::path& directory) {
  const auto run = j.at("run").get<std::string>();
  Require(!run.empty() && run != m.at("source_run").get<std::string>(), "restore requires a fresh run identity");
  Json result{{"coordinator", {{"schema_version", 1}, {"run", run}, {"epoch", m.at("epoch")},
    {"workers", m.at("workers").size()}, {"consistency", "sequential"}, {"max_delay", 0},
    {"restore", m.at("progress")}}}, {"shards", Json::array()}, {"workers", m.at("workers")}, {"model", m.at("model")}};
  const auto policy = m.at("progress").at("consistency").get<unsigned>();
  Require(policy <= 2, "invalid checkpoint progress policy");
  result["coordinator"]["consistency"] = policy == 0 ? "sequential" : policy == 1 ? "eventual" : "bounded_delay";
  result["coordinator"]["max_delay"] = m.at("progress").at("max_delay");
  for (std::size_t i = 0; i < m.at("parameters").size(); ++i) {
    const auto& p = m.at("parameters").at(i); const auto& g = m.at("groups").at(i);
    const auto reduction = g.at("reduction").get<unsigned>(); Require(reduction <= 2, "invalid reduction");
    result["shards"].push_back({{"schema_version", 1}, {"run_id", run}, {"listen", "127.0.0.1:0"},
      {"partition", {{"shard_id", i}, {"shard_count", m.at("parameters").size()}, {"epoch", m.at("epoch")}}},
      {"tables", m.at("tables")}, {"synchronous_training", {{"workers", m.at("workers").size()},
        {"reduction", reduction == 0 ? "sum" : reduction == 1 ? "worker_mean" : "sample_mean"},
        {"restore", {{"group", g}, {"parameter_snapshot", p}}}}},
      {"restore_file", (directory / p.at("file").get<std::string>()).string()}});
  }
  return result;
}
Json Resume(const Json& j, const Json& m, bool activate = true) {
  const auto run = j.at("run").get<std::string>(); const auto epoch = m.at("epoch").get<std::uint64_t>();
  Require(run != m.at("source_run").get<std::string>() && !run.empty(), "resume requires a fresh restored run");
  const auto addresses = j.at("shards").get<std::vector<std::string>>();
  Require(addresses.size() == m.at("parameters").size(), "resume shard membership mismatch");
  BrpcTrainingProgress coordinator(j.at("coordinator"), run, epoch, 120000);
  const auto s = Take(coordinator.Stats());
  Require(s.paused && !s.failed && !s.active && !s.waiting, "restored coordinator is not quiescent");
  auto progress = Progress(Take(coordinator.Checkpoint())); progress["source_run"] = m.at("source_run");
  Require(progress == m.at("progress"), "restored coordinator progress differs from manifest");
  for (std::size_t i = 0; i < addresses.size(); ++i) {
    BrpcSynchronousGradients group(addresses[i], run, epoch, 120000); Take(group.Pause());
    auto c = Encode(Take(group.Checkpoint())); c["source_run"] = m.at("source_run");
    Require(c == m.at("groups").at(i), "restored gradient group differs from manifest");
    BrpcParameterService parameters(addresses[i], 120000);
    const auto actual = Take(parameters.RestoredFrom()); const auto& p = m.at("parameters").at(i);
    Require(actual.loaded && actual.pristine && actual.run == run && actual.source_run == m.at("source_run").get<std::string>() &&
      actual.shard == i && actual.shard_count == addresses.size() && actual.partition_epoch == epoch &&
      actual.bytes == p.at("bytes") && actual.checksum == p.at("checksum"), "restored parameter identity mismatch");
    for (const auto& t : m.at("tables")) Take(parameters.CreateTable(DecodeTable(t, addresses.size(), epoch)));
  }
  // Validate the entire restored generation before any task admission opens.
  if (activate) {
    for (const auto& address : addresses) { BrpcSynchronousGradients group(address, run, epoch, 120000); Take(group.Resume()); }
    Take(coordinator.Resume());
  }
  return {{"ok", true}, {"next_round", Round(m.at("progress"))}, {"paused", !activate}};
}
}  // namespace
Json JobCheckpoint(const Json& config, const std::string& action) {
  if (action == "save-job") return Save(config);
  const auto path = fs::absolute(config.at("manifest").get<std::string>());
  const auto manifest = ReadJson(path); ValidateManifest(manifest, path.parent_path());
  if (action == "restore-config") return RestoreConfig(config, manifest, path.parent_path());
  if (action == "verify-job") return Resume(config, manifest, false);
  if (action == "resume-job") return Resume(config, manifest);
  throw std::runtime_error("unknown job checkpoint action");
}
}  // namespace eps::training
