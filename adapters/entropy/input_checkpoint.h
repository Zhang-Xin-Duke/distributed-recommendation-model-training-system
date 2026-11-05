#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eps { namespace input_checkpoint {

inline constexpr std::uint64_t no_assignment = std::numeric_limits<std::uint64_t>::max();
inline constexpr std::uint64_t item_limit = 1u << 20;
inline constexpr std::uint64_t byte_limit = 64u << 20;

struct Cursor {
  std::string stream;
  std::uint64_t handle = no_assignment;
  std::vector<std::uint64_t> assignments;
  bool active = false;
  std::uint64_t current = no_assignment;
  std::string filename;
  std::uint64_t row = 0;
};

struct Files {
  std::uint64_t handle = 0;
  std::vector<std::string> filenames;
  std::vector<std::uint64_t> excluded;
};

inline std::string AssignmentTag(std::uint64_t value) {
  std::string bytes(8, '\0');
  for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<char>(value >> (i * 8));
  return bytes;
}
inline std::uint64_t AssignmentId(const std::string& bytes) {
  if (bytes.size() != 8) throw std::invalid_argument("input assignment metadata is absent");
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (i * 8);
  return value;
}

class Writer {
 public:
  explicit Writer(const std::string& kind) { String(kind); }
  void Number(std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) bytes_.push_back(static_cast<char>(value >> (i * 8)));
    Check();
  }
  void String(const std::string& value) {
    Number(value.size()); bytes_.append(value); Check();
  }
  void Numbers(const std::vector<std::uint64_t>& values) {
    if (values.size() > item_limit) throw std::invalid_argument("input checkpoint item limit exceeded");
    Number(values.size()); for (auto value : values) Number(value);
  }
  std::string Finish() { return std::move(bytes_); }
 private:
  void Check() const {
    if (bytes_.size() > byte_limit) throw std::invalid_argument("input checkpoint byte limit exceeded");
  }
  std::string bytes_;
};

class Reader {
 public:
  Reader(const std::string& bytes, const std::string& kind) : bytes_(bytes) {
    if (bytes.size() > byte_limit || String() != kind)
      throw std::invalid_argument("input checkpoint format differs");
  }
  std::uint64_t Number() {
    Require(8); std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i)
      value |= std::uint64_t(static_cast<unsigned char>(bytes_[offset_++])) << (i * 8);
    return value;
  }
  std::string String() {
    const auto size = Number(); Require(size);
    auto result = bytes_.substr(offset_, size); offset_ += size; return result;
  }
  std::vector<std::uint64_t> Numbers() {
    const auto size = Count(); std::vector<std::uint64_t> result;
    for (std::uint64_t i = 0; i < size; ++i) result.push_back(Number());
    return result;
  }
  std::uint64_t Count() {
    const auto size = Number();
    if (size > item_limit) throw std::invalid_argument("input checkpoint item limit exceeded");
    return size;
  }
  void Finish() const {
    if (offset_ != bytes_.size()) throw std::invalid_argument("input checkpoint has trailing data");
  }
 private:
  void Require(std::uint64_t size) const {
    if (size > bytes_.size() - offset_) throw std::invalid_argument("truncated input checkpoint");
  }
  const std::string& bytes_;
  std::size_t offset_ = 0;
};

inline void Validate(const Cursor& cursor) {
  if (cursor.stream.empty() || cursor.assignments.size() > item_limit)
    throw std::invalid_argument("invalid input stream");
  std::set<std::uint64_t> unique(cursor.assignments.begin(), cursor.assignments.end());
  if (unique.size() != cursor.assignments.size() || unique.count(no_assignment))
    throw std::invalid_argument("input assignment history is invalid");
  if (cursor.active && (cursor.filename.empty() ||
      (cursor.handle != no_assignment && (cursor.assignments.empty() ||
         cursor.current != cursor.assignments.back()))))
    throw std::invalid_argument("active input cursor differs from its history");
  if (!cursor.active && (cursor.current != no_assignment || cursor.row != 0 || !cursor.filename.empty()))
    throw std::invalid_argument("inactive input cursor has an active position");
  if (cursor.handle == no_assignment && !cursor.assignments.empty())
    throw std::invalid_argument("direct input cursor has feeder assignments");
}

inline void Put(Writer& writer, const Cursor& cursor) {
  Validate(cursor); writer.String(cursor.stream); writer.Number(cursor.handle);
  writer.Numbers(cursor.assignments); writer.Number(cursor.active); writer.Number(cursor.current);
  writer.String(cursor.filename); writer.Number(cursor.row);
}

inline Cursor GetCursor(Reader& reader) {
  Cursor result; result.stream = reader.String(); result.handle = reader.Number();
  result.assignments = reader.Numbers(); const auto active = reader.Number();
  if (active > 1) throw std::invalid_argument("invalid input activity flag");
  result.active = active; result.current = reader.Number(); result.filename = reader.String();
  result.row = reader.Number(); Validate(result); return result;
}

inline std::string Encode(const Cursor& cursor) {
  Writer writer("eps-orc-cursor-v1"); Put(writer, cursor); return writer.Finish();
}

inline Cursor DecodeCursor(const std::string& bytes) {
  Reader reader(bytes, "eps-orc-cursor-v1"); auto result = GetCursor(reader); reader.Finish(); return result;
}

inline std::string Encode(const Files& files) {
  if (files.handle >= 4096 || files.filenames.size() > item_limit)
    throw std::invalid_argument("invalid input source roster");
  std::set<std::uint64_t> unique;
  for (auto id : files.excluded)
    if (id >= files.filenames.size() || !unique.insert(id).second)
      throw std::invalid_argument("invalid excluded input assignment");
  Writer writer("eps-input-files-v1"); writer.Number(files.handle); writer.Number(files.filenames.size());
  for (const auto& path : files.filenames) {
    if (path.empty()) throw std::invalid_argument("empty input filename");
    writer.String(path);
  }
  writer.Numbers(files.excluded); return writer.Finish();
}

inline Files DecodeFiles(const std::string& bytes) {
  Reader reader(bytes, "eps-input-files-v1"); Files result; result.handle = reader.Number();
  const auto count = reader.Count();
  for (std::uint64_t i = 0; i < count; ++i) result.filenames.push_back(reader.String());
  result.excluded = reader.Numbers(); reader.Finish(); (void)Encode(result); return result;
}

inline Files RestoreCut(const Files& source, const std::vector<Cursor>& cursors) {
  (void)Encode(source); Files result = source;
  std::set<std::uint64_t> excluded(source.excluded.begin(), source.excluded.end()), observed;
  for (const auto& cursor : cursors) {
    Validate(cursor);
    if (cursor.handle != source.handle) continue;
    for (auto id : cursor.assignments) {
      if (id >= source.filenames.size() || !observed.insert(id).second)
        throw std::invalid_argument("stream assignments overlap or exceed source roster");
      excluded.insert(id);
    }
    if (cursor.active && source.filenames.at(cursor.current) != cursor.filename)
      throw std::invalid_argument("active reader filename differs from its source assignment");
  }
  result.excluded.assign(excluded.begin(), excluded.end()); return result;
}

struct Source {
  Files full;
  std::vector<std::string>* live = nullptr;
  std::atomic<std::uint32_t>* offset = nullptr;
  std::set<std::uint64_t> excluded;
};
inline std::mutex source_mutex;
inline std::map<const void*, Source> sources;

inline void RegisterSource(const void* owner, std::vector<std::string>* files,
                           std::atomic<std::uint32_t>* offset) {
  Files full; full.filenames = *files; (void)Encode(full);
  Source source{std::move(full), files, offset, {}};
  std::lock_guard<std::mutex> lock(source_mutex); sources[owner] = std::move(source);
}

template <class Next>
inline std::pair<std::string, std::uint64_t> NextAssignment(const void* owner, Next next) {
  std::lock_guard<std::mutex> lock(source_mutex);
  const auto found = sources.find(owner);
  if (found == sources.end()) throw std::invalid_argument("input source has not been initialized");
  auto& source = found->second;
  while (true) {
    const auto& filename = next();
    if (filename.empty()) return {"", no_assignment};
    const auto address = reinterpret_cast<std::uintptr_t>(&filename);
    const auto begin = reinterpret_cast<std::uintptr_t>(source.live->data());
    if (address < begin || (address - begin) % sizeof(std::string) ||
        (address - begin) / sizeof(std::string) >= source.live->size())
      throw std::invalid_argument("input assignment does not belong to its source");
    const auto index = (address - begin) / sizeof(std::string);
    if (!source.excluded.count(index)) return {filename, index};
  }
}

inline Files ExportSource(const void* owner, std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(source_mutex);
  auto result = sources.at(owner).full; result.handle = handle; return result;
}

inline void RestoreSource(const void* owner, const Files& files) {
  (void)Encode(files);
  std::set<std::uint64_t> excluded(files.excluded.begin(), files.excluded.end());
  std::lock_guard<std::mutex> lock(source_mutex); auto& source = sources.at(owner);
  if (*source.live != files.filenames)
    throw std::invalid_argument("restored source order differs from the initialized roster");
  source.excluded = std::move(excluded); source.full = files;
  source.offset->store(0, std::memory_order_release);
}

inline std::string EncodeCapture(const std::map<std::string, Cursor>& cursors) {
  if (cursors.size() > item_limit) throw std::invalid_argument("too many input streams");
  Writer writer("eps-input-capture-v1"); writer.Number(cursors.size());
  for (const auto& item : cursors) { writer.String(item.first); Put(writer, item.second); }
  return writer.Finish();
}

inline std::map<std::string, Cursor> DecodeCapture(const std::string& bytes) {
  Reader reader(bytes, "eps-input-capture-v1"); std::map<std::string, Cursor> result;
  const auto count = reader.Count();
  for (std::uint64_t i = 0; i < count; ++i) {
    auto key = reader.String(); auto cursor = GetCursor(reader);
    if (key.empty() || !result.emplace(key, std::move(cursor)).second)
      throw std::invalid_argument("duplicate input capture key");
  }
  reader.Finish(); return result;
}

inline void WriteFile(const std::string& path, const std::string& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), bytes.size()); output.close();
  if (!output) throw std::runtime_error("cannot write native input checkpoint staging file");
}

inline std::string ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  const auto size = input.tellg();
  if (!input || size < 0 || std::uint64_t(size) > byte_limit)
    throw std::runtime_error("cannot read native input checkpoint staging file");
  std::string result(static_cast<std::size_t>(size), '\0'); input.seekg(0);
  if (!input.read(result.data(), result.size())) throw std::runtime_error("truncated native input staging file");
  return result;
}

// TensorFlow serialization can visit nested readers on different executor
// threads. Python serializes Begin/Finish for each learner's save operation.
inline std::mutex capture_mutex;
inline bool capturing = false;
inline std::map<std::string, Cursor> captured;
inline void BeginCapture() {
  std::lock_guard<std::mutex> lock(capture_mutex);
  if (capturing) throw std::runtime_error("input capture already active");
  captured.clear(); capturing = true;
}
inline void Capture(const std::string& key, const Cursor& cursor) {
  Validate(cursor); std::lock_guard<std::mutex> lock(capture_mutex);
  if (capturing) captured[key] = cursor;
}
inline void FinishCapture(const std::string& path) {
  std::lock_guard<std::mutex> lock(capture_mutex);
  if (!capturing) throw std::runtime_error("input capture is not active");
  auto bytes = EncodeCapture(captured); WriteFile(path, bytes); capturing = false;
}
inline void CancelCapture() {
  std::lock_guard<std::mutex> lock(capture_mutex); captured.clear(); capturing = false;
}

}}  // namespace eps::input_checkpoint
