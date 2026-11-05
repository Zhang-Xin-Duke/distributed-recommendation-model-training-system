#include "adapters/entropy/input_checkpoint.h"

#include <filesystem>
#include <iostream>
#include <thread>

namespace input = eps::input_checkpoint;

void Check(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <class F> void Reject(F function) {
  try { function(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid input checkpoint was accepted");
}

int main(int argc, char** argv) {
  try {
    Check(argc == 2, "capture staging path is required");
    input::Files source{0, {"same.orc", "second.orc", "same.orc", "fourth.orc", "late.orc"}, {}};
    input::Cursor first{"filefeeder://train/handle/0/interleave/0", 0, {0, 2}, true, 2, "same.orc", 2};
    input::Cursor second{"filefeeder://train/handle/0/interleave/1", 0, {1, 3}, true, 3, "fourth.orc", 1};
    const auto encoded = input::Encode(first);
    Check(input::Encode(input::DecodeCursor(encoded)) == encoded, "cursor round trip changed state");
    const auto restored = input::RestoreCut(source, {first, second});
    Check(restored.excluded == std::vector<std::uint64_t>({0, 1, 2, 3}), "saved assignment cut differs");
    Check(input::Encode(input::DecodeFiles(input::Encode(restored))) == input::Encode(restored),
          "source roster round trip changed state");

    // Each ORC occurrence has four rows. Model execution consumed five rows;
    // native readers advanced farther and their pending rows live in serialized
    // TensorFlow buffers. File four was assigned after stream zero's snapshot.
    using Row = std::pair<std::uint64_t, std::uint64_t>;
    const std::set<Row> consumed{{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 2}};
    std::vector<Row> remaining{{0, 2}, {0, 3}, {2, 0}, {2, 1}, {1, 3}, {3, 0}};
    for (const auto& cursor : {first, second})
      for (auto row = cursor.row; row < 4; ++row) remaining.emplace_back(cursor.current, row);
    for (std::uint64_t file = 0; file < source.filenames.size(); ++file)
      if (!std::binary_search(restored.excluded.begin(), restored.excluded.end(), file))
        for (std::uint64_t row = 0; row < 4; ++row) remaining.emplace_back(file, row);
    std::set<Row> all = consumed;
    for (const auto& row : remaining) Check(all.insert(row).second, "a buffered/source row was replayed twice");
    Check(all.size() == 20 && remaining.size() == 15, "input cut dropped rows");

    first.assignments.push_back(4); first.current = 4; first.filename = "late.orc"; first.row = 1;
    second.active = false; second.current = input::no_assignment; second.filename.clear(); second.row = 0;
    const auto twice = input::RestoreCut(restored, {first, second});
    Check(twice.excluded == std::vector<std::uint64_t>({0, 1, 2, 3, 4}), "repeated restore lost earlier assignments");
    auto wrong = first; wrong.filename = "changed.orc";
    Reject([&] { input::RestoreCut(source, {wrong}); });
    Reject([&] { input::RestoreCut(source, {first, first}); });
    for (auto size : {std::size_t(0), std::size_t(8), encoded.size() - 1})
      Reject([&] { input::DecodeCursor(encoded.substr(0, size)); });
    Reject([&] { input::DecodeCursor(encoded + "extra"); });

    std::vector<std::string> live = source.filenames;
    std::atomic<std::uint32_t> offset{0}; const int owner = 0;
    input::RegisterSource(&owner, &live, &offset);
    auto next = [&]() -> const std::string& {
      static const std::string empty;
      const auto id = offset.fetch_add(1);
      return id < live.size() ? live[id] : empty;
    };
    const auto original = input::NextAssignment(&owner, next);
    Check(original.first == "same.orc" && original.second == 0, "initial source assignment differs");
    input::RestoreSource(&owner, restored);
    input::HoldSource(&owner, true);
    Check(input::NextAssignment(&owner, next).first.empty() && offset.load() == 0,
          "restore initialization consumed a held source");
    input::HoldSource(&owner, false);
    const auto late = input::NextAssignment(&owner, next);
    Check(late.first == "late.orc" && late.second == 4, "restored queue lost stable occurrence IDs");
    Check(input::AssignmentId(input::AssignmentTag(late.second)) == 4, "assignment wire tag differs");
    Check(input::NextAssignment(&owner, next).first.empty(), "restored queue contains excluded files");
    input::RestoreSource(&owner, twice);
    Check(input::NextAssignment(&owner, next).first.empty(), "completed source was reopened");
    Check(input::Encode(input::ExportSource(&owner, 0)) == input::Encode(twice), "source export lost its full roster");

    std::vector<std::string> many;
    for (unsigned i = 0; i < 100; ++i) many.push_back("part-" + std::to_string(i));
    std::atomic<std::uint32_t> shared_offset{0}; const int shared_owner = 1;
    input::RegisterSource(&shared_owner, &many, &shared_offset);
    std::mutex assigned_mutex; std::set<std::uint64_t> assigned;
    std::vector<std::thread> fetchers;
    for (unsigned worker = 0; worker < 8; ++worker) fetchers.emplace_back([&] {
      while (true) {
        auto value = input::NextAssignment(&shared_owner, [&]() -> const std::string& {
          static const std::string empty; auto id = shared_offset.fetch_add(1);
          return id < many.size() ? many[id] : empty;
        });
        if (value.first.empty()) break;
        std::lock_guard<std::mutex> lock(assigned_mutex);
        Check(value.first == "part-" + std::to_string(value.second), "parallel assignment ID differs");
        Check(assigned.insert(value.second).second, "parallel source assignment was duplicated");
      }
    });
    for (auto& fetcher : fetchers) fetcher.join();
    Check(assigned.size() == many.size(), "parallel source allocation lost files");

    input::BeginCapture(); std::vector<std::thread> threads;
    for (std::uint64_t i = 0; i < 8; ++i)
      threads.emplace_back([i] {
        input::Cursor cursor{"stream-" + std::to_string(i), 0, {i}, false,
                             input::no_assignment, "", 0};
        input::Capture("iterator-" + std::to_string(i), cursor);
      });
    for (auto& thread : threads) thread.join();
    input::FinishCapture(argv[1]);
    Check(input::DecodeCapture(input::ReadFile(argv[1])).size() == 8, "concurrent reader captures were lost");
    std::filesystem::remove(argv[1]);

    // Parallel interleave may destroy an exhausted leaf while its records stay
    // in upstream serialized buffers. Preserve that leaf's assignment history.
    input::BeginInputGeneration("train");
    Check(input::ReaderScope({"filefeeder://train/handle/0/eps_scope/train/interleave/0"}) == "train",
          "native reader lost its input scope");
    input::RegisterReader("eval");  // An unrelated live input must not enter this cut.
    const auto ended = input::RegisterReader("train");
    input::Cursor ended_cursor{"filefeeder://train/handle/0/eps_scope/train/interleave/0",
                               0, {0}, false, input::no_assignment, "", 0};
    input::ReaderCompleted(ended, ended_cursor); input::ReaderDestroyed(ended);
    const auto reading = input::RegisterReader("train");
    input::Cursor live_cursor{"filefeeder://train/handle/0/eps_scope/train/interleave/1",
                              0, {1}, true, 1, "second.orc", 2};
    input::BeginScopedCapture("train");
    input::Capture("active", live_cursor, reading);
    input::FinishCapture(argv[1]);
    auto cut = input::DecodeCapture(input::ReadFile(argv[1]));
    Check(cut.size() == 2, "destroyed completed reader history disappeared");
    std::vector<input::Cursor> saved;
    for (const auto& item : cut) saved.push_back(item.second);
    Check(input::RestoreCut(source, saved).excluded == std::vector<std::uint64_t>({0, 1}),
          "completed leaf assignments were requeued");
    std::filesystem::remove(argv[1]);

    // A completion after Begin is not proof that its rows are in the saved
    // upstream buffers. An uncaptured reader forces a new serialization cut.
    const auto late_reader = input::RegisterReader("train");
    auto late_cursor = ended_cursor; late_cursor.assignments = {2};
    input::BeginScopedCapture("train");
    input::Capture("active", live_cursor, reading);
    input::ReaderCompleted(late_reader, late_cursor);
    try { input::FinishCapture(argv[1]); throw std::logic_error("unstable reader cut accepted"); }
    catch (const std::runtime_error& error) {
      Check(std::string(error.what()).find("retry input capture:") == 0, "wrong capture rejection");
    }
    Check(!std::filesystem::exists(argv[1]), "rejected input cut published metadata");
    input::CancelCapture();
    input::BeginScopedCapture("train");
    input::Capture("active", live_cursor, reading);
    input::FinishCapture(argv[1]);
    Check(input::DecodeCapture(input::ReadFile(argv[1])).size() == 3, "stable retry lost completed readers");
    std::filesystem::remove(argv[1]);

    input::BeginScopedCapture("train");
    const auto new_reader = input::RegisterReader("train");
    auto new_cursor = ended_cursor; new_cursor.assignments = {3};
    input::ReaderCompleted(new_reader, new_cursor);
    input::Capture("active", live_cursor, reading);
    try { input::FinishCapture(argv[1]); throw std::logic_error("new reader cut accepted"); }
    catch (const std::runtime_error& error) {
      Check(std::string(error.what()).find("retry input capture:") == 0, "wrong new-reader rejection");
    }
    input::CancelCapture();
    input::BeginInputGeneration("train");
    input::BeginScopedCapture("train");
    input::FinishCapture(argv[1]);
    Check(input::DecodeCapture(input::ReadFile(argv[1])).empty(), "previous input generation leaked into new cut");
    std::filesystem::remove(argv[1]);
    std::filesystem::remove(std::string(argv[1]) + ".readers.json");
    std::cout << "{\"test\":\"input_checkpoint_cut\",\"total_rows\":20,\"consumed_rows\":5,"
                 "\"remaining_rows\":15,\"duplicate_filename_occurrences\":true,"
                 "\"post_snapshot_assignments_requeued\":true,\"concurrent_source_fetches\":true,\"concurrent_streams\":8}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
