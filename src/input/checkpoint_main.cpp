#include "adapters/entropy/input_checkpoint.h"

#include <cerrno>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <unistd.h>

namespace input = eps::input_checkpoint;

int main(int argc, char** argv) {
  try {
    std::string source_path, output, inspect;
    std::vector<std::string> capture_paths;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (++i == argc) throw std::invalid_argument("input checkpoint option requires a value");
      if (option == "--source") source_path = argv[i];
      else if (option == "--capture") capture_paths.emplace_back(argv[i]);
      else if (option == "--output") output = argv[i];
      else if (option == "--inspect-capture") inspect = argv[i];
      else throw std::invalid_argument("unknown input checkpoint option");
    }
    std::vector<input::Cursor> cursors;
    if (!inspect.empty()) capture_paths.push_back(inspect);
    if (capture_paths.empty() || capture_paths.size() > 1024)
      throw std::invalid_argument("provide one to 1024 input capture files");
    std::uint64_t bytes = 0;
    for (const auto& path : capture_paths) {
      const auto raw = input::ReadFile(path); bytes += raw.size();
      if (bytes > input::byte_limit) throw std::invalid_argument("input capture group exceeds metadata budget");
      for (const auto& item : input::DecodeCapture(raw)) cursors.push_back(item.second);
    }
    if (!inspect.empty()) {
      if (!source_path.empty() || !output.empty() || capture_paths.size() != 1)
        throw std::invalid_argument("capture inspection cannot publish a restore plan");
      std::size_t active = 0, assignments = 0;
      for (const auto& cursor : cursors) { active += cursor.active; assignments += cursor.assignments.size(); }
      std::cout << "{\"streams\":" << cursors.size() << ",\"active_streams\":" << active
                << ",\"tracked_assignments\":" << assignments << ",\"reader_rows\":[";
      for (std::size_t i = 0; i < cursors.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << cursors[i].row;
      }
      std::cout << "]}\n"; return 0;
    }
    if (source_path.empty() || output.empty()) throw std::invalid_argument("source and output paths are required");
    const auto plan = input::RestoreCut(input::DecodeFiles(input::ReadFile(source_path)), cursors);
    const auto encoded = input::Encode(plan);
    if (std::filesystem::exists(output)) {
      if (input::ReadFile(output) != encoded) throw std::invalid_argument("immutable input restore plan differs");
    } else {
      const auto pending = output + ".pending-" + std::to_string(getpid());
      input::WriteFile(pending, encoded);
      if (link(pending.c_str(), output.c_str()) != 0) {
        const auto error = errno; std::filesystem::remove(pending);
        if (error != EEXIST) throw std::system_error(error, std::generic_category(), "input plan publication");
        if (input::ReadFile(output) != encoded) throw std::invalid_argument("concurrent input restore plan differs");
      } else std::filesystem::remove(pending);
    }
    std::cout << "{\"handle\":" << plan.handle << ",\"files\":" << plan.filenames.size()
              << ",\"excluded_assignments\":" << plan.excluded.size() << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
