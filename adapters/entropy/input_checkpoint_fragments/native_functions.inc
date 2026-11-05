namespace tensorswitch {

eps::input_checkpoint::Cursor CaptureFileFeeder(tensorflow::RandomAccessFile* file) {
  tensorflow::StringPiece name;
  if (!file->Name(&name).ok()) throw std::runtime_error("input stream name is unavailable");
  if (name.find("filefeeder://") == 0)
    return static_cast<tensorflow::FFRandomAccessFile*>(file)->CheckpointCursor();
  eps::input_checkpoint::Cursor cursor;
  cursor.stream = std::string(name.data(), name.size()); cursor.active = true; cursor.filename = cursor.stream;
  return cursor;
}

tensorflow::Status RestoreFileFeeder(const std::string& name,
    const eps::input_checkpoint::Cursor& cursor,
    std::unique_ptr<tensorflow::RandomAccessFile>* result) {
  auto* file = new tensorflow::FFRandomAccessFile(name, cursor.handle);
  result->reset(file);
  auto status = file->RestoreCursor(cursor);
  if (!status.ok()) result->reset();
  return status;
}

template <class Function> int InputCheckpointOperation(Function function) {
  try { function(); return 0; }
  catch (const std::exception& error) {
    LOG(ERROR) << "Native input checkpoint failed: " << error.what(); return -1;
  }
}

TS_REGISTER_LOCAL_FUNCTION("eps_input_begin_capture", []() -> int {
  return InputCheckpointOperation([] { eps::input_checkpoint::BeginCapture(); });
});
TS_REGISTER_LOCAL_FUNCTION("eps_input_begin_generation", [](const std::string& scope) -> int {
  return InputCheckpointOperation([&] { eps::input_checkpoint::BeginInputGeneration(scope); });
});
TS_REGISTER_LOCAL_FUNCTION("eps_input_begin_scoped_capture", [](const std::string& scope) -> int {
  return InputCheckpointOperation([&] { eps::input_checkpoint::BeginScopedCapture(scope); });
});
TS_REGISTER_LOCAL_FUNCTION("eps_input_finish_capture", [](const std::string& path) -> int {
  return InputCheckpointOperation([&] { eps::input_checkpoint::FinishCapture(path); });
});
TS_REGISTER_LOCAL_FUNCTION("eps_input_cancel_capture", []() -> int {
  eps::input_checkpoint::CancelCapture(); return 0;
});
TS_REGISTER_LOCAL_FUNCTION("eps_input_export_source", [](int64_t handle, const std::string& path) -> int {
  return InputCheckpointOperation([&] {
    if (handle < 0 || handle >= server::InputFiles::kResourceCount)
      throw std::invalid_argument("input source handle is outside the roster");
    auto source = eps::input_checkpoint::ExportSource(server::InputFiles::instance(handle), handle);
    eps::input_checkpoint::WriteFile(path, eps::input_checkpoint::Encode(source));
  });
});
TS_REGISTER_LOCAL_FUNCTION("eps_input_restore_source", [](int64_t handle, const std::string& path) -> int {
  return InputCheckpointOperation([&] {
    if (handle < 0 || handle >= server::InputFiles::kResourceCount)
      throw std::invalid_argument("input source handle is outside the roster");
    auto source = eps::input_checkpoint::DecodeFiles(eps::input_checkpoint::ReadFile(path));
    if (source.handle != std::uint64_t(handle)) throw std::invalid_argument("input source identity differs");
    eps::input_checkpoint::RestoreSource(server::InputFiles::instance(handle), source);
  });
});

}  // namespace tensorswitch

extern "C" __attribute__((visibility("default"))) int eps_entropy_input_checkpoint_version() {
  return 2;
}
