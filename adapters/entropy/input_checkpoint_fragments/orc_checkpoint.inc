    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock lock(mu_);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name(kCurrentFileIndex), current_file_index_));
      auto cursor = saved_cursor_;
      if (file_) cursor = tensorswitch::CaptureFileFeeder(file_.get());
      if (cursor.stream.empty()) cursor.stream = prefix();
      if (reader_ && cursor.active) { cursor.filename = reader_->CheckpointFilename(); cursor.row = reader_->CheckpointRow(); }
      const auto bytes = eps::input_checkpoint::Encode(cursor);
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("eps_input_cursor_v1"), tstring(bytes)));
      eps::input_checkpoint::Capture(full_name("eps_input_cursor_v1"), cursor);
      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock lock(mu_);
      ResetStreamsLocked();
      int64 current;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kCurrentFileIndex), &current));
      if (current < 0 || std::uint64_t(current) > dataset()->filenames_.size())
        return errors::InvalidArgument("restored ORC file index is outside the dataset");
      current_file_index_ = current;
      if (!reader->Contains(full_name("eps_input_cursor_v1"))) {
        // Earlier checkpoints retain their original filename-index behavior.
        return OkStatus();
      }
      tstring bytes;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("eps_input_cursor_v1"), &bytes));
      try {
        saved_cursor_ = eps::input_checkpoint::DecodeCursor(std::string(bytes.data(), bytes.size()));
      } catch (const std::exception& error) {
        return errors::InvalidArgument(error.what());
      }
      if (!saved_cursor_.active) return OkStatus();
      if (current_file_index_ >= dataset()->filenames_.size())
        return errors::InvalidArgument("active ORC reader is beyond dataset EOF");
      const auto& stream = dataset()->filenames_[current_file_index_];
      if (saved_cursor_.stream != stream)
        return errors::InvalidArgument("restored ORC stream differs from the dataset");
      if (saved_cursor_.handle == eps::input_checkpoint::no_assignment) {
        TF_RETURN_IF_ERROR(ctx->env()->NewRandomAccessFile(stream, &file_));
      } else {
        TF_RETURN_IF_ERROR(tensorswitch::RestoreFileFeeder(stream, saved_cursor_, &file_));
      }
      reader_ = absl::make_unique<io::OrcRowReader>(file_.get(), dataset()->field_name_);
      return reader_->RestoreRow(saved_cursor_.row);
    }

