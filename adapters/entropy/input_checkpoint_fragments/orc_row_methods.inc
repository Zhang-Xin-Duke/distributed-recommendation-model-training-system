  const std::string& CheckpointFilename() const { return checkpoint_filename_; }
  std::uint64_t CheckpointRow() const { return checkpoint_row_; }
  Status RestoreRow(std::uint64_t row) {
    if (row > reader_->getNumberOfRows()) return errors::InvalidArgument("ORC row is beyond EOF");
    try {
      if (row == reader_->getNumberOfRows()) { row_reader_.reset(); batch_.reset(); }
      else { row_reader_->seekToRow(row); batch_ = row_reader_->createRowBatch(1024); }
      next_row_index_ = 0; checkpoint_row_ = row;
      return OkStatus();
    } catch (const std::exception& error) {
      return errors::InvalidArgument(error.what());
    }
  }

