  eps::input_checkpoint::Cursor CheckpointCursor() const {
    eps::input_checkpoint::Cursor cursor;
    cursor.stream = name_; cursor.handle = handle_; cursor.assignments = assignment_ids_;
    cursor.active = bool(underlying_) && !filenames_.empty();
    if (cursor.active) {
      cursor.current = assignment_ids_.back(); cursor.filename = filenames_.back();
    }
    return cursor;
  }

  Status RestoreCursor(const eps::input_checkpoint::Cursor& cursor) {
    eps::input_checkpoint::Validate(cursor);
    if (!cursor.active || cursor.stream != name_ || cursor.handle != handle_)
      return errors::InvalidArgument("restored FileFeeder stream identity differs");
    assignment_ids_ = cursor.assignments; filenames_ = {cursor.filename}; global_pos_ = 0;
    return Env::Default()->NewRandomAccessFile(cursor.filename, &underlying_);
  }

