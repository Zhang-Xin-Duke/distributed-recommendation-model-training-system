    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      mutex_lock lock(input_mu_);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("eps_partition_version"), int64_t(1)));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("eps_first_eof"), int64_t(first_batch_eof_)));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("eps_first_size"), int64_t(first_batch_.size())));
      for (std::size_t i = 0; i < first_batch_.size(); ++i)
        TF_RETURN_IF_ERROR(writer->WriteTensor(full_name("eps_first_" + std::to_string(i)), first_batch_[i]));
      const auto state = dataset()->partition_queue_->CheckpointSnapshot();
      Tensor queued(DT_INT32, TensorShape({int64_t(state.queue.size())}));
      for (std::size_t i = 0; i < state.queue.size(); ++i) queued.flat<int32>()(i) = state.queue[i].partition_id;
      TF_RETURN_IF_ERROR(writer->WriteTensor(full_name("eps_partition_queue"), queued));
      TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("eps_last_partition"), int64_t(state.last_peek.partition_id)));
      return OkStatus();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock lock(input_mu_);
      if (!reader->Contains(full_name("eps_partition_version")))
        return errors::FailedPrecondition("input checkpoint has no prepared batch or partition queue");
      int64_t version, eof, size, last;
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("eps_partition_version"), &version));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("eps_first_eof"), &eof));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("eps_first_size"), &size));
      TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("eps_last_partition"), &last));
      if (version != 1 || (eof != 0 && eof != 1) || size < 0 || size > 1048576 ||
          last < -2 || last > std::numeric_limits<int>::max())
        return errors::InvalidArgument("invalid prepared batch checkpoint");
      std::vector<Tensor> first(size);
      for (int64_t i = 0; i < size; ++i)
        TF_RETURN_IF_ERROR(reader->ReadTensor(ctx->flr(), full_name("eps_first_" + std::to_string(i)), &first[i]));
      Tensor queued;
      TF_RETURN_IF_ERROR(reader->ReadTensor(ctx->flr(), full_name("eps_partition_queue"), &queued));
      if (queued.dtype() != DT_INT32 || queued.dims() != 1 || queued.NumElements() > 1048576)
        return errors::InvalidArgument("invalid partition notification checkpoint");
      decltype(dataset()->partition_queue_->CheckpointSnapshot()) state;
      for (int64_t i = 0; i < queued.NumElements(); ++i) {
        const auto partition = queued.flat<int32>()(i);
        if (partition < -2) return errors::InvalidArgument("invalid queued partition");
        state.queue.push_back(tensorswitch::data::CpuPartitionType{partition});
      }
      state.last_peek.partition_id = int(last);
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      first_batch_ = std::move(first); first_batch_eof_ = bool(eof);
      dataset()->partition_queue_->RestoreCheckpoint(std::move(state));
      return OkStatus();
    }

