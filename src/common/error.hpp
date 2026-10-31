#pragma once

#include <stdexcept>
#include <utility>

#include "eps/types.hpp"

namespace eps::detail {

class Failure : public std::runtime_error {
 public:
  explicit Failure(Error error)
      : std::runtime_error(error.message), error_(std::move(error)) {}
  const Error& error() const { return error_; }
 private:
  Error error_;
};

inline void Require(bool condition, const char* message,
                    Error::Code code = Error::Code::invalid_argument) {
  if (!condition) throw Failure({code, message});
}

}  // namespace eps::detail
