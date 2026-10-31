#include "update/lazy_adam.hpp"
#include <cmath>
#include "common/error.hpp"

namespace eps::detail {
void ValidateLazyAdam(const TableState& t, const RowGradients& g) {
  const auto dim = t.definition.dimension;
  const auto& o = t.definition.optimizer;
  const auto lr = o.learning_rate * std::sqrt(1 - t.beta2_power) / (1 - t.beta1_power);
  Require(std::isfinite(lr), "optimizer learning-rate overflow");
  for (std::size_t i = 0; i < g.rows.size(); ++i) {
    const auto row = t.At(t.index.at(g.rows[i].key));
    for (std::uint32_t d = 0; d < dim; ++d) {
      const auto grad = g.values[i * dim + d];
      const auto m = o.beta1 * row.first[d] + (1 - o.beta1) * grad;
      const auto v = o.beta2 * row.second[d] + (1 - o.beta2) * grad * grad;
      const auto w = row.weights[d] - lr * m / (std::sqrt(v) + o.epsilon);
      Require(std::isfinite(m) && std::isfinite(v) && std::isfinite(w),
              "optimizer state would overflow");
    }
  }
}
void ApplyLazyAdam(TableState& t, const RowGradients& g) {
  const auto dim = t.definition.dimension;
  const auto& o = t.definition.optimizer;
  const auto lr = o.learning_rate * std::sqrt(1 - t.beta2_power) / (1 - t.beta1_power);
  for (std::size_t i = 0; i < g.rows.size(); ++i) {
    auto row = t.At(t.index.at(g.rows[i].key));
    for (std::uint32_t d = 0; d < dim; ++d) {
      const auto grad = g.values[i * dim + d];
      row.first[d] = o.beta1 * row.first[d] + (1 - o.beta1) * grad;
      row.second[d] = o.beta2 * row.second[d] + (1 - o.beta2) * grad * grad;
      row.weights[d] -= lr * row.first[d] / (std::sqrt(row.second[d]) + o.epsilon);
    }
    ++*row.updates;
  }
  ++t.optimizer_sequence;
  t.beta1_power *= o.beta1;
  t.beta2_power *= o.beta2;
}
}  // namespace eps::detail
