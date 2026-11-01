"""Map TensorSwitch SparseLazyAdam options to an external parameter table."""
import math


def sparse_table_definition(table_id, dimension, optimizer, *, seed=0,
                            initialization=None, adam_share_segment=0):
    """Preserve supported optimizer options without changing the model's gradients.

    `optimizer` is the existing `ts.train.SparseLazyAdam` instance. An explicit
    initialization pair supports the framework's constant initializer. Otherwise
    `initial_range` supplies the symmetric uniform sampling interval.
    """
    if getattr(optimizer, "name", None) != "SparseLazyAdam":
        raise ValueError("external sparse tables require SparseLazyAdam")
    if not 0 <= table_id < 2**32 or not 1 <= dimension <= 65536 or not 0 <= seed < 2**64:
        raise ValueError("invalid table ID, dimension or initialization seed")
    options = optimizer.options
    if adam_share_segment or any(getattr(options, field, 0) for field in
                                ("weight_decay", "non_opt_dim", "adam_share_segment")):
        raise ValueError("shared moments, weight decay and non-optimized dimensions are unsupported")
    policies = ("_embedding_create_threshold", "_embedding_purge_threshold",
                "_embedding_serving_purge_threshold", "_embedding_latest_show_decay_rate",
                "_embedding_time_decay_threshold", "_embedding_cache", "_embedding_fuzz")
    if any(getattr(optimizer, key, 0) for key in policies) or getattr(optimizer, "_purge_policy", "NoPurge") != "NoPurge":
        raise ValueError("feature creation, purge and decay policies require separate integration")
    fields = dict(kind="sparse_lazy_adam", learning_rate=float(options.learning_rate),
                  beta1=float(options.beta1), beta2=float(options.beta2), epsilon=float(options.epsilon),
                  initial_beta1_power=float(options.beta1_power), initial_beta2_power=float(options.beta2_power))
    if not all(math.isfinite(v) for k, v in fields.items() if k != "kind"):
        raise ValueError("optimizer options must be finite")
    if fields["learning_rate"] <= 0 or fields["epsilon"] <= 0 or any(
            not 0 <= fields[key] < 1 for key in ("beta1", "beta2", "initial_beta1_power", "initial_beta2_power")):
        raise ValueError("invalid sparse Adam hyperparameters")
    bounds = list(options.weight_bounds)
    if bounds:
        fields["weight_bounds"] = _interval(bounds, "weight bounds")
    interval = initialization if initialization is not None else (-float(options.initial_range), float(options.initial_range))
    lo, hi = _interval(interval, "initialization")
    return dict(id=int(table_id), dimension=int(dimension), dtype="float32", optimizer=fields,
                initialization=dict(min=lo, max=hi, seed=int(seed)))


def _interval(values, name):
    if len(values) != 2:
        raise ValueError(name + " requires two endpoints")
    lo, hi = map(float, values)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError("invalid " + name)
    return [lo, hi]
