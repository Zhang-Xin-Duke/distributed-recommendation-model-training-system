# 单张 RTX 5090 上的同步与异步训练对照设计

目标是比较当前项目的同步与异步训练，在相同数据和单机硬件条件下，哪种更快达到事先定义的验证损失。用 Worker 等待、分片提交时延和收敛曲线解释差异。

## 要补齐的历史实验

现有验证已经测过不同 Worker 放行策略。它们都使用逐 Worker 更新，因此还需要在同一套评估下比较按轮聚合的同步训练与逐 Worker 更新的异步训练。另做一组窗口 1 与 10 的对照，测量单个发送器允许更多未完成 Push 时，吞吐与收敛如何变化。窗口对照固定 Worker 放行策略，放行策略对照固定 Push 行为。

## 同步与异步对照组

| 组 | 进度策略 | 参数更新方式 | 回答的问题 |
| --- | --- | --- | --- |
| A | Sequential | 所有 Worker 的梯度按样本数加权归并，每组更新一次 Adam | 当前同步训练基线 |
| B | Eventual | Worker 完成自己的 batch 后独立提交一次 Adam 更新 | 当前异步路径整体表现 |
| C | Bounded Delay，上限 2 | 与 B 相同 | 限制领先能否改善异步路径 |
| D | Sequential | 与 B 相同 | 控制更新方式后，单独比较 Worker 放行策略 |

A、B 是主比较，包含两个系统实际采用的更新规则。D、B、C 保持更新方式相同，用来拆分调度的影响。A 和 D 的更新频率与梯度聚合方式不同，比较时同时报告处理样本数和 Adam 更新次数。

## 工作负载与控制

- 在一张 5090 上启动两个独立 TensorFlow Worker，两个 CPU 参数分片和一个协调器，通过实际 BRPC 通信。两个 Worker 共享 GPU，参数放置和缓存配置在各组保持一致。
- 先用已有 Embedding + MLP 工作负载检查 GPU 执行和更新计数。性能比较使用固定的训练和验证数据划分，并把训练长度增加到足以区分启动抖动与稳态表现。
- 每组配对使用同一初始化、样本序列、各 Worker 的 local batch size 和可比较的 Adam 配置。两个 Worker 处理相同样本总数。保持线程、显存分配方式、同机负载与进程布局一致。
- 分开跑无人工延迟和有慢 Worker 两种条件。慢 Worker 条件先沿用每任务 20 ms 的受控延迟，放在已知位置并记录实际延迟，供解释结果。每个条件至少做五次配对重复，轮换运行顺序。
- 先用独立预跑确定各组可用的测量区间和验证目标，再冻结目标、样本预算和超时。主比较使用预先固定的超参。若需要比较各路径调优后的性能，另给每组相同的调参预算，单独报告。

## 测量和验收

- 主指标为达到固定验证 loss 的实际训练时间。预热后同时放行，按相同的已处理样本数评估。评估使用一致的已完成检查点或各组相同的独立 fresh-run 端点。使用独立端点时报告首次通过的测量端点，不插值为精确达标时刻。
- 同时报告固定样本预算下的验证 loss、样本吞吐、每个 Worker 的放行等待、梯度提交时延、Adam 更新次数、GPU 利用率和峰值显存。逐次保留配对结果与波动范围。预算内未达标的试验单独计数。
- 对 A 检查每个分片每个全局轮次只执行一次更新。对 B、C、D 检查每个 Worker 已提交任务的更新计数。C 的任务准入领先不得超过设定值，所有组都检查模型确实在 GPU 计算、状态有限、进程正常退出。
- 在同一负载条件下，优先看哪组更稳定地达到目标，再看达标耗时的配对差值及波动。只在有慢 Worker 时观察到收益，就把结论限定在该条件。单卡结果覆盖进程级协调和共享设备性能，跨机器网络与多 GPU 扩展需要独立实验。

## Push 窗口对照

在同一异步实现上只改变单个发送器允许在途的 batch Push 数量，比较上限 1 和 10。每个 batch Push 可以扇出到多个分片。它占用一个名额，直到该批所有分片回调结束。记录每批的输入位置、发出和完成时间、状态以及发送器的在途峰值。RPC 失败须进入结果记录并让该次试验失败。显式排空所有在途 Push 后才能读取最终状态和验证 loss。

现有逐 Worker 训练入口在分片确认后才报告本轮完成。仅加一个计数器会让窗口 10 实际仍停在 1。接入时需要让发送器在同一训练进程里真正存在多个未完成的 batch Push，并记录下一批参数读取与前一批回调的先后关系。先用受控回调延迟证明上限 1 和 10 都会实际触发，再开展无注入的收敛对照。窗口可能改变读参数和应用更新的交错顺序，这是该对照要观察的行为。两组使用同样的训练和评估流程，报告固定样本量的验证 loss、达标耗时、吞吐及全部 Push 排空时间。

## 5090 接入后立即执行

以下命令从仓库根目录执行，假设 Linux 环境已经安装 [BRPC 构建依赖](../RUNNING.md)，当前 Python 环境有 `pybind11`、NumPy 和可在这台机器上执行 GPU 算子的 TensorFlow。参数分片在 CPU 上运行，所以这里编译的是 CPU 参数服务与 Python 适配器。GPU 模型计算由当前 TensorFlow 环境提供。

```sh
git rev-parse HEAD
nvidia-smi -L
python - <<'PYGPU'
import tensorflow as tf

tf.config.set_soft_device_placement(False)
gpus = tf.config.list_physical_devices("GPU")
assert len(gpus) == 1, gpus
tf.config.experimental.set_memory_growth(gpus[0], True)
with tf.device("/GPU:0"):
    x = tf.ones((32, 32))
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = tf.reduce_sum(x @ x)
    grad = tape.gradient(loss, x)
assert grad is not None and "GPU:0" in grad.device, grad
print(tf.__version__, gpus, grad.device)
PYGPU

cmake -S . -B build/5090 -DCMAKE_BUILD_TYPE=Release \
  -DEPS_BUILD_BRPC=ON -DEPS_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/5090 -j 6
python tests/tensorflow_consistency_training.py --build build/5090 \
  --device /GPU:0 --workers 2 --iterations 8 --repeats 1 --delay-ms 20 \
  --out build/5090/results/per-worker-smoke.json
```

预检和 smoke 都通过后，用现有脚本收集 B、C、D 的受控参考。这个脚本当前输出的是固定训练集的 loss。正式主实验要使用下面约定的固定验证集。

```sh
for delay in 0 20; do
  python tests/tensorflow_consistency_training.py --build build/5090 \
    --device /GPU:0 --workers 2 --iterations 2,4,8,16 --repeats 5 \
    --delay-ms "$delay" --target-loss 0.5 \
    --out "build/5090/results/per-worker-delay-${delay}.json"
done
```

检查报告中的 `complete`、`gpu`、`trials`、`first_tested_endpoints_meeting_target` 和各 Worker 的设备及最大领先轮数。环境预检失败时记录实际报错，避免把 CPU 回退记录成 GPU 数据。

## 正式对照的接入顺序

1. 在 `tests/tensorflow_consistency_training.py` 中扩展现有 harness。给每组独立指定进度策略和更新方式，为 A 的每个参数分片配置 `synchronous_training`，设 `workers=2`、`reduction=sample_mean`。沿用 `tests/tensorflow_synchronous_training.py` 里启动同步梯度收集器的配置。A 和 B、C、D 都使用相同的 CPU 参数分片配置和同一份训练、验证划分。同步组沿用 `examples/tensorflow_training_worker.py` 的同步提交入口。
2. 统一端点记录和验收。在同步组按全局轮次检查 Adam 更新数，在逐 Worker 组按已提交的 Worker 任务检查。验证集由独立、固定的样本生成规则产生，确保评估用的键可读取，并保留训练与评估样本的清单。先短跑两种策略，确认更新数、最终状态、GPU 设备和输出指标，再启动配对重复。结果写入 `build/5090/results/sync-vs-async/`。
3. 在相同异步配置上接入批级 Push 窗口，比较 1 与 10。先验收实际在途深度和排空行为，再测训练效果，结果写入 `build/5090/results/push-window/`。若窗口 10 的运行仍只有 1 个在途 batch，将该轮标记为窗口未生效。

预检与逐 Worker 对照使用仓库现有脚本。完整的 A/B 同脚本比较和 Push 窗口对照需要按上述步骤接入，实验结果尚未生成。
