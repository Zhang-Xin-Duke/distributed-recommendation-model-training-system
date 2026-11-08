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

## 用工作负载决定策略

先用固定 Embedding + MLP 做受控对照，找出策略改变结果的条件，再用公开推荐数据的固定子集做复核。推荐验证可选 Criteo 点击日志的可取得版本，记录数据来源、时间划分和样本清单，按固定时间切分训练和验证。Embedding + MLP 和子集是本项目的验证负载。质量目标计时参考 [MLPerf Training 的 time-to-quality 方法](https://mlcommons.org/benchmarks/training/)。异步等待与收敛之间的假设来自 [OSDI Parameter Server 论文第 5.1 节](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf)。

| 工作负载 | 如何生成或确认 | 要检验的假设 |
| --- | --- | --- |
| Worker 速度接近 | 同样的 local batch，关闭人工延迟，检查 Worker 阶段耗时分布 | 同步屏障的等待较少，异步的收益可能有限 |
| 一个 Worker 持续慢 | 固定 Worker 受控延迟，扫描其正常 batch 耗时的约 0.25、0.5、1 倍，另保留无延迟组 | 区分有限领先能吸收的差异与长期慢 Worker 造成的积压 |
| 间歇慢 Worker | 预先生成相同的延迟序列，以少量长停顿与持续短停顿做平均延迟匹配 | 有界异步可能吸收短时波动，同时控制领先 |
| Embedding 键低重叠与热点共享 | 固定维度、batch、bag 长度及 Worker 各自的 ID 热度分布，通过重映射改变跨 Worker 热键重叠。验证观测到的重叠率和各分片负载 | 反复更新同一行时，更新交错和较旧参数可能增加达到目标所需的样本 |
| 参数服务开销较高 | 单独改变每批触及的键数或梯度 payload，记录服务队列、RPC 字节和等待。对所有策略使用相同条件 | 确认吞吐由参数分片决定时，增加并发会隐藏等待还是增加排队 |

第一轮筛选用三种 Worker 延迟形态与两档 Embedding 键重叠，比较 A、B、C、D。第二轮只对结论有变化的区域加密延迟档位或单独加参数服务压力。每个负载内比较策略。跨负载更改模型或数据时重新确定相同组间目标，不能跨不同目标比较绝对时间。Embedding 键的重叠和共享 MLP 的更新分别记录，因为低 Embedding 重叠时 MLP 仍共享。

每个条件至少五次配对独立重复，配对共享随机种子、样本与延迟序列，轮换策略顺序。预跑后固定数据划分、损失门槛、评价频率、样本预算和最小实际收益。发布每次运行的数据及配对差值。差异小于预设实际收益，或波动范围无法区分时，保留「无明确优势」。报告各组预算内达标率，避免只统计成功达标的运行。调参使用单独的开发集，最终指标在预先固定的验证集上报告。分别报告固定超参比较与等预算调优结果。

结论表按工作负载列出达到同一验证质量的耗时、样本数、达标率、端到端吞吐以及分片瓶颈。若异步的达标样本数增加，就检查样本吞吐提升是否抵消了这部分额外工作。仅 Worker 等待减少时，结论只覆盖等待。对有界异步还要报告准入领先，以及能够可靠测量的读版本或分片更新差，避免用 Worker 的轮次领先代替参数实际陈旧度。

单张 5090 用于筛选和复核共享设备条件下的策略。多机判断需要在至少两个独立 GPU 节点、真实网络上复跑有差异的 workload 和无明确优势的边界组。固定拓扑、网络路径和故障条件，分别报告健康运行和恢复后的达标时间。异步恢复尚未实现时，只报告其健康运行结果。

## Push 窗口对照

在同一异步实现上只改变单个发送器允许在途的 batch Push 数量，比较上限 1 和 10。每个 batch Push 可以扇出到多个分片。它占用一个名额，直到该批所有分片回调结束。记录每批的输入位置、发出和完成时间、状态以及发送器的在途峰值。RPC 失败须进入结果记录并让该次试验失败。显式排空所有在途 Push 后才能读取最终状态和验证 loss。

现有逐 Worker 训练入口在分片确认后才报告本轮完成。仅加一个计数器会让窗口 10 实际仍停在 1。接入时需要让发送器在同一训练进程里真正存在多个未完成的 batch Push，并记录下一批参数读取与前一批回调的先后关系。先用受控回调延迟证明上限 1 和 10 都会实际触发，再开展无注入的收敛对照。窗口可能改变读参数和应用更新的交错顺序，这是该对照要观察的行为。两组使用同样的训练和评估流程，报告固定样本量的验证 loss、达标耗时、吞吐及全部 Push 排空时间。

## 单卡 GPU 环境预检

以下命令适用于 RTX 5090 或 RTX A6000 等单卡 Linux 主机。从仓库根目录执行，假设 Linux 环境已经安装 [BRPC 构建依赖](../RUNNING.md)，当前 Python 环境有 `pybind11`、NumPy 和可在这台机器上执行 GPU 算子的 TensorFlow。参数分片在 CPU 上运行，所以这里编译的是 CPU 参数服务与 Python 适配器。GPU 模型计算由当前 TensorFlow 环境提供。

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

### 运行策略对照

在项目根目录使用已安装 TensorFlow 的 Python 环境。先运行四组短测，检查 GPU 设备、Worker 进度和提交样本数。

```bash
python tests/tensorflow_strategy_workloads.py --build build/5090 --device /GPU:0 \
  --patterns balanced --overlap shared --iterations 4 --repeats 1 \
  --out build/5090/results/strategy-smoke.json
```

默认矩阵包含 3 种 Worker 延迟、2 种键重叠、4 种策略、3 个样本预算端点和 5 次配对重复，共 360 次独立运行。预跑后固定验证门槛，再用 `--target-loss` 指定。省略 `--patterns`、`--overlap`、`--iterations` 和 `--repeats` 即运行默认矩阵。CPU 冒烟可指定 `--device /CPU:0`。结果 JSON 包含逐次记录、各策略达标率和配对比较，达标时间按首个通过的实测端点报告。

`--dataset` 接收 JSON，其中 `batches[worker][batch][example]` 为两个 Worker 的等大小 batch 流，`validation[example]` 为独立验证集。每条样本形如 `{"keys":[11,12],"label":1}`，键需重映射至 `[11, 32768)`。复核公开数据时，另存数据来源与切分清单。

默认在一台机器上启动两个 Worker、协调器及参数分片。两个 Worker 使用同一指定设备。

双机运行时，在两台 Linux 主机准备同一提交、TensorFlow 和 BRPC 构建。主机 A 运行脚本、Worker 0、协调器和两个参数分片。主机 B 运行 Worker 1。先验证 A 能用非交互 SSH 登录 B，且 B 的 SSH 服务允许反向端口转发。RPC 端点只在两端回环地址监听，Worker 1 的 RPC 经 A 到 B 的 SSH 会话返回 A。

```bash
python tests/tensorflow_strategy_workloads.py --build build/5090 --device /GPU:0 \
  --remote-worker worker-b /opt/eps /opt/eps-venv/bin/python /opt/eps/build/5090 \
  --patterns balanced --overlap shared --iterations 4 --repeats 1 \
  --out build/5090/results/two-host-smoke.json
```

远端四个参数依次为 A 上可用的 SSH 主机名、B 上的仓库绝对路径、Python 绝对路径和构建目录绝对路径。双机时由 A 的单一时钟计时，从放行到两个 Worker 进程退出，包含 SSH 与退出开销。保留节点、网络和 GPU 信息，并在同一拓扑上比较策略。单机和双机计时范围分别记录。Push 窗口 1 与 10 的对照仍需独立接入。
