# 执行一组 H3 镜头实验

以下命令从本仓库根目录运行。先选一个有合适镜头和条件包的单属性样本，再扩展到整组。

## 复跑历史红车样本

在原 H3 worker 主机的本仓库目录运行。脚本使用历史沿海公路广告的 seed 6101，并依次提交原包重放、车色方向和等强度随机方向。方向沿用历史供体和强度 4，使用本工程的正交投影实现。它无需录制新母版。

```sh
H3_EXPERIMENT_BASE=/path/to/h3-auto-ad-vector-edit \
H3_PYTHON=/path/to/h3/venv/bin/python \
bash scripts/replay-historical-color.sh
```

先确保历史包和供体包还在 worker 指向的目录，并且本机端口 8192 的 worker 已启动。必要时用 `LIENYU_H3_PACKAGE_ROOT` 和 `H3_BRIDGE_SCRIPT` 指定自定义路径。任选其他属性或 seed 时，按下文分步执行。

## 1. 生成计划

复制 `experiments/spec.example.json` 到 `experiments/spec.local.json`。为每个镜头填入母版条件包中的实际 seed，以及录制时使用的宽、高、时长、采样步数和模型模式。对于 Ref2VA，按原始请求配置 `settings.reference_model: true`。H3 worker 使用的模型及工作流也需要与录制时一致。旧 manifest 只记录载荷和 seed，CLI 无法从中推断模型权重、工作流及采样参数。

```sh
python3 -m vehicle_edit_lab plan \
  --phase pilot --attribute vehicle.paint_color \
  --spec experiments/spec.local.json --output outputs/paint-pilot.json
```

每个任务有 `id`、`variant`、`base_package`、采样设置及依赖。A3 在第一镜头复用 A1；第二镜头的 `first_frame_from_job` 指向上一镜头的 A3 结果。轮毂和车窗的探索镜头必须让目标清晰可辨；示例尺寸需按模型支持和实际素材调整。

## 2. 学习方向并准备配对包

下面的变量用本机路径赋值。编辑工具和既有 ComfyUI worker 都应访问同一个 `PACKAGE_ROOT`。启动 worker 时设置 `LIENYU_H3_PACKAGE_ROOT`，已有 worker 环境需单独确认。

```sh
H3_PYTHON=/path/to/h3/venv/bin/python
PACKAGE_ROOT=/path/to/recorded-h3-packages
BRIDGE_ROOT=/path/to/your/h3-experiment
BRIDGE_SCRIPT=/path/to/run_h3.py

A1=pilot--vehicle.paint_color--exterior--take-1--A1
A2=pilot--vehicle.paint_color--exterior--take-1--A2

"$H3_PYTHON" -m vehicle_edit_lab learn-direction \
  --package-root "$PACKAGE_ROOT" --attribute vehicle.paint_color \
  --pair donor-1-red donor-1-blue \
  --pair donor-2-red donor-2-blue \
  --pair donor-3-red donor-3-blue \
  --output outputs/paint-direction.safetensors

for job in "$A1" "$A2"; do
  "$H3_PYTHON" -m vehicle_edit_lab prepare \
    --plan outputs/paint-pilot.json --job "$job" \
    --package-root "$PACKAGE_ROOT" \
    --direction outputs/paint-direction.safetensors --alpha 1
done
```

供体需要 `conditioning`，可用仅编码的包。母版同时需要 `conditioning`、`latent`、实际 `noise` 和录制 seed。默认编辑 `t000000`，适用于既有实验中的 `[1, tokens, features]` 条件张量。方向由成对条件行的均值差聚合而来，可以通过重复 `--nuisance 文件` 投影掉已知干扰方向。示例强度 `1` 是探索起点。

A2 必须先准备同一配对样本的 A1，并使用相同方向文件、张量名和强度。两者在加到原张量前的扰动范数相同，转回原始 dtype 后的实际差值另行记录。编辑只写新包，原包和其他载荷的校验值保留。

## 3. 回注并收集镜头

在现有 H3 主机执行本步骤。适配器使用既有脚本的 `replay` 模式和 worker 地址；worker、实验环境及其 `evidence/` 目录需要已准备好。调用前检查条件包的哈希和配对关系，生成时只传包名及采样设置。

```sh
A0=pilot--vehicle.paint_color--exterior--take-1--A0
for job in "$A0" "$A1" "$A2"; do
  "$H3_PYTHON" -m vehicle_edit_lab replay \
    --plan outputs/paint-pilot.json --job "$job" \
    --package-root "$PACKAGE_ROOT" --worker-python "$H3_PYTHON" \
    --bridge-root "$BRIDGE_ROOT" --bridge-script "$BRIDGE_SCRIPT"
done
```

每个任务在 `results/runs/<job>/` 生成起始记录、桥接日志及完成时的视频和回执。遇到错误或超时，先检查桥接日志与 worker 队列；目录预留后同一任务不会再次自动提交。选择新的任务 ID 或在确认原任务状态后归档既有本地记录，再重试。

## 4. 合成连续两个镜头

用 `continuity` 计划取得同组的两个镜头。将两份完整的结果目录放入 `examples/hypit/local/`（如直接使用 MP4，把文件放在同一目录下即可）。安装并配置 Hypit 后，使用模板导出，分别给两个镜头提供素材：

```sh
python3 -m vehicle_edit_lab export-svrun \
  --author examples/hypit/ad.svml \
  --output examples/hypit/local/pair.svrun \
  --receipt shot-01.video=examples/hypit/local/shot-01/receipt.json \
  --receipt shot-02.video=examples/hypit/local/shot-02/receipt.json

cd examples/hypit
hypit plan local/pair.svrun
hypit build local/pair.svrun --follow
```

模板用于口播之外的纯画面对照。编译时请绑定两个镜头并确认计划仅含标准化和合成等下游任务；模板的原始镜头声明是 H3 生成节点。任意已有 MP4 用 `--clip 输出名=路径` 导入，历史 Build 产物用 `--reuse 输出名=BuildID:历史输出名` 复用。将外部执行的 A3 和 A4 结果作为 MP4 传入时，在评审记录里注明视觉条件或模型模式。
