# 视频广告研究

从一条 15–60 秒汽车广告母版出发，批量生成指定变体，保住原剧情、机位、非目标部件和跨镜头连续性，并压低每条**合格成片**的成本。

本仓库是研究过程的归档，包含大量负结果和未达标结论。数字口径以 [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) 为准，各工程的细节和证据在对应目录的 README 里。

## 结论速览

更新于 2026-11-26。

| 能力 | 状态 |
| --- | --- |
| H3 条件包跨进程精确重放 | ✅ 36/36 组解码音画一致（视频 RGB、音频 PCM 分别比对 SHA-256） |
| 按镜头替换 + Hypit 时间线合成 | ✅ 30 秒接缝样片、36 秒整片对照均已完成 |
| 车漆定向换色 | ⚠️ 目标命中 9/9，严格通过 0–1/9 |
| 粗粒度换车型 | ⚠️ 目标命中 9/9，严格通过 0/9，原车色或机位会漂移 |
| 轮毂、后侧窗等细部件 | ❌ 目标命中 2–3/9，未通过 |
| 指定车型 / 车标 / 整片批量交付 | ❌ 未通过 |
| 单条合格成片成本 | ⏳ 未测，缺合格样本作分母 |

**验收线**：9 条留出样本中至少 8 条严格通过。严格通过 = 目标命中 + 时间稳定 + 车身、场景、相对机位和受保护部件全部保住。只算目标属性叫「命中」，两条口径经常同时成立（9/9 命中、0/9 严格通过），不要混读。

## 核心链路

```
母版生成 ── 录制条件包 / 图像 latent / 实际音视频初始噪声 / 生成配置
   │
   ▼
变体生成（独立进程，只读条件包与噪声；工作流无原 prompt、无文本编码节点）
   │
   ▼
按镜头替换 ── 只重跑受影响的镜头
   │
   ▼
Hypit 合成 ── 同一时间线绑定新旧镜头，保留各段音轨
```

关键点是**回放录制下来的噪声张量**，而不是靠 seed 在另一个环境重建——这是 36/36 精确重放成立的前提。

36 秒对照的实测收益：复用 6 秒无车插镜后，H3 调用 **3 → 2** 次，串行耗时 **147.7 s → 124.0 s**，两路成片解码音画完全相同。该片未过质量门槛，所以这组数字只反映一次工程运行成本，不是合格成片成本。

## 工程地图

| 工程 | 作用 | 关键结论 |
| --- | --- | --- |
| [batch-pipeline](batch-pipeline/README.md) | ModelArts 独立进程回注，条件压缩、单条件行干预、最终 latent 复用负对照 | 36/36 组音画一致；FP16 压到约 50%，INT8 约 28–29% |
| [lienyu_h3_nodes](lienyu_h3_nodes/README.md) | ComfyUI 节点：条件包、初始噪声、最终 latent 的导出与回注 | 上述重放能力的底层工具，6 个节点 |
| [remote-audit](remote-audit/README.md) | 远端节点源码审计快照，含独有的 DiT 跳层节点 | 5 组层消融中 4 组严重坍缩，跳层不可安全复用 |
| [h3-auto-ad-vector-edit](h3-auto-ad-vector-edit/README.md) | 条件向量方向编辑、30/60 秒分镜、单行与层组归因、内容缓存 | 红转蓝与跑车转 SUV 命中 9/9；找不到可独立编辑的条件槽 |
| [h3-ad-template-extraction](h3-ad-template-extraction/README.md) | 从有溯源母版或纯 MP4 提取可编辑模板 | 纯 MP4 切镜召回 0/3，车辆蒙版不完整，人物未定位 |
| [h3-hypit-vehicle-edit-lab](h3-hypit-vehicle-edit-lab/README.md) | 四类车辆属性对照 + 首帧同步接缝 + Hypit 合成 | 162/162 计划任务完成；四类属性严格通过均为 0/9 |
| [lienyu-h3-localized-edit](lienyu-h3-localized-edit/README.md) | 逐帧车辆区域约束、每步回注原镜头潜变量、像素级色相迁移 | 稳定命中提升到 6–9/9，严格通过仍 0/9；几何与机位是瓶颈 |
| [lienyu-h3-ad-workbench](lienyu-h3-ad-workbench/README.md) | 部件约束换色工作台、36 秒复用插镜成本对照 | 部件约束把稳定命中从 1/9 提到 6/9，未过线；成本对照成立 |
| [hypit-upstream-main](hypit-upstream-main/README.md) | Hypit 0.3.3 上游源码快照 | 提供语义时间线、运行绑定和媒体合成边界 |

## 目录结构

```
video-ad-research/
├── PROJECT_PROGRESS.md          研究总览，数字口径来源
├── batch-pipeline/              条件包重放、压缩、编辑与 latent 复用
├── lienyu_h3_nodes/             ComfyUI 条件/噪声/latent 节点
├── remote-audit/                远端节点审计快照（DiT 跳层）
├── h3-auto-ad-vector-edit/      条件向量编辑、分镜、归因、缓存
├── h3-ad-template-extraction/   母版与 MP4 的模板提取
├── h3-hypit-vehicle-edit-lab/   H3 × Hypit 车辆属性与接缝实验
├── lienyu-h3-localized-edit/    区域约束与原镜头潜变量回注
├── lienyu-h3-ad-workbench/      部件换色工作台与成本对照
└── hypit-upstream-main/         Hypit 上游快照（见下方合规说明）
```

## 快速开始

### 唯一能开箱即跑的部分

纯 MP4 像素基线，只依赖 OpenCV 与 NumPy：

```bash
python3 -m pip install opencv-python numpy
python3 h3-ad-template-extraction/extract_pixel_template.py source.mp4 --output out/
```

输出镜头时码、红车包围框和抽帧。

### 其余部分需要外部环境

这些脚本是为当时的 ModelArts + ComfyUI + Hypit 部署写的，**不能开箱复现**：

| 依赖 | 用途 |
| --- | --- |
| ModelArts H3 worker（含 `lienyu_h3_nodes`） | 母版生成、条件包录制、变体重放 |
| 录制好的条件包与对应模型配置 | 所有对照实验的输入 |
| Hypit 0.3.3 运行时 + ffmpeg | 多镜头时间线合成与成片校验 |

脚本里保留的是当时的远端目录结构，复跑前需要按各工程 README 调整路径。每条实验的通过与否都记录在对应目录的 `results/`、`*.json` 评估文件里。

## 仓库未包含的内容

`results/`、`r2/`、模型权重、抽帧与成片音视频没有入库，原因是体积超过 GitHub 单文件 100 MB 限制，且多数是可再生的实验产物。需要复核时以本地归档为准。

## 合规说明

`hypit-upstream-main/` 是 Hypit 上游源码快照，遵循其自带的 [LICENSE](hypit-upstream-main/LICENSE)：修改版 Apache 2.0，附带商业使用条件。**该目录不得对外再分发**，本仓库也不因此获得任何再授权。其余本工程内容为内部研究资料。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [PROJECT_PROGRESS.md](PROJECT_PROGRESS.md) | 总览、工程地图、已验证链路、待验证清单 |
| [h3-ad-template-extraction/TEMPLATE_SCHEMA.md](h3-ad-template-extraction/TEMPLATE_SCHEMA.md) | 可编辑模板的数据结构 |
| [h3-hypit-vehicle-edit-lab/docs/experiment-protocol.md](h3-hypit-vehicle-edit-lab/docs/experiment-protocol.md) | 车辆实验协议与组别定义 |
| [h3-hypit-vehicle-edit-lab/docs/hypit-integration.md](h3-hypit-vehicle-edit-lab/docs/hypit-integration.md) | Hypit 接入设计 |
| [batch-pipeline/reports/H3_AUTO_AD_RESEARCH_REPORT.md](batch-pipeline/reports/H3_AUTO_AD_RESEARCH_REPORT.md) | 汽车广告研究报告 |

## 尚待验证

1. 换色过线：区分车漆与车窗边缘、尾灯、暗部、反光，复验 8/9 严格通过。
2. 同机位同路线换指定车型：显式约束车头朝向、轮胎落点、遮挡与相机轨迹。
3. 模板驱动批量出片：绑定车辆身份、部件、镜头时段与首尾帧依赖。
4. 合格成片成本：同一批任务比较整片重生成与按镜头替换的质量、调用数、耗时与返工。
