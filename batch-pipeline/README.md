# MiniMax H3 嵌入场批量验证

## 目的

验证是否能在编码进程中将原始 prompt 转成完整的 H3 条件包，再由独立生成进程读取并生成等价音画。进一步测量条件压缩、局部颜色编辑和上一次生成结果复用的效果。

## 方案与流程

1. 用 [阶段 2 脚本](scripts/run_phase2.py) 提交正常 ComfyUI 工作流。通过 [自定义节点](../lienyu_h3_nodes/README.md) 保存条件张量、元数据、初始 latent 结构和实际音视频初始噪声。
2. 替换编码和噪声节点，提交仅从文件加载的重放工作流。使用相同 H3 权重、采样参数和已录制实际噪声。重放路径提交前检查文本占位符已经移除。
3. [阶段 3 批处理](scripts/run_phase3_batch.py) 使用 [12 条输入](configs/phase3_prompts.json) 和 101、202、303 三个 seed，分别生成原版和重放版。解码视频为 RGB、音频为 PCM，再分别比较 SHA-256。
4. 阶段 4 分为 [条件压缩](scripts/run_phase4_compression.py)、[单条件行干预](scripts/run_phase4_attribute.py) 与 [等预算最终 latent 复用](scripts/run_phase4_drift.py)。各组保留抽帧对照。

## 结果

| 验证 | 结果与边界 |
| --- | --- |
| 条件传输与原意 | 36/36 组解码音画一致。宽口径抽帧观察 36/36 保留核心视觉意图。独立进程位于同一台 ModelArts 机器。 |
| 条件压缩 | FP16 约为原条件载荷的 50%，INT8 约 28–29%。每种方式各 5 例保留核心语义，像素有变化，INT8 变化更明显。比例仅计条件载荷。 |
| 属性干预 | 找到成对条件中 L2 差异最大的行并换入。在纸船、外套、墨水三例中改变了目标颜色并基本保留主要内容。 |
| 有益细节复用 | 使用相同噪声和 4 步预算，在 `denoise=1.0` 注入前次最终音视频 latent 后，解码输出与基线完全一致。该注入方式未产生收益。 |

证据见 [阶段 3 抽帧](contact-sheets/phase3-contact-sheets)、[压缩抽帧](contact-sheets/phase4-compression-sheets)、[属性对照](contact-sheets/phase4-attribute-sheets)、[latent 复用对照](contact-sheets/phase4-drift-sheet.jpg) 和 [原验证会话](codex://threads/01a1f803-0f5f-74b0-95fa-6a54d54e140c)。

## 工程意义与复现

这是后续汽车广告实验的可重放输入基础。生成器仍需处理条件序列和自身的联合注意力。复现需要匹配的 ModelArts H3/ComfyUI 桥接环境和原始条件包，脚本仍引用当时部署的远端目录。随目录归档的是脚本、配置和部分可视化证据。 [后续汽车广告研究报告](reports/H3_AUTO_AD_RESEARCH_REPORT.md)
