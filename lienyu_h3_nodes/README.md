# H3 条件包与噪声的 ComfyUI 节点

## 目的

将 H3/ComfyUI 工作流中的条件与实际初始噪声保存成可传输文件，供另一生成进程重放，并保留最终音视频 latent 的对照入口。

## 方案与流程

| 节点 | 职责 |
| --- | --- |
| `LienyuH3ExportConditionPackage` / `LienyuH3LoadConditionPackage` | 将条件和初始 latent 的嵌套对象拆成张量及结构描述，再按原结构加载 |
| `LienyuH3RecordInitialNoise` / `LienyuH3ReplayInitialNoise` | 保存采样器实际收到的音视频初始噪声，并在重放时返回该张量 |
| `LienyuH3SaveGeneratedLatent` / `LienyuH3LoadGeneratedLatent` | 保存和读取最终音视频 latent，支撑等预算的生成状态复用对照 |

节点实现在 [__init__.py](__init__.py)。包目录由 `LIENYU_H3_PACKAGE_ROOT` 配置，载荷使用 safetensors 和 JSON 结构树，manifest 记录张量特征和校验信息。正常路径先导出条件及 latent、再记录采样噪声。独立工作流替换对应节点并保持 H3 模型、模式、采样和解码设置一致。

## 结果与工程意义

该节点链支撑 [批量验证](../batch-pipeline/README.md) 中 **36/36** 组解码音画精确重放，并成为车辆实验固定原包与随机对照的底层工具。保存真实噪声让对照不依赖仅凭 seed 在另一环境重建噪声。最终 latent 保存节点还支撑了一个可复查的负结果：全去噪条件下，该复用方式未改变输出。

部署需要匹配的 H3/ComfyUI 环境、PyTorch 与 safetensors。跨模型模式、版本或硬件的等价性需另行验证。DiT 层消融实现见 [远端审计快照](../remote-audit/README.md)。
