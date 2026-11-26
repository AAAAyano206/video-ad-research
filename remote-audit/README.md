# H3 远端节点审计快照

## 目的与方案

保留当时远端 H3 实验使用的自定义节点源码，便于复核 DiT 分段消融所调用的实际实现。快照位于 [lienyu_h3_nodes/__init__.py](lienyu_h3_nodes/__init__.py)。对比 [本地基础节点](../lienyu_h3_nodes/__init__.py)，快照多出 `SkipH3Blocks` 及其注册项，接受 `0 ≤ start ≤ end ≤ 50` 的半开区间，并对克隆模型的对应层注册 identity 替换。

## 流程与结果

向量实验在原包重放请求上逐组替换 10 个 DiT 块，其他主要条件固定，对比输出画面。五组中四组发生严重坍缩，10–20 层组仍能辨认为运动汽车广告，但车型细节和构图明显改变。 [层消融评估](../h3-auto-ad-vector-edit/fine-attribution-evaluation.json)

## 工程意义

该快照将报告里“跳层”的描述对应到具体代码，支撑「当前测试无法通过跳过一组层安全复用计算」的结论。它是带差异的审计资产。日常条件包工作流的入口见 [基础节点说明](../lienyu_h3_nodes/README.md)。
