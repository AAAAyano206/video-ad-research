# Hypit 接入设计

## 模板记录什么

模板把改动映射到明确的车和镜头。母版工程有生成溯源时，优先使用原分镜计划、参考素材和条件包清单。只有平铺 MP4 时，先通过视觉分析和对象跟踪建立镜头及对象记录，低置信区域可补标。轮毂和后侧窗分别绑定到同一车的子区域；遇到多个车时，每辆车有各自的 ID。

| 字段 | 含义 |
| --- | --- |
| `entity_id`, `attribute_id` | 哪辆车及哪个属性，如 `hero-car` / `vehicle.wheel_design` |
| `shot_ids`, `visible_ranges`, `roi_ref` | 出现在哪些镜头、哪些时间区间、如何找到局部评估区域 |
| `base_package_ref`, `base_output_ref` | 原条件包及成片或镜头素材的稳定引用 |
| `visual_dependencies` | 例如 `shot-02.first_frame ← shot-01.last_frame`，由结果所在的变体版本解析 |
| `edit_strategy`, `direction_ref` | 可执行的编辑方法和方向版本。方向相关的层、行及强度有证据时可记录 |
| `protected_attributes` | 应保留的元素与允许变化的边界，来自实验协议 |

## 第一期通过文件对接

1. 工作台读取模板和替换请求，例如 `hero-car / vehicle.wheel_design / mesh_multispoke`，从影响关系生成渲染计划。
2. H3 执行端按实验组加载条件包、修改张量或应用参考媒体，生成镜头 MP4。只把结果交给合成侧。需要 A3 时，先生成前一镜头，再从它提取新末帧并准备下一镜头的视觉条件。必须通过一次回注检查，确认该工作流可以按预期组合文本和视觉输入。
3. 每个成片版本生成独立的 `.svrun`。对已生成的镜头，用 `<file>` 和 `<satisfy>` 绑定本地 MP4。其余仍适用的历史产物用 `<build-record>` 显式复用。替换 MP4 后，依赖内容变化的标准化、语音对齐或跟踪证据需要重新计算。
4. 用 `hypit plan` 核对各组共用的时间线、应该复用的素材和需要执行的节点，再构建。保存展开的计划、镜头哈希、依赖输入版本和成片链接。

这个例子假设 Author Source 已声明名为 `shot-02.video` 的 Blob 输出：

```xml
<?svml using="@hypit/markup/run@1"?>
<svrun version="1">
  <author source="./ad.svml"/>
  <target output="final.video"/>
  <file id="shot-02-variant" type="@hypit/blob@1#Blob"
    from="./outputs/shot-02.mp4" media-type="video/mp4"/>
  <satisfy output="shot-02.video" candidate="shot-02-variant"/>
</svrun>
```

## 第二期接入 H3 条件包能力

文件方式跑通后，可以增加项目自己的 Hypit Model / Provider 接口，提交条件包引用、方向版本、视觉条件依赖及采样配置，收集生成镜头。H3 的原生 Hypit 接口已经覆盖文字、参考媒体与首尾帧；条件张量回注由现有 H3 实验执行端提供。

公开仓库只保存研究协议、模板、适合公开的汇总和授权素材。机器路径、账户配置、权重、条件张量和原始广告素材保存在本地。若产品后续要直接集成 Hypit 代码，对外托管和再分发应按 [上游许可证](https://github.com/hypit-ai/hypit/blob/main/LICENSE) 处理。
