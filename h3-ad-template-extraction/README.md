# H3 汽车广告模板提取

## 目的

验证工作台能否从已有广告找到镜头边界、主角及相关属性，再把编辑请求对应到可重放的 H3 条件或参考媒体。分别验证有生成记录的母版和只有 MP4 的盲输入。

## 方案与流程

1. **有生成记录**。从同一个母版条件包派生 4 个条件行区段消融和 4 个视频噪声时间段清零变体，重放后观察属性、机位及整段运动的变化。入口见 [探针构造](make_probes.py) 和 [执行脚本](run_probes.sh)。
2. **纯 MP4 像素基线**。[extract_pixel_template.py](extract_pixel_template.py) 逐帧比较 HSV 直方图，用 HSV 红色阈值、形态学和最大轮廓找红车区域，输出时码、框和抽帧。需要 Python、OpenCV 与 NumPy，可用 `python3 extract_pixel_template.py source.mp4 --output example-output` 处理本地视频。
3. **纯 MP4 参考生成**。[Ref2VA 探针](run_reference_inversion_probe.py) 仅给源视频和固定通用指令，保存产生的条件及实际噪声，再在匹配的 Ref2VA 模式下独立重放。

## 结果

| 路线 | 实测 |
| --- | --- |
| 因果探针 | 条件前几个区段分别偏向外观、镜头距离和车/司机呈现。清零局部视频噪声会影响整段构图和运动。探针适合建立影响地图，未找到可独立切分的语义或时间槽。 |
| 盲切镜与跟踪 | 60 秒、4 个原镜头的样本中只检测到 2 个镜头，真实边界召回 0/3、误报 1。红色目标覆盖 1439/1440 帧，但蒙版常漏掉整车结构，人物未定位。 |
| Ref2VA 参考条件 | 保留红色跑车、海岸公路和男性司机等元素。保存后的参考条件生成可精确重放；源广告的逐帧内容和原始时间线未恢复。 |

查看 [探针对照](probe-sheet.jpg)、[盲提取模板](blind-story-base/template.json)、[模板提取评估](template-extraction-evaluation.json) 和 [参考条件审计](reference-inversion-evaluation.json)。

## 工程意义

定义了两个接入边界。有溯源素材可以继承母版条件和镜头记录，并附上因果影响。外部成片需要专门的切镜、检测、分割、身份跟踪及低置信结果确认。 [可编辑模板协议](TEMPLATE_SCHEMA.md)
