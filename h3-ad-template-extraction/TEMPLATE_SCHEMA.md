# H3广告可编辑模板协议（探索版）

模板由 `source_hash`、模型/工作流版本、镜头时码、实体轨迹、可编辑属性和H3因果探针组成。每个可编辑项至少记录：`entity_id`、`attribute`、`time_range`、`spatial_region`、`condition_rows`、`latent_time_range`、推荐方向向量、有效强度范围、非目标副作用、置信度和验证规则。

对有H3生成溯源的广告，通过条件行消融和视频噪声时间窗消融重放，测量每个探针对各时间段和空间区域的影响，形成 `condition_rows/latent_time_range → visual effect` 映射。对没有H3条件包的外部广告，目前不能从成片唯一反演出原条件场；需要先做镜头切分、对象跟踪和视觉条件反演，H3本身不是完整的视频理解/分割器，因此低置信映射必须交给人工确认。
