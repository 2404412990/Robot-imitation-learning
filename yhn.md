
# 4.16

姿势识别控制系统 UI 模块 (Pose Recognition Control System UI)
<img width="1002" height="497" alt="屏幕截图 2026-04-16 143036" src="https://github.com/user-attachments/assets/662335a4-59fc-45a2-b9f2-35a4f962bf47" />
已完成的工作
模块	内容	状态
UI界面设计	Canvas + 2个下拉列表 + 4个按钮 + 状态文本	 完成
脚本编写	UIManager、PoseRecognitionController、PoseRecognitionEngine、VideoPlayerManager	 完成
事件绑定	UI组件与脚本的事件关联	 完成
功能接口	机器人选择、CSV路径选择、导入视频、重放、开始/停止识别	 完成





一、项目简介
GVHMR + GMR 是一个三维人体姿态估计与运动重建项目。

二、完成工作

2.1 环境配置
 成功配置项目运行环境

2.2 模型推理
 使用 subject3 完成一次完整的前向推理

2.3 结果观察
获取输出结果并进行了初步观察

<img src=docs/video/s3.png alt="animated" />

三、不足
快速运动响应不足
表现：对快速动作（如跳跃、转身）的跟踪出现滞后或抖动

改进方向：优化时序平滑算法；提高帧率或引入运动先验

四、未来工作
模型微调
在特定场景数据（如舞蹈动作、体育动作）上对模型进行微调，提升在目标领域的表现
