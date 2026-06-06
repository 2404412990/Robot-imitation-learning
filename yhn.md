
# 4.16

姿势识别控制系统 UI 模块 (Pose Recognition Control System UI)
<img width="1002" height="497" alt="屏幕截图 2026-04-16 143036" src="https://github.com/user-attachments/assets/662335a4-59fc-45a2-b9f2-35a4f962bf47" />
已完成的工作

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



# 5.1

一、RTMPose3D方案复现与问题记录
按照组内选定的方向，复现了RTMPose3D → GMR实时映射管线。

遇到的问题：输出为17个稀疏关键点，缺少SMPL参数和运动学约束，直接映射到G1（29 DOF）时动作明显失真。

解决与结论：记录并反馈了该方案的局限性（缺少骨骼旋转、全局根节点），协助组内确认后续放弃该路线，转向WHAM。

二、Windows环境迁移中的踩坑与修复
跟随组内的WHAM+GMR技术路线，在Windows环境下进行适配。

遇到的主要困难：

PyTorch3D和DPVO在Windows上编译失败

setup.py因GBK编码问题报错

NumPy 2.x与PyTorch 1.11.0不兼容

清华conda镜像不稳定导致依赖安装失败

解决方式：

逐一排查编译错误，调整环境变量与依赖版本

切换至官方源+代理，编写setup_env.ps1脚本自动化安装

将NumPy降级至1.x，拆分requirements文件避免环境冲突

# 5.18
三、实时联调中的卡顿与抽搐问题排查
参与实时CSV管道的联调测试。

遇到的问题：

Unity端重定向时机器人出现明显抽搐和卡顿

视角乱飞，实时效果远差于离线回放

排查过程：

对比离线CSV（效果正常）与实时CSV（抽搐）的表现差异

定位根因为CSV写入与读取的竞态条件：Unity读取速度 > Python生成速度，导致重复读帧或跳帧


解决方式：

调整WHAM的detect_interval/infer_interval参数

降低Unity端读取频率，增加帧间插值平滑

四、机器人切换时的异常问题跟进
跟随组内多机器人切换框架的集成工作。

遇到的问题：

切换机器人后，原机器人躯体出现异常（关节状态损坏）
# 5.30

G1小臂出现翻转，映射顺序错误

解决方式：

协助验证切换时需先停止当前管道，通过Renderer可见性切换避免状态损坏

发现关节映射顺序与URDF不一致的问题，配合修正为1:1恒等映射
