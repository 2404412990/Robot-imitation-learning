# 2026/6/26

这周继续改 imitation 的前端入口和机器人切换逻辑，同时完成了第二部分技术路线汇报 PPT。

1. 我把机器人选择入口重新改了一版，把机器人显示名、内部 key、CSV 路径刷新从分散的写法里抽出来，统一到同一套映射关系里。现在机器人切换时不会再同时手改多处字符串和路径。

2. 我把 `SupportedRobotNames`、`RobotAliases` 和 `CsvExpectedColumnsByRobot` 这一套机器人配置继续补齐，前端已经能按 key 区分 `unitree_g1`、`unitree_h1`、`openloong`、`x02lite` 这几类对象，后面接新机器人时只需要补映射而不用重写整套 UI。

3. 我继续处理了两个还没完全收敛的 bug：
- Stop 按钮停止链路还需要继续压稳
- 下拉列表的点击命中和弹出层级还要继续修

4. 这周制作了第二部分技术路线汇报 PPT，重点写了：
- WHAM 到 GMR 的整体链路
- Unity 侧实时 CSV 重放的实现方式
- 前端切换机器人、状态显示、停止链路这几块我自己改过的内容


<img width="1081" height="601" alt="image" src="https://github.com/user-attachments/assets/fc823143-3234-4962-90db-e8ce8c2bb05b" />


<img width="1084" height="600" alt="image" src="https://github.com/user-attachments/assets/3c30c40e-023f-4525-a8a3-e116f6bcdaa2" />


# 2026/6/19

这周继续改机器人切换和运行时对象控制，主要是把“显示哪个机器人”和“谁在接收动作”这两件事彻底分开。

1. 我把机器人当前状态拆成了三层：
- 是否显示
- 是否激活
- 当前是否接收动作

这样改完以后，前端切换机器人时不会再把显示状态和动作接收状态混在一起。

2. 我把机器人切换相关的动作往中间控制层收了一步。原来很多按钮是直接去碰底层对象，现在 `Start / Stop / 切换机器人 / 刷新 CSV` 都统一从固定入口走，前端脚本里散着写的直接调用少了很多。

3. 我把 `CurrentSelectedRobotKey`、`CurrentSelectedRobotRoot`、`ResolveSceneObjectByRobotKey`、`ApplyRobotVisibility` 这一套切换逻辑串起来了。现在选中机器人后，前端能稳定找到对应场景对象，并切换可见性。

4. 我还把只对当前选中机器人添加运行时碰撞体这件事接进去了，避免所有机器人一起开碰撞导致场景更乱。相关逻辑已经放到 `EnsureRuntimeCollidersForSelectedRobot` 和 `RuntimeRobotColliderProxy` 这条线上。


# 2026/6/12

这周主要改 Stop 链路和模式切换后的状态恢复，解决以前“一个按钮把所有东西一起关掉”的问题。

1. 我把停止实时推理的流程拆开了，不再是一键全杀，而是按下面的顺序走：
- 先停止前端继续发送控制事件
- 再停掉 CSV 监控
- 再关实时窗口
- 再结束 bash / python 进程和子进程
- 最后清理显存/缓存和界面状态

2. 代码里把这条链路拆成了 `OnStopButtonClicked`、`StopBashProcessAsync`、`StopBashProcessCore`、`StopCsvMonitor` 这些入口。这样出问题时能明确判断是按钮层、CSV 监控层，还是进程层出了问题。

3. 我把 Stop 相关的后台停止做成了异步方式，避免 Unity 主线程卡死。打包日志里已经能看到 `Background stop started`、`taskkill /T /F /PID ... executed`、`Background stop completed` 这一整条链路。

4. 我还把模式切换后的状态恢复重新改了一版，把按钮可交互状态、日志区刷新和当前模式显示拆开处理，减少切换后界面还停留在旧状态的问题。

# 2026/6/5

这周主要改了实时 CSV 重放逻辑，不再只是盯着文件有没有生成，而是开始按“它是不是还在持续更新”来处理。

1. 我把机器人切换和 CSV 列表刷新改成了联动式处理。切换机器人时先清掉旧 CSV 选择，再重新加载当前机器人目录下可用的文件，避免不同机器人共用错误动作文件。

2. 我把数据集路径按机器人目录重新接到了前端，当前运行包里已经能看到：
- `ImitationDataset/unitree_g1`
- `ImitationDataset/unitree_h1`
- `ImitationDataset/openloong`
- `ImitationDataset/x02lite`

前端 CSV 下拉框就是按这个结构刷新的。

3. 我加了实时 CSV worker，这条线已经包含：
- `csvWorkerThread`
- `csvReadOffset`
- `csvPollInterval`
- `csvMonitorStartUtc`

也就是不再每次整文件重读，而是按偏移量持续追踪 `live_motion.csv` 新追加的内容。

4. 我把实时数据状态判断也接进去了，直接看两个量：
- `live_motion.csv` 的最后修改时间
- 当前帧号是否持续递增

如果这两个量长时间不变，前端就显示“暂无新数据”，而不是继续把旧帧当成实时结果。

5. 我还加了实时帧率估计和缓冲参数，日志里已经能看到 `estimateFps=True`、`fpsWindow=2.00s`、`fallbackFps=5.00` 这些配置。这样 Unity 侧实时重放不会死盯固定帧率，而是按 CSV 实际追加速度去跟。


# 2026/5/29

这周主要改状态显示、日志输出和按机器人筛选 CSV 的逻辑，同时把整条技术路径在代码层真正串清楚了。

1. 我把控制面板的状态显示补了一版，现在前端已经会显示：
- 当前运行模式
- 当前机器人
- 当前 CSV 路径
- Start / Stop 状态

2. 我把前端、WHAM、GMR、Unity 重放这四部分真正按运行顺序串了一遍：前端负责输入和状态，WHAM 负责人体运动，GMR 输出机器人动作，Unity 再消费 `live_motion.csv` 做展示和控制。这一条现在已经不是口头路线，而是跑在 `StartInput -> handle_wham_gmr.py -> live_motion.csv -> IMimicAgent` 这条线上。
   
3.整理技术路径，相关工作，绘制流程图。 
<img width="843" height="480" alt="image" src="https://github.com/user-attachments/assets/c8b68fad-75be-4129-8b81-e02a493fc600" />


# 2026/5/22

这周主要改了 `StartInput` 的控制层和 Stop 安全退出逻辑，核心是把原来混在一起的流程拆开。

1. 我把启动流程拆成了几层：
- UI 层：按钮和状态显示
- Controller 层：事件分发
- ProcessManager 层：bash/python 进程启动和停止
- Reader 层：CSV 或 TCP 数据消费

这样改完以后，按钮、进程、文件和显示状态不再全堆在一个脚本里。

2. 我把启动 bash/python 流程独立成了 `StartOrRestartBashProcess`，停止流程独立成了 `StopBashProcessAsync / StopBashProcessCore`，同时加了 `SafeProcessId` 和 Windows 侧的 `KillProcessTreeWindowsQueued`。这样 Stop 不再只是“关一个父进程”，而是按进程树处理。

3. 我加了一层轻量的运行时状态记录，至少会记当前机器人、CSV 路径、TCP 开关和输出目录。`unity_startinput.log` 里现在已经能直接打印：
- `ROBOT=...`
- `CSV=...`
- `OUTPUT_ROOT=...`
- `TCP=...`



# 2026/5/15

这周主要处理两个最明显的问题：Stop 会把电脑直接重启，下拉列表经常点不中。

1. 我先把这两个问题分开处理了，不再混在一起查。Stop 问题归到进程管理链路，下拉框问题归到 UI 层级和射线检测。

2. Stop 这块我加了单独的 debug log 和停止入口，先把按钮事件、进程状态、CSV 状态分别打出来，避免以前一按 Stop 只看到程序没了，不知道具体卡在哪一步。

3. 下拉框这块我把 `TMP_Dropdown`、`EventSystem`、`GraphicRaycaster` 和弹出模板层级重新过了一遍，问题基本定位到点击层级和遮挡，而不是数据源本身。

4. 这样改完以后，前端交互问题和后端进程问题已经能分开查，不会再出现一个问题把两边都搅在一起的情况。


# 2026/5/8

这周主要把 imitation 场景缺的前端依赖补齐，并把几个关键状态真正接到界面上。

1. 我把 `TextMeshPro` 和 `Runtime Inspector & Hierarchy` 补进去了。前者解决按钮、状态文本、下拉列表的显示问题，后者用来在运行时直接看对象层级和脚本字段。

2. 我把控制面板重新收了一下，把机器人选择、CSV 路径、视频导入和实时控制尽量放到同一块区域，减少联调时到处找对象和按钮。

3. 我把几个最关键的状态先接到了界面和运行时调试里：
- 当前机器人
- 当前 CSV
- 当前按钮事件
- 当前是否正在推理

4. 这样改完以后，前端已经不只是一个按钮面板，而是能在运行时直接看出当前到底卡在选择、启动、读 CSV 还是停止。


# 2026/5/1

这周主要把格物平台里 imitation 这套入口统一了一次，让它从“几个分散按钮”变成真正可用的控制入口。

1. 我把前端入口先统一成了几个核心功能：
- 机器人型号选择
- CSV 文件选择
- 视频导入与回放
- Start / Stop 实时识别
- 运行状态提示

2. 我把 `StartInput` 这条主入口拉起来以后，前端已经能区分三种不同状态：
- 离线 CSV 回放
- 实时推理启动
- 推理停止和资源回收

3. 我重新过了一遍 Windows 环境下的平台运行方式，确认前端控制和后端推理进程不能继续强耦合。后面所有 Start / Stop 都改成从统一入口走，而不是按钮直接绑系统命令。



# 2026/4/17
在windows服务器administrator@10.60.244.142里面部署了gewu平台。进行了初步的ui设计。
<img width="1286" height="1470" alt="1921777525638_ pic" src="https://github.com/user-attachments/assets/a686e0de-c39b-46f8-b5f9-6c841e218a31" />



# 2026/4/10
在4060 Laptop上对最新版本的retarget进行测试
成果：基本验证实时性。

# 2026/4/3
对GVHMR进行实时化改造

1、VideoCapture(0)	支持摄像头输入

  --skip_frames	跳帧处理，减轻推理压力
  
  window_size 8	从 10 降到 8，减少初始缓冲延迟
  
  --no_render	可选跳过 mesh 渲染，延迟减半
  
  cv2.imshow	实时显示窗口，按 q 退出
  
  --video 测试模式	无显示器时用视频文件验证流程
  
  ViTPose-B 权重（vitpose-b-multi-coco.pth）尚未获取，换上后 ViTPose 推理速度预计提升 3-4x

2、在服务器上用 kunkundance.mp4 跑通全流程，248帧无报错，稳定 ~2.5fps

# 2026/3/26
苏钰林跑通从视频到mujoco中h1动作重定向的流程

做了一个basketballFOX的测试样例

Current Limitation：

1、动作映射仍然存在卡顿现象，这与一开始GVHMR中视频映射到3dpose时数据不完整有关。

2、暂时的物理模型效果较差。

Future work：对视频中的动作状态进行补全。
