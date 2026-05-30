# Unity Humanoid Imitation Framework

基于 Unity ML-Agents 的人形机器人模仿学习与重定向（retargeting）系统。Unity 端负责物理仿真、动作回放、PPO 训练；Python 端（`Robot-imitation-learning/`，不在本仓库范围内）跑 WHAM（视频 → 3D 人体动作）+ GMR（人体动作 → 机器人关节 IK），两端通过 CSV 文件做实时桥接。

支持多种 Unitree 系列人形机器人：**G1（29 DOF）**、**H1（19 DOF）**，以及只做可视化的 X02Lite、openloong。运行时下拉框切换目标机器人。

---

## 目录结构

```
Imitation/
├── G1.unity               # 主场景：Unitree G1，含 UI / 实时 retargeting / CSV 回放
├── G1Replay.unity         # G1 纯回放场景
├── G1mimicAgent.cs        # G1（29 DOF）的 ML-Agents Agent，PPO 训练 + 回放 + 实时
├── G1mimic1Agent.cs       # G1 早期变体（不同 prefab 配置，保留作参考）
├── H1mimicAgent.cs        # H1（19 DOF）的 ML-Agents Agent
├── config.yaml            # ML-Agents PPO 训练超参
├── g1m1.onnx              # G1 训练好的策略（29 DOF 输出）
├── h1-all-in-one.onnx     # H1 训练好的策略（19 DOF 输出）
├── dataset/               # G1 格式的 retargeting CSV 数据集（36 列 / 行）
├── Scripts/               # UI、桥接、Registry 等工具脚本
│   ├── IMimicAgent.cs            # 多机器人统一接口
│   ├── MimicAgentRegistry.cs     # 场景级 Agent 注册表（按 RobotKey 路由）
│   ├── StartInput.cs             # WHAM+GMR 子进程启动、CSV 监听、机器人切换
│   ├── Replay.cs                 # Replay 按钮：选 CSV 触发 Agent 回放
│   ├── Stop.cs                   # Stop 按钮：停管线 + 切回回放模式
│   ├── FileBrowser.cs            # 文件/文件夹选择下拉
│   ├── Move.cs                   # FPS 风格相机控制
│   ├── FPSUiInteractor.cs        # FPS 风格 UI 射线点击
│   └── ...
└── README.md              # 本文件
```

`Robot-imitation-learning/`（WHAM+GMR Python pipeline）不包含在本仓库里，需要按上游仓库的指引单独 clone 到这个目录下，否则实时 retargeting 路径无法启动。

---

## 快速开始

### 前置条件

- Unity 2022.3+（项目使用 ArticulationBody，需要较新的 PhysX）
- ML-Agents Package 2.0+
- TextMeshPro Package
- Windows 10/11（脚本里的 PowerShell 子进程启动只在 Windows 上验证过；Linux 走 bash 路径理论可行但本仓库默认配置是 Windows）

### 仅运行回放（最简路径）

1. Unity 打开 `G1.unity` 场景，按 Play
2. 顶部下拉框选 robot（G1 / H1 / 其他）
3. CSV 下拉选 dataset 里的一条动作（如 `dance1_subject1`）
4. 点 `Replay` 按钮，机器人按 CSV 重演该动作

### 运行实时 retargeting（视频 → 机器人）

1. 在本目录下放好 `Robot-imitation-learning/`（WHAM+GMR pipeline）
2. Inspector 里选中 StartInput 物体，配好以下字段：
   - **Bash Working Directory**：指向 `Robot-imitation-learning/` 的绝对路径
   - **Video Path**：要 retargeting 的视频文件（相对 working dir）
   - **Default Robot Name**：`unitree_g1` 或 `unitree_h1`
3. 按 Play → 点 `Start` 按钮，Unity 会用 PowerShell 拉起 `run.ps1` 子进程
4. Python 端跑 WHAM → GMR 实时写 `output/<run>/csv/live_motion.csv`
5. Unity 端 `StartInput` 协程轮询该 CSV，逐行喂给当前选中的 Agent
6. 机器人实时跟随视频中的人体动作

---

## 架构说明

### 多机器人抽象（IMimicAgent + Registry）

为了让 UI 脚本（Replay / Stop / StartInput）不硬绑死到 `G1mimicAgent`，每个 Agent 实现 `IMimicAgent` 接口：

```csharp
interface IMimicAgent
{
    string RobotKey { get; }                    // "unitree_g1" / "unitree_h1" ...
    GameObject AgentGameObject { get; }
    bool UseExternalReplayData { get; set; }
    bool ReplayMode { get; set; }
    int MotionId { get; set; }
    bool LoadReplayCsvFromPath(string filePath, bool keepProgress);
    void RequestEndEpisode();
    void ResetToInitialState();                 // 同步硬重置 articulation
}
```

每个 Agent 在 `Initialize()` 末尾调用 `MimicAgentRegistry.Instance.Register(this)` 注册自己。UI 脚本通过 `RobotKey`（如 `"unitree_g1"`）查找对应 Agent，向它发命令。新增机器人只需挂一个实现 `IMimicAgent` 的脚本并设好 `RobotKey`，UI 自动识别。

### 数据流

```
视频 / 摄像头
    │
    ▼
WHAM (Python)  ──→ GMR (Python)
                       │
                       ▼
                  live_motion.csv  (36 floats × N 帧, G1 格式)
                       │
                       │  StartInput 协程逐行读
                       ▼
              G1mimicAgent.ApplyRealtimeCsvToAgent()
                       │
                       │  FixedUpdate 设 xDrive.target + TeleportRoot
                       ▼
              ArticulationBody (Unity PhysX)
```

回放路径同理，只是数据源换成 `dataset/*.csv`。

### 可见性切换（重要细节）

下拉框切换机器人时 **绝对不能**用 `GameObject.SetActive(false/true)` 来隐藏机器人——Unity 会 tear down 再重建 ArticulationBody，rebuild 从上次 cache 状态出发（不是 prefab bind pose），多次切换后关节会累积错位到无法识别。

实际做法（`StartInput.ApplyRobotVisibility`）：

1. **首次激活**：如果选中的机器人当前 `activeSelf == false`，做一次性 `SetActive(true)`，让它 Initialize 干净跑一遍并注册到 Registry。之后这个机器人永远保持 active。
2. **可见性切换**：通过 `Renderer.enabled = false/true` 关闭非选中机器人的所有渲染组件（MeshRenderer + SkinnedMeshRenderer）。GameObject 不动，articulation 不重建。
3. **隐藏 Agent 的 replay 暂停**：把它们的 `ReplayMode` 和 `UseExternalReplayData` 设为 `false`，防止它们在后台偷偷 `TeleportRoot` 自己。

---

## CSV 数据格式

### G1 格式（36 列 / 行，30 Hz）

```
[root_pos_x, root_pos_y, root_pos_z, root_rot_x, root_rot_y, root_rot_z, root_rot_w, dof_0, ..., dof_28]
```

29 个 DOF 按 URDF 顺序：

| 索引 | 区段 | 关节 |
|---|---|---|
| 0..5 | 左腿 | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll |
| 6..11 | 右腿 | 同上 |
| 12..14 | 腰 | yaw, roll, pitch |
| 15..21 | 左臂 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw |
| 22..28 | 右臂 | 同上 |

### H1 格式（26 列 / 行，30 Hz）

```
[root_pos × 3, root_quat × 4, dof × 19]
```

19 个 DOF 按 URDF 顺序：

| 索引 | 区段 | 关节 |
|---|---|---|
| 0..4 | 左腿 | hip_yaw, hip_roll, hip_pitch, knee, ankle |
| 5..9 | 右腿 | 同上 |
| 10 | 躯干 | torso |
| 11..14 | 左臂 | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow |
| 15..18 | 右臂 | 同上 |

### 坐标系约定

- WHAM 输出：Y-up（OpenGL）
- GMR/MuJoCo：Z-up
- Unity：Y-up

Root 位置映射：`Unity = Vector3(-csv[1], csv[2], csv[0])`
Root 旋转映射（G1）：`Unity = Quaternion(-csv[1], csv[2], csv[0], -csv[3])`
关节 DOF：CSV 是弧度，Unity ArticulationBody xDrive.target 用角度，做 `* 180 / π` 转换。

### CSV → Unity 关节映射

**G1 和 H1 当前 prefab 都使用 1:1 identity mapping**——Unity 的 `GetComponentsInChildren<ArticulationBody>()` 深度优先遍历刚好产出 URDF 顺序。`G1mimicAgent` / `H1mimicAgent` 启动时会用 `DumpJointMapping()` 打印实际关节顺序到 Console，可对照表格验证。

如果将来换 prefab 出现顺序错乱，**不要**加 SerializedField 的置换表（Unity 会序列化旧值进场景导致难以调试），改成硬编码的 `private static readonly int[]` 常量。

---

## 关键脚本

### `G1mimicAgent.cs` / `H1mimicAgent.cs`

ML-Agents `Agent` 子类，支持三种模式：

- **Training**：PPO 训练，PD controller 跟踪参考 pose。奖励 = `live + rot_error + pos_error + dof_error`。
- **Replay**：从 `dataset/` 读 CSV，运动学（零重力 + TeleportRoot）回放。
- **Live**：`StartInput` 把实时 CSV 喂进来（`useExternalReplayData = true`）。

PD 增益：
- G1：`stiffness=180, damping=8`
- H1：`stiffness=2000, damping=200`（H1 更大是因为执行器更强）

### `Scripts/StartInput.cs`

- 启动 WHAM+GMR 子进程（Windows：`powershell -File run.ps1`；Linux：`bash run.sh`），所有配置通过环境变量传递（`OUTPUT_ROOT`、`ROBOT`、`WHAM_USE_AMP` 等）
- 协程轮询输出 CSV `csv/live_motion.csv`，把每一行喂给当前选中的 Agent
- 子进程生命周期管理：Windows 用 `taskkill /T` 杀进程树，Linux 用 `setsid` 包裹后 `kill -9 -<pgid>` 杀进程组
- RoboList 下拉切换：调 `OnRoboListChanged` → `ApplyRobotVisibility`（Renderer 切换） → 复位选中 Agent

### `Scripts/MimicAgentRegistry.cs`

单例 `MonoBehaviour`，第一次访问自动创建持有 GameObject。`Register(agent)` 把 Agent 按 `RobotKey` 加入字典，`FindByKey(key)` 查找。重复 key 的注册会被忽略并 warning。

### `Scripts/Replay.cs`

`Replay` 按钮的事件处理：
1. 从 CSV 下拉拿到选中文件名，解析为绝对路径
2. 优先调 `agent.LoadReplayCsvFromPath()`（适配不同机器人的 CSV 格式），成功就置 `UseExternalReplayData = true` 让 OnEpisodeBegin 不再覆盖
3. 失败时退回老的 `MotionId` 路径（按索引让 Agent 自己加载）
4. 触发 `RequestEndEpisode()` 让 OnEpisodeBegin 重新开始

### `Scripts/Stop.cs`

停掉 WHAM+GMR 管线，切换所有 Agent 到回放模式（不再消费实时 CSV，回到磁盘 dataset）。

---

## 训练

`config.yaml` 定义 `gewu` 行为：

- **Trainer**：PPO
- **Max steps**：200,000,000
- **Batch / Buffer**：2048 / 20480
- **Hidden layers**：3 × 512
- **Reward discount**：γ = 0.995
- **Entropy**：β = 0.005
- **Normalize observations**：true

启动训练（项目根目录）：

```bash
mlagents-learn config.yaml --run-id=gewu_run_1
```

然后 Unity 端按 Play。多实例并行训练通过 `G1mimicAgent.Start()` 里克隆 prefab 实现（默认 34 个 G1 副本 / 24 个 H1 副本，按 X 轴 2m 间距排开）。

---

## 重要细节 / 已知坑

### ArticulationBody Cache Size 必须探测

`arts[0].dofCount` 在 `immovable` 切换后**和引擎实际 cache size 不同步**（滞后一帧）。所有 `SetJointPositions/SetJointVelocities` 之前必须用 `GetJointPositions(probe)` 的 `Count` 拿真实 cache size，不能用 `dofCount` 求和。已封装在 `SafeSetJointPositions`。

### `immovable=true` 会让 cache 缩小 6 个槽

Root（6 DOF：3 pos + 3 rot）的 cache 槽在 `immovable=true` 时被剥离。如果 `restPositions` 是在 `immovable=false` 时捕获的（35 项 = 6 root + 29 joints），而当前 cache 是 29 项（joints only），直接截断前 29 会让 root 值挤进 joint 槽 → 整个 pose 错位。`AlignToCache()` 处理这个 case：源比 cache 长 6 项时取末尾 cache size 项（关节部分）。

### `Initialize()` 可能被 ML-Agents 重新触发

`Agent.OnEnable` 在每次 `SetActive(true)` 都会调 `LazyInitialize`。如果 Initialize 里捕获 `restPositions = GetJointPositions(...)`，二次触发会捕获到**上次 replay 末尾的脏 cache 值**而非 bind pose。守卫：

```csharp
if (restPositions == null)
{
    art0.GetJointPositions(P0);
    restPositions = P0.ToArray();
    ...
}
```

只在真正的首次 Initialize 捕获。

### 不要在 Agent 子类 override `OnEnable` / `OnDisable`

ML-Agents 的 `Agent.OnEnable` 不是 virtual，子类覆盖会屏蔽掉 `LazyInitialize → Initialize` 调用。Registry 注册放在 `Initialize()` 末尾。

### `Physics.gravity = Vector3.zero` 是全局污染

`G1mimicAgent.FixedUpdate` 的 replay 分支会把 `Physics.gravity` 设为零（避免 root teleport 和重力打架）。切机器人时必须在 `OnRoboListChanged` 里复位 `Physics.gravity = (0, -9.81, 0)`，否则下一个机器人会飘起来。

### CollectObservations 的防御写法

`Agent.OnDisable → NotifyAgentDone → CollectObservations` 在 SetActive(false) 时会触发，此刻 `jh[i].jointPosition.dofCount == 0`，直接 index 会抛 `IndexOutOfRangeException`。要么守卫 dofCount > 0 用零填充，要么避免 SetActive 流程（本项目采用后者 + 前者双保险）。

---

## 文件清单速查

| 文件 | 用途 |
|---|---|
| `G1.unity` | 主场景 |
| `G1Replay.unity` | 纯回放场景 |
| `config.yaml` | ML-Agents PPO 训练参数 |
| `g1m1.onnx` | G1 训好的策略 |
| `h1-all-in-one.onnx` | H1 训好的策略 |
| `dataset/*.csv` | G1 格式（36 列）的预录动作 |
| `Scripts/IMimicAgent.cs` | 多机器人统一接口 |
| `Scripts/MimicAgentRegistry.cs` | 场景级 Agent 注册表 |
| `Scripts/StartInput.cs` | 桥接 Python 管线 + 机器人切换 |

---

## 致谢

- WHAM: [Shin et al. CVPR 2024](https://github.com/yohanshin/WHAM)
- GMR: General Motion Retargeting
- Unity ML-Agents Toolkit
- Unitree G1 / H1 URDF 与官方 SDK
