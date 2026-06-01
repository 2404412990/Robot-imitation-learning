我构建了一条完整的人体运动生成与机器人重定向（retargeting）流程，实现了

视频 → 人体运动（Human Motion）→ G1机器人动作

的映射

具体通过利用 GVHMR 从视频中恢复人体运动参数，利用 GMR 将人体运动映射到 G1 机器人，最后在服务器环境下完成完整 pipeline 的运行与验证

完整实验流程：

Step 1：服务器环境搭建与登录：通过ssh -p 11111 group1@58.199.176.97进入学校服务器

Step 2：数据准备：通过使用 yt-dlp 从哔站上下载视频并上传至学校服务器

Step 3：运行 GVHMR：从视频中提取人体三维运动信息

Step 4：运行 GMR：将人体运动映射到 G1 机器人关节空间

Step 5：将其中的csv文件导入unity-gewu环境中再次进行仿真检验

<img src=docs/video/image.png alt="animated" />

### 4.17 — 方法探索与选型

**目标**：验证不同人体姿态估计方案对机器人重定向的效果影响。

**实验内容**：
- 使用 **RTMPose3D** 直接从视频提取 17 个 3D 关键点，跳过 SMPL 参数估计，直接映射到机器人关节
- **结论**：效果较差。RTMPose3D 输出的是稀疏关键点而非 SMPL 参数，缺少完整的人体运动学约束（无骨骼旋转、无全局旋转角），直接将 17 个点映射到 29 DOF G1 机器人关节导致动作失真
- **优点**：速度快、延迟低
- **后续计划**：尝试 TRACE (ROMP) 模型，该模型直接输出 SMPL 的 24 个高精度三维物理关节 + 真实根节点全局旋转角，并自带时序平滑

---

### 5.15 — 多机器人定向与 Windows 环境

**目标**：将 WHAM+GMR 管线从 Linux 服务器迁移到 Windows 本地，实现 Unity 内多机器人切换。

**完成内容**：
- 在 Windows 上搭建 WHAM+GMR 环境（CUDA 11.3 + PyTorch 1.11.0 + MuJoCo + ViTPose）
- 环境配置遇到大量编译问题（PyTorch3D / DPVO / setuptools 兼容性等），部分解决
- Unity 侧实现了多机器人场景切换：通过 `MimicAgentRegistry` 注册表模式 + `IMimicAgent` 接口，解耦 UI 与具体机器人实现
- 机器人切换时自动停止当前管道，通过 Renderer 可见性切换避免 ArticulationBody 状态损坏

**实现视频**：[多机器人切换演示](https://github.com/user-attachments/assets/6416208d-94b7-42b5-9819-7c58e32f8a8d)

---

### 5.22 — 管线调通与 Bug 排查

**目标**：修复 WHAM+GMR 管线在 Windows + Unity 联调中的问题。

**发现与解决**：

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| WHAM 只跑 3 帧就卡住 | Unity 侧 CSV 轮询速度 > Python 管道输出速度 | 调整 WHAM 推理参数（detect_interval/infer_interval） |
| 实时重映射时机器人抽搐 | CSV 写入与读取竞态——机器人读取速度远快于 WHAM 生成，Agent 反复读到同一帧再跳到新帧 | 降低读取频率、增加帧间插值平滑 |
| 离线 CSV 回放正常 | 完整 CSV 不存在竞态，证实问题在实时 I/O 而非重定向算法 |
| G1 小臂翻转 Bug | 旧 G1 prefab 关节遍历顺序与 URDF 不一致，CSV 中的值被写入了错误的关节插槽 | 验证关节映射顺序，采用 1:1 恒等映射（当前 prefab 已修正） |

**重要结论**：实时模式下抽搐的根因是 CSV 读写同步问题，不是 WHAM 或 GMR 的重定向质量问题。离线模式下动作质量良好。

---

### 6.1 — 环境完善、Bug 修复与多机器人适配

**目标**：彻底修复 Unity 编译错误、完善 WHAM+GMR Windows 环境配置、新增 X02Lite 机器人支持。

#### 6.1.1 Unity 项目修复

**编译错误清理**（~99 errors → 0 errors）：

| 修复项 | 详情 |
|--------|------|
| 删除重复脚本 | `Assets/unity_UI/` 下 7 个旧版 Agent/UI 脚本与 `Assets/Imitation/` 新版冲突，已全部删除 |
| 删除废弃模板 | `Assets/UI/` 旧模板代码（含 UIManager/VideoPlayerManager 冲突） |
| 移除无效 using | G1mimicAgent / H1mimicAgent / G1mimic1Agent 中 `using Unity.MLAgentsExamples;` 不存在，已删除 |
| 禁用警告即错误 | `csc.rsp` 中 `-warnaserror+` 已注释，避免未使用变量等警告阻塞编译 |
| 修复变量 | `G1mimicrealtime.cs` 中 `catch (Exception e)` → `catch (Exception)` |

#### 6.1.2 WHAM+GMR Windows 环境配置

**完整环境搭建**：

| 环境名 | Python | 核心依赖 | 用途 |
|--------|--------|---------|------|
| `gewu` | 3.10.12 | PyTorch 2.2.1, mlagents 1.1.0 | ML-Agents PPO 训练 |
| `wham_gmr` | 3.10 | PyTorch 1.11.0 + CUDA 11.3, MuJoCo, GMR | WHAM+GMR 实时管道 |

**关键修复**：
- `setup.py` 编码问题（Windows GBK → UTF-8）
- `run.ps1` 添加 `D:\anaconda3_envs` Python 路径自动检测
- `StartInput.cs` 修正 `bashWorkingDirectory` 默认路径
- 清华 conda 镜像不稳定 → 切换官方源 + 代理
- NumPy 2.x 与 PyTorch 1.11.0 不兼容 → 降级 NumPy 1.x
- `requirements.txt` 拆分：`requirements-gewu.txt`（训练）与 `Robot-imitation-learning/requirements.txt`（管道），避免混淆
- 创建 `setup_env.ps1` 自动化安装脚本

#### 6.1.3 X02Lite 机器人适配

**机器人规格**：10 DOF（纯下肢，每腿 5 关节），无手臂自由度。

**新增文件**：

| 文件 | 说明 |
|------|------|
| `X02LiteMimicAgent.cs` | Agent 脚本，实现 IMimicAgent，支持训练/回放/实时三种模式 |
| `smplx_to_x02lite.json` | GMR IK 重定向配置，7 个 IK 目标（pelvis + 双腿 6 链接） |
| `dataset/x02lite/neutral_stand.csv` | 示例数据集（200 帧 × 17 列） |

**StartInput.cs 修改**：
- `SupportedRobotNames` 添加 `"x02lite"`
- `RobotAliases` 添加 `"X02Lite"` / `"X02"`
- 新增 `PopulateRoboListDropdown()` 方法——启动时自动从 `SupportedRobotNames` + `sceneRobots` 填充下拉列表，无需手动配置

#### 6.1.4 文档与协作

- 创建 `CLAUDE.md`：项目架构、环境配置、训练命令、常见边界情况
- 创建 `操作手册.md`：从零搭建到运行的完整流程，含新机器人适配步骤模板
- 创建 `.gitignore`：排除 Python 管道、生成文件、IDE 配置
- 代码已推送至 GitHub `imitation` 分支

---
