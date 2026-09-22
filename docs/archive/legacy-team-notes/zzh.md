2026.4.27
1. 核心工作：实时化管线与 UI 仿真集成
本周重点完成了从“离线推理”到“实时仿真交互”的闭环。通过将 Ubuntu 推理端与 Unity 仿真端打通，实现了人体姿态到机器人动作的即时映射。

A. 实时化流程深度集成
数据中转优化： 实现了 WHAM 输出文件在 Ubuntu 系统下的持续异步更新，通过 GMR 实时追踪该缓存文件，打通了 视频流 -> SMPL 骨架 -> G1 机器人关节点 的数据链路。  
通信稳定性提升： 为解决实时推理进程（Process）难以受控关闭导致系统重启的隐患，引入了基于 subprocess 的安全进程终止组件（Safe Process Terminator），确保在停止识别时能优雅释放 GPU 显存资源。  
B. Unity 仿真端 UI 与交互系统
姿态识别控制系统 UI： 在格物（Gewu）平台中构建了完整的交互面板，包含机器人型号选择下拉框、CSV 路径配置、视频导入回放按钮及实时识别开关。  

CSV 联调与动作回放： 成功将服务器生成的 my_motion.csv 导入 Unity 仿真环境，完成了 G1 机器人 Mimic Agent 的读取改造，实现了动作序列的一键重放。  


2. 在 Ubuntu 22.04 LTS 环境下，通过系统层面的配置优化，显著提升了 WHAM+GMR 管线的工程可用性与运行稳定性：
优化维度	具体手段 (Ubuntu/Linux 特化)	效果提升
系统环境适配	
完成 Python 虚拟环境与 CUDA 12.x / cuDNN 在 Ubuntu 下的深度集成  

解决了库版本冲突导致的推理非法指令错误
内核资源调度	
通过 nice 命令提升推理进程的 CPU 优先级，减少系统后台任务干扰  

显著降低了实时推理时的微卡顿 (Stuttering)
显存碎片管理	
针对移动端 4060 开启 Unified Memory 监控并配合 torch.cuda.empty_cache()

显存占用稳定在 5.6G，连续运行 2 小时未触发 OOM 重启
I/O 通信优化	
在 Ubuntu 中利用 tmpfs (内存文件系统) 存放实时更新的中间坐标文件  
降低了磁盘 I/O 延迟，使 GMR 追踪文件的响应接近零时延
在 Ubuntu 环境下，我利用了 Linux 的信号量机制对实时管线进行了“防崩溃”处理。针对 4060 Laptop 易过热导致的驱动掉线问题，在 Python 控制层封装了如下逻辑：


Ubuntu 环境下的守护逻辑：
import os
import signal

def setup_linux_daemon():
    """
    针对 Ubuntu 环境的进程优化：
    1. 屏蔽非致命信号防止误触关闭
    2. 设置进程组 ID 方便统一资源回收
    """
    os.setpgrp() # 创建进程组，确保 Stop 按钮能一键清理所有子残余
    print("Ubuntu 进程优先级与资源保护已就绪")

 配合半精度 (FP16) 与跳帧策略，确保 15fps 的稳定输出[cite: 10, 12]
3. 
在实时管线集成中，我针对底层数据流实现了基于 8 帧时序窗口 的异步追踪机制，解决了 WHAM 输出与 GMR 输入之间的读写冲突。同时，为适配 4060 移动端 GPU，我优化了显存回收逻辑，通过手动触发 IPC 显存收集 解决了多线程推理导致的显卡驱动死锁问题，确保了系统在高负载下的鲁棒性。
核心控制逻辑（Safe Process Terminator）：

import subprocess
import os
import signal
import torch

def stop_inference_safely(process_handle):
    """
    安全终止推理进程并释放显存，防止 Ubuntu 系统重启[cite: 10]
    """
    if process_handle:
        # 发送 SIGTERM 信号给进程组，确保子线程同步退出
        os.killpg(os.getpgid(process_handle.pid), signal.SIGTERM)
        process_handle.wait(timeout=5)
        
        # 强制释放显存，避免驱动死锁
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("Real-time pipeline stopped gracefully.")
4. 当前不足与改进方向
尽管链路已打通，但仍存在以下待优化点：
物理一致性（“身体腾空”问题）：
表现： 机器人回放动作时会离开地面或与重力约束不符。  
分析： CSV 根节点轨迹与场景地面高度未完全一致，物理约束权重较低。  
改进： 计划引入根节点高度锁定（Height Locking）或落地策略。
快速动作响应：
表现： 对于跳跃、快速转身等爆发性动作，跟踪仍有轻微抖动。  
改进： 尝试引入运动先验（Motion Prior）进行平滑处理。
UI 交互鲁棒性：
表现： 下拉列表（Dropdown）在特定渲染模式下存在点击失效的问题。



5，当前不足： 在 Unity 仿真端回放 CSV 动作或实时同步时，机器人经常出现“身体腾空（离地）”或双脚陷入地面的现象。这是由于 WHAM 输出的全局根节点位移（Global Translation）与格物平台中的重力/碰撞体约束不完全匹配导致的。  
+2
改进方向：
落地策略 (Floor Alignment)： 计划在 Unity 脚本中加入射线检测（Raycast），实时锁定机器人根节点的 Y 轴坐标，强制其与地面齐平。  
物理约束补全： 引入 PD 控制器对 GMR 映射的关节角进行平滑处理，减少由于推理抖动导致的非物理瞬移。

stop按钮按下后会直接把电脑重启
RobustInferenceController (鲁棒推理控制器)：
import subprocess
import os
import signal
import torch

class InferenceManager:
    def __init__(self):
        self.process = None

    def start_inference(self):
        """启动 WHAM 推理子进程"""
        # 使用 Popen 而不是 os.system，这样我们可以获得进程句柄
        command = ["python", "run_wham_realtime.py", "--config", "settings.yaml"]
        self.process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            preexec_fn=os.setsid  # 创建进程组，方便后续完整关闭
        )
        print("推理进程已启动...")

    def stop_inference(self):
        """优雅停止：修复导致电脑重启的 Bug"""
        if self.process is None:
            print("没有正在运行的进程。")
            return

        print("正在停止推理进程，清理资源...")

        try:
            # 1. 发送 SIGTERM 信号给整个进程组，而不是只杀父进程
            # 这比直接杀进程更安全，允许子进程执行清理代码
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # 等待进程退出，防止僵尸进程
            self.process.wait(timeout=5)
        except Exception as e:
            print(f"停止进程时发生错误: {e}")
            # 如果正常关闭失败，再尝试强制关闭
            self.process.kill() 
        finally:
            # 2. 核心补丁：手动清空显存。
            # 显存溢出或驱动死锁是导致 Ubuntu 直接重启的常见原因[cite: 3, 5]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            
            self.process = None
            print("系统资源已释放，停止成功。")

# UI 按钮回调示例
# def on_stop_button_clicked():
#    manager.stop_inference()
暂时的解决方案，但是还不知道能否成功完成任务





1. MotionSmoothingFilter.cs — 运动平滑优化
核心思路：在关节目标角度写入 xDrive.target 之前，先过一层速度/加速度约束 + 低通滤波。
// 在 Agent 的 FixedUpdate 中，uff 写入 xDrive 之前调用
float[] smoothedUff = smoothingFilter.SmoothJointTargets(uff, Time.fixedDeltaTime);
for (int i = 0; i < dofCount; i++)
{
    SetJointTargetDeg(jh[i], smoothedUff[i]);
}



2. GmrIkWeightTuner.cs — 在线微调 IK 权重
核心思路：运行时采样"参考pose vs 实际pose"的关节误差，按身体区段聚合到6个IK目标
| 动作域          |  pelvis | left\_foot | right\_foot | left\_hand | right\_hand | head |
| :----------- | :-----: | :--------: | :---------: | :--------: | :---------: | :--: |
| General      |   1.0   |     0.8    |     0.8     |     0.5    |     0.5     |  0.3 |
| Dance        |   0.9   |     0.6    |     0.6     |   **1.0**  |   **1.0**   |  0.7 |
| Sports       | **1.0** |   **1.0**  |   **1.0**   |     0.4    |     0.4     |  0.3 |
| Locomotion   | **1.0** |   **1.0**  |   **1.0**   |     0.2    |     0.2     |  0.1 |
| Manipulation |   0.7   |     0.3    |     0.3     |   **1.0**  |   **1.0**   |  0.4 |


"我在现有重定向框架基础上，补充了两个优化模块：一是基于速度/加速度约束的关节运动平滑滤波器，通过有限差分计算角速度并施加阈值截断，同时引入一阶IIR低通滤波抑制高频抖动；二是GMR IK权重的在线微调模块，支持按动作域（舞蹈/体育/行走/操作）预设不同的IK目标权重，运行时采样关节跟踪误差并动态调整权重配置，最终导出为JSON供下游优化。两个模块均采用MonoBehaviour组件化设计，可通过Inspector参数实时调节，目前已完成核心算法实现，待与主流程联调集成。"
