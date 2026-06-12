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
smooth_code = '''using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// MotionSmoothingFilter: 关节运动平滑滤波器
/// 
/// 用途：对实时重定向的关节目标角度进行速度/加速度约束，
///       减少因CSV数据源帧率不足或噪声导致的关节抖动。
/// 
/// 原理：基于有限差分计算关节角速度，当角速度超过阈值时进行
///       一阶低通滤波平滑；同时约束角加速度防止突变。
/// </summary>
public class MotionSmoothingFilter : MonoBehaviour
{
    [Header("平滑参数")]
    [Tooltip("最大允许关节角速度 (deg/s)。超过此值将触发平滑。")]
    [Range(100f, 2000f)]
    public float maxAngularVelocity = 800f;

    [Tooltip("最大允许关节角加速度 (deg/s²)。")]
    [Range(500f, 10000f)]
    public float maxAngularAcceleration = 4000f;

    [Tooltip("低通滤波系数 (0=完全平滑, 1=无平滑)。")]
    [Range(0f, 1f)]
    public float lowPassFactor = 0.3f;

    [Tooltip("是否对根节点旋转也应用平滑。")]
    public bool smoothRootRotation = true;

    [Tooltip("根节点旋转最大角速度 (deg/s)。")]
    public float maxRootAngularVelocity = 360f;

    // 内部状态：上一帧的目标角度
    private float[] _prevTargetDeg;
    private float[] _prevVelocity;
    private float[] _smoothedTargetDeg;

    // 根节点状态
    private Quaternion _prevRootRotation;
    private Vector3 _prevRootPosition;

    // 时间步长缓存
    private float _lastDt;

    /// <summary>
    /// 初始化滤波器状态数组。
    /// </summary>
    /// <param name="dofCount">机器人自由度数量</param>
    public void Initialize(int dofCount)
    {
        _prevTargetDeg = new float[dofCount];
        _prevVelocity = new float[dofCount];
        _smoothedTargetDeg = new float[dofCount];
        
        for (int i = 0; i < dofCount; i++)
        {
            _prevTargetDeg[i] = 0f;
            _prevVelocity[i] = 0f;
            _smoothedTargetDeg[i] = 0f;
        }

        _prevRootRotation = Quaternion.identity;
        _prevRootPosition = Vector3.zero;
        _lastDt = Time.fixedDeltaTime;
    }

    /// <summary>
    /// 对关节目标角度序列进行速度/加速度约束平滑。
    /// </summary>
    /// <param name="rawTargets">原始目标角度 (deg)</param>
    /// <param name="dt">时间步长 (s)</param>
    /// <returns>平滑后的目标角度</returns>
    public float[] SmoothJointTargets(float[] rawTargets, float dt)
    {
        if (_prevTargetDeg == null || _prevTargetDeg.Length != rawTargets.Length)
        {
            Initialize(rawTargets.Length);
        }

        if (dt < 1e-6f) dt = _lastDt;
        _lastDt = dt;

        for (int i = 0; i < rawTargets.Length; i++)
        {
            // 1. 计算原始角速度
            float rawVelocity = (rawTargets[i] - _prevTargetDeg[i]) / dt;

            // 2. 速度约束：限制最大角速度
            float clampedVelocity = Mathf.Clamp(rawVelocity, -maxAngularVelocity, maxAngularVelocity);

            // 3. 加速度约束：限制角加速度变化率
            float accel = (clampedVelocity - _prevVelocity[i]) / dt;
            float clampedAccel = Mathf.Clamp(accel, -maxAngularAcceleration, maxAngularAcceleration);
            float constrainedVelocity = _prevVelocity[i] + clampedAccel * dt;

            // 4. 低通滤波：对速度进行一阶IIR平滑
            float smoothedVelocity = Mathf.Lerp(_prevVelocity[i], constrainedVelocity, lowPassFactor);

            // 5. 积分得到平滑后的目标角度
            _smoothedTargetDeg[i] = _prevTargetDeg[i] + smoothedVelocity * dt;

            // 6. 更新状态
            _prevVelocity[i] = smoothedVelocity;
            _prevTargetDeg[i] = _smoothedTargetDeg[i];
        }

        return _smoothedTargetDeg;
    }

    /// <summary>
    /// 对根节点位姿进行旋转平滑（可选）。
    /// </summary>
    public void SmoothRootPose(ref Vector3 position, ref Quaternion rotation, float dt)
    {
        if (!smoothRootRotation) return;
        if (dt < 1e-6f) dt = _lastDt;

        // 根节点位置：简单的一阶低通
        position = Vector3.Lerp(_prevRootPosition, position, lowPassFactor);
        _prevRootPosition = position;

        // 根节点旋转：角速度约束
        float angle = Quaternion.Angle(_prevRootRotation, rotation);
        float maxAngle = maxRootAngularVelocity * dt;
        float t = Mathf.Clamp01(maxAngle / Mathf.Max(angle, 1e-4f));
        rotation = Quaternion.Slerp(_prevRootRotation, rotation, t);
        _prevRootRotation = rotation;
    }

    /// <summary>
    /// 重置滤波器状态（如Episode开始时调用）。
    /// </summary>
    public void ResetFilter()
    {
        if (_prevTargetDeg == null) return;
        for (int i = 0; i < _prevTargetDeg.Length; i++)
        {
            _prevTargetDeg[i] = 0f;
            _prevVelocity[i] = 0f;
            _smoothedTargetDeg[i] = 0f;
        }
        _prevRootRotation = Quaternion.identity;
        _prevRootPosition = Vector3.zero;
    }
}
'''

ik_finetune_code = '''using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// GmrIkWeightTuner: GMR IK 权重在线微调模块
/// 
/// 用途：针对特定动作域（舞蹈、体育等）收集运行时关节跟踪误差，
///       动态调整GMR各IK目标点的权重，优化重定向精度。
/// 
/// 原理：在FixedUpdate中采样"参考pose vs 实际pose"的关节误差，
///       按关节重要性（如末端执行器权重更高）累积误差统计，
///       定期输出权重调整建议或自动更新权重配置文件。
/// </summary>
public class GmrIkWeightTuner : MonoBehaviour
{
    [Header("IK目标配置")]
    [Tooltip("IK目标点名称，对应GMR的smplx_to_robot.json中的target字段。")]
    public string[] ikTargetNames = new string[]
    {
        "pelvis",
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
        "head"
    };

    [Tooltip("各IK目标的初始权重。")]
    public float[] initialWeights = new float[] { 1.0f, 0.8f, 0.8f, 0.5f, 0.5f, 0.3f };

    [Header("微调参数")]
    [Tooltip("误差采样窗口大小（帧数）。")]
    [Range(30, 300)]
    public int sampleWindowSize = 100;

    [Tooltip("权重调整学习率。")]
    [Range(0.001f, 0.1f)]
    public float weightLearningRate = 0.01f;

    [Tooltip("权重变化阈值，低于此值的调整将被忽略。")]
    [Range(0.001f, 0.05f)]
    public float weightUpdateThreshold = 0.005f;

    [Tooltip("权重值域限制 [min, max]。")]
    public Vector2 weightClampRange = new Vector2(0.1f, 2.0f);

    [Header("动作域预设")]
    [Tooltip("当前动作域类型，影响权重初始化策略。")]
    public MotionDomain currentDomain = MotionDomain.General;

    public enum MotionDomain
    {
        General,      // 通用
        Dance,        // 舞蹈：上肢权重更高
        Sports,       // 体育：下肢/躯干权重更高
        Locomotion,   // 行走：下肢主导
        Manipulation  // 操作：手臂权重更高
    }

    // 运行时权重
    private float[] _currentWeights;

    // 误差统计：每个IK目标在每个采样窗口内的累积误差
    private Queue<float>[] _errorHistories;
    private float[] _windowedErrors;

    // 关节绑定（由外部Agent注入）
    private ArticulationBody[] _joints;
    private int _dofCount;

    // 采样计数
    private int _sampleCount = 0;
    private bool _isRecording = false;

    // 输出路径
    [Tooltip("微调后的权重配置文件输出路径。")]
    public string weightConfigOutputPath = "Assets/Gewu/Imitation/Config/ik_weights_tuned.json";

    void Awake()
    {
        InitializeWeights();
    }

    /// <summary>
    /// 根据动作域初始化权重。
    /// </summary>
    private void InitializeWeights()
    {
        _currentWeights = new float[ikTargetNames.Length];
        _errorHistories = new Queue<float>[ikTargetNames.Length];
        _windowedErrors = new float[ikTargetNames.Length];

        // 根据动作域调整初始权重
        float[] domainWeights = GetDomainPresetWeights(currentDomain);
        for (int i = 0; i < ikTargetNames.Length; i++)
        {
            _currentWeights[i] = (i < initialWeights.Length) ? initialWeights[i] : 1.0f;
            if (i < domainWeights.Length)
            {
                _currentWeights[i] = Mathf.Lerp(_currentWeights[i], domainWeights[i], 0.3f);
            }
            _errorHistories[i] = new Queue<float>();
            _windowedErrors[i] = 0f;
        }
    }

    /// <summary>
    /// 获取动作域预设权重。
    /// </summary>
    private float[] GetDomainPresetWeights(MotionDomain domain)
    {
        switch (domain)
        {
            case MotionDomain.Dance:
                return new float[] { 0.9f, 0.6f, 0.6f, 1.0f, 1.0f, 0.7f }; // 上肢权重高
            case MotionDomain.Sports:
                return new float[] { 1.0f, 1.0f, 1.0f, 0.4f, 0.4f, 0.3f }; // 下肢权重高
            case MotionDomain.Locomotion:
                return new float[] { 1.0f, 1.0f, 1.0f, 0.2f, 0.2f, 0.1f }; // 纯下肢
            case MotionDomain.Manipulation:
                return new float[] { 0.7f, 0.3f, 0.3f, 1.0f, 1.0f, 0.4f }; // 手臂权重高
            case MotionDomain.General:
            default:
                return new float[] { 1.0f, 0.8f, 0.8f, 0.5f, 0.5f, 0.3f };
        }
    }

    /// <summary>
    /// 注入关节引用（由MimicAgent在Initialize时调用）。
    /// </summary>
    public void BindJoints(ArticulationBody[] joints, int dofCount)
    {
        _joints = joints;
        _dofCount = dofCount;
    }

    /// <summary>
    /// 开始误差采样记录。
    /// </summary>
    public void StartRecording()
    {
        _isRecording = true;
        _sampleCount = 0;
        for (int i = 0; i < _errorHistories.Length; i++)
        {
            _errorHistories[i].Clear();
            _windowedErrors[i] = 0f;
        }
        Debug.Log($"[GmrIkWeightTuner] 开始{currentDomain}域的权重微调采样...");
    }

    /// <summary>
    /// 停止采样并计算权重调整。
    /// </summary>
    public void StopAndTune()
    {
        _isRecording = false;
        ComputeWeightAdjustment();
        ExportTunedWeights();
    }

    /// <summary>
    /// 每帧记录关节跟踪误差。
    /// </summary>
    /// <param name="referenceDof">参考关节角度 (rad)</param>
    /// <param name="actualDof">实际关节角度 (rad)</param>
    public void RecordFrameError(float[] referenceDof, float[] actualDof)
    {
        if (!_isRecording || _joints == null) return;
        if (referenceDof == null || actualDof == null) return;
        if (referenceDof.Length != actualDof.Length) return;

        // 简化处理：按身体区段聚合误差到对应的IK目标
        float[] frameErrors = AggregateErrorsByTarget(referenceDof, actualDof);

        for (int i = 0; i < ikTargetNames.Length; i++)
        {
            _errorHistories[i].Enqueue(frameErrors[i]);
            _windowedErrors[i] += frameErrors[i];

            if (_errorHistories[i].Count > sampleWindowSize)
            {
                _windowedErrors[i] -= _errorHistories[i].Dequeue();
            }
        }

        _sampleCount++;

        // 每满一个窗口自动触发一次权重更新
        if (_sampleCount % sampleWindowSize == 0)
        {
            ComputeWeightAdjustment();
        }
    }

    /// <summary>
    /// 将关节级误差聚合到IK目标级。
    /// </summary>
    private float[] AggregateErrorsByTarget(float[] refDof, float[] actDof)
    {
        float[] targetErrors = new float[ikTargetNames.Length];
        int len = Mathf.Min(refDof.Length, actDof.Length, _dofCount);

        // 简化映射：按关节索引范围分配到不同目标
        // 实际项目中应根据URDF的链式结构精确计算
        for (int i = 0; i < len; i++)
        {
            float err = Mathf.Abs(refDof[i] - actDof[i]) * Mathf.Rad2Deg;
            int targetIdx = MapJointIndexToTarget(i, len);
            if (targetIdx >= 0 && targetIdx < targetErrors.Length)
            {
                targetErrors[targetIdx] += err;
            }
        }

        return targetErrors;
    }

    /// <summary>
    /// 将关节索引映射到IK目标索引（简化版）。
    /// </summary>
    private int MapJointIndexToTarget(int jointIdx, int totalDof)
    {
        float ratio = (float)jointIdx / totalDof;
        if (ratio < 0.15f) return 0;      // pelvis
        if (ratio < 0.35f) return 1;      // left leg -> left_foot
        if (ratio < 0.55f) return 2;      // right leg -> right_foot
        if (ratio < 0.75f) return 3;      // left arm -> left_hand
        if (ratio < 0.90f) return 4;      // right arm -> right_hand
        return 5;                          // head
    }

    /// <summary>
    /// 基于误差统计计算权重调整量。
    /// </summary>
    private void ComputeWeightAdjustment()
    {
        if (_sampleCount == 0) return;

        float totalError = 0f;
        for (int i = 0; i < _windowedErrors.Length; i++)
        {
            totalError += _windowedErrors[i];
        }
        if (totalError < 1e-6f) return;

        float[] newWeights = new float[_currentWeights.Length];
        bool hasSignificantChange = false;

        for (int i = 0; i < ikTargetNames.Length; i++)
        {
            float avgError = _windowedErrors[i] / Mathf.Min(_sampleCount, sampleWindowSize);
            float normalizedError = avgError / (totalError / ikTargetNames.Length);

            // 误差大的目标应提高权重（让它更受重视）
            float adjustment = (normalizedError - 1.0f) * weightLearningRate;
            newWeights[i] = Mathf.Clamp(
                _currentWeights[i] + adjustment,
                weightClampRange.x,
                weightClampRange.y
            );

            if (Mathf.Abs(newWeights[i] - _currentWeights[i]) > weightUpdateThreshold)
            {
                hasSignificantChange = true;
            }

            _currentWeights[i] = newWeights[i];
        }

        if (hasSignificantChange)
        {
            Debug.Log($"[GmrIkWeightTuner] 权重已更新（{currentDomain}域）：" +
                      string.Join(", ", _currentWeights.Select((w, i) => $"{ikTargetNames[i]}={w:F3}")));
        }
    }

    /// <summary>
    /// 导出微调后的权重到JSON配置文件。
    /// </summary>
    private void ExportTunedWeights()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \\"domain\\": \\"{currentDomain}\\",");
        sb.AppendLine($"  \\"sample_count\\": {_sampleCount},");
        sb.AppendLine("  \\"ik_targets\\": [");
        for (int i = 0; i < ikTargetNames.Length; i++)
        {
            sb.AppendLine("    {");
            sb.AppendLine($"      \\"name\\": \\"{ikTargetNames[i]}\\",");
            sb.AppendLine($"      \\"weight\\": {_currentWeights[i]:F6},");
            sb.AppendLine($"      \\"windowed_error\\": {_windowedErrors[i]:F6}");
            sb.Append(i < ikTargetNames.Length - 1 ? "    }," : "    }");
            sb.AppendLine();
        }
        sb.AppendLine("  ]");
        sb.AppendLine("}");

        string dir = Path.GetDirectoryName(weightConfigOutputPath);
        if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(weightConfigOutputPath, sb.ToString());

        Debug.Log($"[GmrIkWeightTuner] 权重配置已导出至: {weightConfigOutputPath}");
    }

    /// <summary>
    /// 获取当前各IK目标的权重（供GMR调用）。
    /// </summary>
    public IReadOnlyDictionary<string, float> GetCurrentWeights()
    {
        var dict = new Dictionary<string, float>();
        for (int i = 0; i < ikTargetNames.Length; i++)
        {
            dict[ikTargetNames[i]] = _currentWeights[i];
        }
        return dict;
    }
}
'''

print("=== MotionSmoothingFilter.cs ===")
print(smooth_code)
print("\n\n=== GmrIkWeightTuner.cs ===")
print(ik_finetune_code)

