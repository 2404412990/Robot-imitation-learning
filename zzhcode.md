1. MotionSmoothingFilter.cs — 运动平滑优化


using UnityEngine;
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

    // 误差统计
    private Queue<float>[] _errorHistories;
    private float[] _windowedErrors;

    // 关节绑定（由外部Agent注入）
    private ArticulationBody[] _joints;
    private int _dofCount;

    // 采样计数
    private int _sampleCount = 0;
    private bool _isRecording = false;

    [Tooltip("微调后的权重配置文件输出路径。")]
    public string weightConfigOutputPath = "Assets/Gewu/Imitation/Config/ik_weights_tuned.json";

    void Awake()
    {
        InitializeWeights();
    }

    private void InitializeWeights()
    {
        _currentWeights = new float[ikTargetNames.Length];
        _errorHistories = new Queue<float>[ikTargetNames.Length];
        _windowedErrors = new float[ikTargetNames.Length];

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

    private float[] GetDomainPresetWeights(MotionDomain domain)
    {
        switch (domain)
        {
            case MotionDomain.Dance:
                return new float[] { 0.9f, 0.6f, 0.6f, 1.0f, 1.0f, 0.7f };
            case MotionDomain.Sports:
                return new float[] { 1.0f, 1.0f, 1.0f, 0.4f, 0.4f, 0.3f };
            case MotionDomain.Locomotion:
                return new float[] { 1.0f, 1.0f, 1.0f, 0.2f, 0.2f, 0.1f };
            case MotionDomain.Manipulation:
                return new float[] { 0.7f, 0.3f, 0.3f, 1.0f, 1.0f, 0.4f };
            default:
                return new float[] { 1.0f, 0.8f, 0.8f, 0.5f, 0.5f, 0.3f };
        }
    }

    public void BindJoints(ArticulationBody[] joints, int dofCount)
    {
        _joints = joints;
        _dofCount = dofCount;
    }

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

    public void StopAndTune()
    {
        _isRecording = false;
        ComputeWeightAdjustment();
        ExportTunedWeights();
    }

    public void RecordFrameError(float[] referenceDof, float[] actualDof)
    {
        if (!_isRecording || _joints == null) return;
        if (referenceDof == null || actualDof == null) return;
        if (referenceDof.Length != actualDof.Length) return;

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

        if (_sampleCount % sampleWindowSize == 0)
        {
            ComputeWeightAdjustment();
        }
    }

    private float[] AggregateErrorsByTarget(float[] refDof, float[] actDof)
    {
        float[] targetErrors = new float[ikTargetNames.Length];
        int len = Mathf.Min(refDof.Length, actDof.Length, _dofCount);

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

    private void ExportTunedWeights()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"domain\": \"{currentDomain}\",");
        sb.AppendLine($"  \"sample_count\": {_sampleCount},");
        sb.AppendLine("  \"ik_targets\": [");
        for (int i = 0; i < ikTargetNames.Length; i++)
        {
            sb.AppendLine("    {");
            sb.AppendLine($"      \"name\": \"{ikTargetNames[i]}\",");
            sb.AppendLine($"      \"weight\": {_currentWeights[i]:F6},");
            sb.AppendLine($"      \"windowed_error\": {_windowedErrors[i]:F6}");
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







2. GmrIkWeightTuner.cs — 在线微调 IK 权重


[Uploading Motionusing UnityEngine;
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
SmoothingFilter.cs…]()
