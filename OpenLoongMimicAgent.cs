using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using Gewu.Imitation;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class OpenLoongMimicAgent : Agent, IMimicAgent, IRealtimeCsvMimicAgent, ISelectableMimicAgent, IReplayRootOffsetMimicAgent
{
    private const int RootPosCols = 3;
    private const int RootRotCols = 4;
    private const int DofCols = 31;
    private const int ExpectedCols = RootPosCols + RootRotCols + DofCols;

    private static readonly string[] CsvJointNames =
    {
        "J_head_yaw",
        "J_head_pitch",
        "J_arm_r_01",
        "J_arm_r_02",
        "J_arm_r_03",
        "J_arm_r_04",
        "J_arm_r_05",
        "J_arm_r_06",
        "J_arm_r_07",
        "J_arm_l_01",
        "J_arm_l_02",
        "J_arm_l_03",
        "J_arm_l_04",
        "J_arm_l_05",
        "J_arm_l_06",
        "J_arm_l_07",
        "J_waist_pitch",
        "J_waist_roll",
        "J_waist_yaw",
        "J_hip_r_roll",
        "J_hip_r_yaw",
        "J_hip_r_pitch",
        "J_knee_r_pitch",
        "J_ankle_r_pitch",
        "J_ankle_r_roll",
        "J_hip_l_roll",
        "J_hip_l_yaw",
        "J_hip_l_pitch",
        "J_knee_l_pitch",
        "J_ankle_l_pitch",
        "J_ankle_l_roll",
    };

    private static readonly string[] CsvBodyNames =
    {
        "Link_head_yaw",
        "Link_head_pitch",
        "Link_arm_r_01",
        "Link_arm_r_02",
        "Link_arm_r_03",
        "Link_arm_r_04",
        "Link_arm_r_05",
        "Link_arm_r_06",
        "Link_arm_r_07",
        "Link_arm_l_01",
        "Link_arm_l_02",
        "Link_arm_l_03",
        "Link_arm_l_04",
        "Link_arm_l_05",
        "Link_arm_l_06",
        "Link_arm_l_07",
        "Link_waist_pitch",
        "Link_waist_roll",
        "Link_waist_yaw",
        "Link_hip_r_roll",
        "Link_hip_r_yaw",
        "Link_hip_r_pitch",
        "Link_knee_r_pitch",
        "Link_ankle_r_pitch",
        "Link_ankle_r_roll",
        "Link_hip_l_roll",
        "Link_hip_l_yaw",
        "Link_hip_l_pitch",
        "Link_knee_l_pitch",
        "Link_ankle_l_pitch",
        "Link_ankle_l_roll",
    };

    private static readonly float[] UnityDriveSigns =
    {
        1f, 1f,
        1f, 1f, 1f, 1f, -1f, -1f, -1f,
        1f, -1f, 1f, 1f, 1f, 1f, 1f,
        1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f, 1f,
    };

    private static readonly float[] UnityDriveOffsetsRad =
    {
        0f, 0f,
        0f, 0f, 0f, 0f, 0f, 0f, 0f,
        0f, 0f, 0f, 0f, 0f, 0f, 0f,
        0f, 0f, 0f,
        0f, 0f, 0f, 0f, 0f, 0f,
        0f, 0f, 0f, 0f, 0f, 0f,
    };

    private static readonly UnityRetargetCalibrationEntry[] UnityCalibration =
    {
        new UnityRetargetCalibrationEntry(0,  "J_head_yaw",          1f, 0f),
        new UnityRetargetCalibrationEntry(1,  "J_head_pitch",        1f, 0f),
        new UnityRetargetCalibrationEntry(2,  "J_arm_r_01",          1f, 0f),
        new UnityRetargetCalibrationEntry(3,  "J_arm_r_02",          1f, 0f),
        new UnityRetargetCalibrationEntry(4,  "J_arm_r_03",          1f, 0f),
        new UnityRetargetCalibrationEntry(5,  "J_arm_r_04",          1f, 0f),
        new UnityRetargetCalibrationEntry(6,  "J_arm_r_05",         -1f, 0f),
        new UnityRetargetCalibrationEntry(7,  "J_arm_r_06",         -1f, 0f),
        new UnityRetargetCalibrationEntry(8,  "J_arm_r_07",         -1f, 0f),
        new UnityRetargetCalibrationEntry(9,  "J_arm_l_01",          1f, 0f),
        new UnityRetargetCalibrationEntry(10, "J_arm_l_02",         -1f, 0f),
        new UnityRetargetCalibrationEntry(11, "J_arm_l_03",          1f, 0f),
        new UnityRetargetCalibrationEntry(12, "J_arm_l_04",          1f, 0f),
        new UnityRetargetCalibrationEntry(13, "J_arm_l_05",          1f, 0f),
        new UnityRetargetCalibrationEntry(14, "J_arm_l_06",          1f, 0f),
        new UnityRetargetCalibrationEntry(15, "J_arm_l_07",          1f, 0f),
        new UnityRetargetCalibrationEntry(16, "J_waist_pitch",       1f, 0f),
        new UnityRetargetCalibrationEntry(17, "J_waist_roll",        1f, 0f),
        new UnityRetargetCalibrationEntry(18, "J_waist_yaw",         1f, 0f),
        new UnityRetargetCalibrationEntry(19, "J_hip_r_roll",        1f, 0f),
        new UnityRetargetCalibrationEntry(20, "J_hip_r_yaw",         1f, 0f),
        new UnityRetargetCalibrationEntry(21, "J_hip_r_pitch",       1f, 0f),
        new UnityRetargetCalibrationEntry(22, "J_knee_r_pitch",      1f, 0f),
        new UnityRetargetCalibrationEntry(23, "J_ankle_r_pitch",     1f, 0f),
        new UnityRetargetCalibrationEntry(24, "J_ankle_r_roll",      1f, 0f),
        new UnityRetargetCalibrationEntry(25, "J_hip_l_roll",        1f, 0f),
        new UnityRetargetCalibrationEntry(26, "J_hip_l_yaw",         1f, 0f),
        new UnityRetargetCalibrationEntry(27, "J_hip_l_pitch",       1f, 0f),
        new UnityRetargetCalibrationEntry(28, "J_knee_l_pitch",      1f, 0f),
        new UnityRetargetCalibrationEntry(29, "J_ankle_l_pitch",     1f, 0f),
        new UnityRetargetCalibrationEntry(30, "J_ankle_l_roll",      1f, 0f),
    };

    [SerializeField] private string robotKey = "openloong";
    [SerializeField] private bool replay;
    [SerializeField] private bool useExternalReplayData;
    [SerializeField] private string datasetRelativePath = "Assets/Gewu/Imitation/dataset/openloong";
    [SerializeField] private int frame0;
    [SerializeField] private float jointStiffness = 2000f;
    [SerializeField] private float jointDamping = 200f;
    [SerializeField] private bool logJointMappingOnStart = true;
    [Tooltip("Correct Unity articulation reduced-axis signs that differ from MuJoCo qpos axes. Does not change CSV order.")]
    [SerializeField] private bool applyMjcfAxisSignCorrection = true;
    [SerializeField] private bool clampTargetsToDriveLimits = true;
    [SerializeField] private float replayJointForceLimit = 300f;
    [SerializeField] private bool keepRootImmovableDuringReplay = true;
    [SerializeField] private bool writeReplayJointPositionsDirectly = true;
    [SerializeField] private bool zeroReplayJointVelocitiesOnDirectWrite = true;
    [Tooltip("Keep OpenLoong wrist axes neutral in replay/live when MuJoCo wrist qpos does not match the Unity hand rig.")]
    [SerializeField] private bool neutralizeWristDofsInMirror = true;
    [SerializeField] private float wristNeutralRad = 0f;
    [SerializeField] private bool logRetargetCalibrationDiagnostics;
    [SerializeField] private int retargetCalibrationLogInterval = 120;

    private enum ReplayDataMode
    {
        DatasetReplay,
        ExternalCsvReplay,
        LiveRealtimeCsv
    }

    private ReplayDataMode replayDataMode = ReplayDataMode.DatasetReplay;

    private static readonly string[] DefaultDatasetSearchPaths =
    {
        "Assets/Gewu/Imitation/dataset/openloong",
        "Assets/Imitation/dataset/openloong",
    };

    private readonly float[] uff = new float[DofCols];
    private readonly float[] u = new float[DofCols];
    private readonly float[] utotal = new float[DofCols];
    private readonly ArticulationBody[] jh = new ArticulationBody[DofCols];
    private readonly List<float> positionCache = new List<float>();
    private readonly List<float> velocityCache = new List<float>();

    private ArticulationBody[] arts = Array.Empty<ArticulationBody>();
    private ArticulationBody rootArticulation;
    private Transform body;
    private Vector3 pos0;
    private Quaternion rot0;
    private float[] restPositions = Array.Empty<float>();
    private float[] restVelocities = Array.Empty<float>();
    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();
    private readonly float[] realtimeSampledRow = new float[ExpectedCols];
    private int currentFrame;
    private int realtimePreviousMaxStep = -1;
    private float realtimeFrameCursor;
    private float realtimePlaybackFps = ReplayCsvUtility.SourceFps;
    private float realtimePlaybackBufferSeconds = 0.2f;
    private int tt;
    private Vector3 newPosition;
    private Quaternion newRotation;
    private int appliedRowDebugCount;
    private bool isRobotSelectedInScene = true;
    private bool holdNeutralPose = true;
    // private bool neutralPoseRestorePending = true;
    private bool hasLoggedDirectJointStateError;
    private Vector3 replayRootOffset = Vector3.zero;
    private int retargetCalibrationLogCounter;

    public string RobotKey => string.IsNullOrWhiteSpace(robotKey) ? "openloong" : robotKey.Trim();
    public GameObject AgentGameObject => this == null ? null : gameObject;
    public bool UseExternalReplayData
    {
        get => useExternalReplayData;
        set
        {
            useExternalReplayData = value;
            if (!value)
            {
                replayDataMode = ReplayDataMode.DatasetReplay;
                if (!replay)
                {
                    QueueNeutralPoseRestore();
                }
            }
            else if (replayDataMode != ReplayDataMode.LiveRealtimeCsv)
            {
                holdNeutralPose = false;
                // neutralPoseRestorePending = false;
                replayDataMode = ReplayDataMode.ExternalCsvReplay;
            }
        }
    }
    public bool ReplayMode
    {
        get => replay;
        set
        {
            replay = value;
            if (value)
            {
                holdNeutralPose = false;
                // neutralPoseRestorePending = false;
            }
            else if (!useExternalReplayData)
            {
                QueueNeutralPoseRestore();
            }
        }
    }
    public int MotionId { get; set; }
    public void RequestEndEpisode() => EndEpisode();
    public int ExpectedCsvColumns => ExpectedCols;
    private bool IsLiveRealtimeCsv => useExternalReplayData && replayDataMode == ReplayDataMode.LiveRealtimeCsv;

    private bool ShouldHoldNeutralPose()
    {
        return !replay && !useExternalReplayData && holdNeutralPose;
    }

    private void QueueNeutralPoseRestore()
    {
        holdNeutralPose = true;
        // neutralPoseRestorePending = true;
    }

    private void ApplyNeutralPoseNow()
    {
        if (rootArticulation == null)
        {
            return;
        }

        rootArticulation.immovable = false;
        rootArticulation.TeleportRoot(pos0, rot0);
        rootArticulation.velocity = Vector3.zero;
        rootArticulation.angularVelocity = Vector3.zero;
        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        for (int i = 0; i < DofCols; i++)
        {
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
            SetJointTargetDeg(jh[i], 0f);
        }

        currentFrame = frame0;
        realtimeFrameCursor = 0f;
        tt = 0;
        rootArticulation.immovable = true;
        // neutralPoseRestorePending = false;
    }

    public void SetReplayRootOffset(Vector3 offset)
    {
        replayRootOffset = offset;
    }

    public void SetRobotSelectedInScene(bool isSelected)
    {
        isRobotSelectedInScene = isSelected;
        if (rootArticulation == null)
        {
            return;
        }

        if (isSelected)
        {
            return;
        }

        UseExternalReplayData = false;
        ReplayMode = false;
        replayDataMode = ReplayDataMode.DatasetReplay;
        rootArticulation.immovable = false;
        rootArticulation.TeleportRoot(pos0, rot0);
        rootArticulation.velocity = Vector3.zero;
        rootArticulation.angularVelocity = Vector3.zero;
        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        for (int i = 0; i < DofCols; i++)
        {
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
            SetJointTargetDeg(jh[i], 0f);
        }

        currentFrame = frame0;
        realtimeFrameCursor = 0f;
        tt = 0;
        rootArticulation.immovable = true;
    }

    public bool BeginRealtimeCsv()
    {
        if (realtimePreviousMaxStep < 0)
        {
            realtimePreviousMaxStep = MaxStep;
        }

        MaxStep = 0;
        replayDataMode = ReplayDataMode.LiveRealtimeCsv;
        refData = new List<float[]>();
        itpData = new List<float[]>();
        currentFrame = 0;
        realtimeFrameCursor = 0f;
        appliedRowDebugCount = 0;
        tt = 0;
        return true;
    }

    public void SetRealtimePlaybackRate(float framesPerSecond, float bufferSeconds)
    {
        realtimePlaybackFps = ReplayCsvUtility.ClampRealtimeFps(framesPerSecond);
        realtimePlaybackBufferSeconds = Mathf.Max(0f, bufferSeconds);
    }

    public bool AppendRealtimeCsvRows(IReadOnlyList<float[]> rows)
    {
        return ReplayCsvUtility.AppendRawRows(itpData, rows, ExpectedCsvColumns, copyRows: false) > 0;
    }

    public void EndRealtimeCsv()
    {
        if (realtimePreviousMaxStep >= 0)
        {
            MaxStep = realtimePreviousMaxStep;
            realtimePreviousMaxStep = -1;
        }

        UseExternalReplayData = false;
        replayDataMode = ReplayDataMode.DatasetReplay;
    }

    public override void Initialize()
    {
        Time.fixedDeltaTime = 0.02f;
        DisableConflictingLegacyControllers();
        arts = GetComponentsInChildren<ArticulationBody>();
        if (arts == null || arts.Length == 0)
        {
            Debug.LogError("[OpenLoong] No ArticulationBody components found.");
            return;
        }

        body = arts[0].transform;
        rootArticulation = arts[0];
        pos0 = body.position;
        rot0 = body.rotation;

        ResolveCsvJointMap();
        CaptureRestPose();
        if (logJointMappingOnStart) DumpJointMapping();
        TryLoadCurrentMotionData(keepProgress: false);

        MimicAgentRegistry registry = MimicAgentRegistry.Instance;
        if (registry != null)
        {
            registry.Register(this);
        }
    }

    private void CaptureRestPose()
    {
        positionCache.Clear();
        velocityCache.Clear();
        rootArticulation.GetJointPositions(positionCache);
        rootArticulation.GetJointVelocities(velocityCache);
        restPositions = positionCache.ToArray();
        restVelocities = velocityCache.ToArray();
    }

    private void DisableConflictingLegacyControllers()
    {
        MonoBehaviour[] behaviours = GetComponents<MonoBehaviour>();
        for (int i = 0; i < behaviours.Length; i++)
        {
            MonoBehaviour behaviour = behaviours[i];
            if (behaviour == null || ReferenceEquals(behaviour, this))
            {
                continue;
            }

            string typeName = behaviour.GetType().Name;
            if (string.Equals(typeName, "RobotRLAgent", StringComparison.Ordinal) ||
                string.Equals(typeName, "LoongAgent", StringComparison.Ordinal))
            {
                if (behaviour.enabled)
                {
                    behaviour.enabled = false;
                    Debug.LogWarning($"[OpenLoong] Disabled conflicting legacy controller '{typeName}' on '{gameObject.name}'.");
                }
            }
        }
    }

    public override void OnEpisodeBegin()
    {
        if (rootArticulation == null) return;

        if (ShouldHoldNeutralPose())
        {
            ApplyNeutralPoseNow();
            return;
        }

        if (replay || useExternalReplayData)
        {
            rootArticulation.immovable = true;
            rootArticulation.velocity = Vector3.zero;
            rootArticulation.angularVelocity = Vector3.zero;
            ZeroAllArticulationVelocities();
            currentFrame = useExternalReplayData ? 0 : frame0;
            if (IsLiveRealtimeCsv)
            {
                realtimeFrameCursor = 0f;
            }
            tt = 0;

            if (replay && !useExternalReplayData)
            {
                TryLoadCurrentMotionData(keepProgress: false);
            }

            if (itpData != null && itpData.Count > 0)
            {
                ApplyFrame(Mathf.Clamp(currentFrame, 0, itpData.Count - 1), teleportRoot: true);
            }
            return;
        }

        rootArticulation.immovable = false;
        rootArticulation.TeleportRoot(pos0, rot0);
        rootArticulation.velocity = Vector3.zero;
        rootArticulation.angularVelocity = Vector3.zero;
        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        Array.Clear(u, 0, u.Length);
        Array.Clear(uff, 0, uff.Length);
        Array.Clear(utotal, 0, utotal.Length);
        currentFrame = useExternalReplayData ? 0 : frame0;
        if (IsLiveRealtimeCsv)
        {
            realtimeFrameCursor = 0f;
        }
        tt = 0;

        if (replay && !useExternalReplayData)
        {
            TryLoadCurrentMotionData(keepProgress: false);
        }

        if (itpData == null || itpData.Count == 0)
        {
            Debug.LogWarning("[OpenLoong] No replay data loaded.");
            return;
        }

        ApplyFrame(Mathf.Clamp(currentFrame, 0, itpData.Count - 1), teleportRoot: true);
        rootArticulation.immovable = true;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (!isRobotSelectedInScene)
        {
            return;
        }

        if (replay || useExternalReplayData)
        {
            return;
        }

        var continuousActions = actionBuffers.ContinuousActions;
        float actionGain = replay ? 0f : 50f;
        const float Smooth = 0.9f;

        for (int i = 0; i < DofCols; i++)
        {
            float action = i < continuousActions.Length ? continuousActions[i] : 0f;
            u[i] = u[i] * Smooth + (1f - Smooth) * action;
            utotal[i] = actionGain * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }

    private void FixedUpdate()
    {
        if (!isRobotSelectedInScene)
        {
            return;
        }

        if (ShouldHoldNeutralPose())
        {
            ApplyNeutralPoseNow();
            return;
        }

        if (rootArticulation == null || itpData == null || itpData.Count == 0)
        {
            return;
        }

        bool mirrorMode = replay || useExternalReplayData;
        if (IsLiveRealtimeCsv)
        {
            realtimeFrameCursor = Mathf.Clamp(realtimeFrameCursor, 0f, itpData.Count - 1);
            ApplyInterpolatedRealtimeFrame(realtimeFrameCursor, teleportRoot: mirrorMode);
            currentFrame = Mathf.Clamp(Mathf.FloorToInt(realtimeFrameCursor), 0, itpData.Count - 1);
        }
        else
        {
            ApplyFrame(Mathf.Clamp(currentFrame, 0, itpData.Count - 1), teleportRoot: mirrorMode);
        }

        tt++;
        if (mirrorMode)
        {
            rootArticulation.immovable = keepRootImmovableDuringReplay || mirrorMode;
            AdvancePlaybackFrame();
            return;
        }

        if (tt > 3)
        {
            rootArticulation.immovable = false;
        }

        AdvancePlaybackFrame();
    }

    private void AdvancePlaybackFrame()
    {
        if (currentFrame < itpData.Count - 1)
        {
            if (IsLiveRealtimeCsv)
            {
                realtimeFrameCursor = ReplayCsvUtility.AdvanceRealtimeCursor(
                    realtimeFrameCursor,
                    itpData.Count,
                    realtimePlaybackFps,
                    Time.fixedDeltaTime,
                    realtimePlaybackBufferSeconds);
                currentFrame = Mathf.Clamp(Mathf.FloorToInt(realtimeFrameCursor), 0, itpData.Count - 1);
            }
            else
            {
                currentFrame++;
            }
        }
    }

    private void ApplyInterpolatedRealtimeFrame(float frameCursor, bool teleportRoot)
    {
        if (!ReplayCsvUtility.SampleRowsAtFrame(itpData, frameCursor, ExpectedCsvColumns, realtimeSampledRow))
        {
            ApplyFrame(Mathf.Clamp(currentFrame, 0, itpData.Count - 1), teleportRoot);
            return;
        }

        ApplyRow(realtimeSampledRow, teleportRoot);
    }

    public bool LoadReplayCsvFromPath(string filePath, bool keepProgress)
    {
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
        {
            Debug.LogWarning("[OpenLoong] CSV path is missing: " + filePath);
            return false;
        }

        List<float[]> data = LoadDataFromFile(filePath);
        if (data.Count == 0)
        {
            Debug.LogWarning("[OpenLoong] CSV contains no valid 38-column rows: " + filePath);
            return false;
        }

        replayDataMode = ReplayDataMode.ExternalCsvReplay;
        useExternalReplayData = true;
        realtimeFrameCursor = 0f;
        return ApplyReplayData(data, keepProgress);
    }

    public void ResetToInitialState()
    {
        if (rootArticulation == null) return;

        UseExternalReplayData = false;
        ReplayMode = false;
        replayDataMode = ReplayDataMode.DatasetReplay;
        rootArticulation.immovable = false;
        rootArticulation.TeleportRoot(pos0, rot0);
        rootArticulation.velocity = Vector3.zero;
        rootArticulation.angularVelocity = Vector3.zero;
        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        for (int i = 0; i < DofCols; i++)
        {
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
            SetJointTargetDeg(jh[i], 0f);
        }

        currentFrame = frame0;
        tt = 0;

        // Clear stale replay data so FixedUpdate does not play old CSV frames
        // after the agent has been reset to neutral pose.
        itpData = new List<float[]>();
        refData = new List<float[]>();
    }

    private bool TryLoadCurrentMotionData(bool keepProgress)
    {
        string datasetPath = ResolveDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath)) return false;

        List<string> csvFiles = Directory.GetFiles(datasetPath, "*.csv", SearchOption.AllDirectories)
            .OrderBy(path => string.Equals(Path.GetFileName(path), "neutral_stand.csv", StringComparison.OrdinalIgnoreCase) ? 0 : 1)
            .ThenBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (csvFiles.Count == 0) return false;

        string selectedCsv = csvFiles[Mathf.Clamp(MotionId, 0, csvFiles.Count - 1)];
        return ApplyReplayData(LoadDataFromFile(selectedCsv), keepProgress);
    }

    private bool ApplyReplayData(List<float[]> data, bool keepProgress)
    {
        if (data == null || data.Count == 0) return false;

        int oldFrame = currentFrame;
        refData = data;
        itpData = ReplayCsvUtility.Resample30FpsToFixed50Hz(refData);
        if (itpData.Count == 0) return false;

        currentFrame = keepProgress
            ? Mathf.Clamp(oldFrame, 0, itpData.Count - 1)
            : Mathf.Clamp(frame0, 0, itpData.Count - 1);
        if (!keepProgress)
        {
            appliedRowDebugCount = 0;
        }
        realtimeFrameCursor = currentFrame;
        return true;
    }

    private List<float[]> LoadDataFromFile(string filePath)
    {
        var data = new List<float[]>();
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (StreamReader reader = new StreamReader(fs))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                string[] tokens = line.Split(',');
                if (tokens.Length < ExpectedCols) continue;

                float[] row = new float[ExpectedCols];
                bool ok = true;
                for (int i = 0; i < ExpectedCols; i++)
                {
                    string token = tokens[i].Trim();
                    if (!float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) &&
                        !float.TryParse(token, out value))
                    {
                        ok = false;
                        break;
                    }
                    row[i] = value;
                }

                if (ok) data.Add(row);
            }
        }

        return data;
    }

    private void ApplyFrame(int frameIndex, bool teleportRoot)
    {
        float[] row = itpData[frameIndex];
        ApplyRow(row, teleportRoot);
    }

    private void ApplyRow(float[] row, bool teleportRoot)
    {
        float[] pos = new float[RootPosCols];
        float[] rot = new float[RootRotCols];
        Array.Copy(row, 0, pos, 0, RootPosCols);
        Array.Copy(row, RootPosCols, rot, 0, RootRotCols);

        for (int i = 0; i < DofCols; i++)
        {
            uff[i] = GetCalibratedCsvJointRadians(row, i) * Mathf.Rad2Deg;
            SetJointTargetDeg(jh[i], uff[i], clampToDriveLimits: !(replay || useExternalReplayData));
        }

        LogAppliedRowDebug(row);

        newPosition = UnityQposMapper.MapRootPosition(pos, pos0, replayRootOffset);
        newRotation = UnityQposMapper.MapRootRotationFromCsvXyzw(rot);

        if (teleportRoot)
        {
            Physics.gravity = Vector3.zero;
            rootArticulation.TeleportRoot(newPosition, newRotation);
            rootArticulation.velocity = Vector3.zero;
            rootArticulation.angularVelocity = Vector3.zero;
            ZeroAllArticulationVelocities();
            rootArticulation.immovable = true;
        }

        if ((replay || useExternalReplayData) && writeReplayJointPositionsDirectly)
        {
            ApplyDirectJointStateFromCsvRow(row);
        }
    }

    private void ApplyDirectJointStateFromCsvRow(float[] row)
    {
        int wroteCount = 0;
        for (int i = 0; i < DofCols; i++)
        {
            ArticulationBody joint = jh[i];
            if (joint == null)
            {
                continue;
            }

            ArticulationReducedSpace jointPosition = joint.jointPosition;
            if (jointPosition.dofCount <= 0)
            {
                continue;
            }

            float targetRad = GetCalibratedCsvJointRadians(row, i);
            jointPosition[0] = targetRad;
            joint.jointPosition = jointPosition;
            LogRetargetCalibrationIfNeeded(row, i, targetRad);
            wroteCount++;

            if (zeroReplayJointVelocitiesOnDirectWrite)
            {
                ArticulationReducedSpace jointVelocity = joint.jointVelocity;
                if (jointVelocity.dofCount > 0)
                {
                    jointVelocity[0] = 0f;
                    joint.jointVelocity = jointVelocity;
                }
            }
        }

        if (wroteCount == 0 && !hasLoggedDirectJointStateError)
        {
            Debug.LogError("[OpenLoong] Direct qpos replay could not write any joint positions because all bound joints reported jointPosition.dofCount=0.");
            hasLoggedDirectJointStateError = true;
        }
    }

    private void ZeroAllArticulationVelocities()
    {
        if (arts == null)
        {
            return;
        }

        for (int i = 0; i < arts.Length; i++)
        {
            ArticulationBody art = arts[i];
            if (art == null)
            {
                continue;
            }

            art.velocity = Vector3.zero;
            art.angularVelocity = Vector3.zero;
            try
            {
                ArticulationReducedSpace jointVelocity = art.jointVelocity;
                if (jointVelocity.dofCount > 0)
                {
                    for (int dof = 0; dof < jointVelocity.dofCount; dof++)
                    {
                        jointVelocity[dof] = 0f;
                    }
                    art.jointVelocity = jointVelocity;
                }
            }
            catch (Exception)
            {
                // Unity can expose an empty reduced-space cache during articulation rebuild.
            }
        }
    }

    private float ClampJointRadiansToDrive(ArticulationBody joint, float radians)
    {
        if (!clampTargetsToDriveLimits || joint == null)
        {
            return radians;
        }

        ArticulationDrive drive = joint.xDrive;
        if (drive.lowerLimit >= drive.upperLimit)
        {
            return radians;
        }

        float degrees = Mathf.Clamp(radians * Mathf.Rad2Deg, drive.lowerLimit, drive.upperLimit);
        return degrees * Mathf.Deg2Rad;
    }

    private float GetCalibratedCsvJointRadians(float[] row, int dofIndex)
    {
        float value = row[RootPosCols + RootRotCols + dofIndex];
        if ((replay || useExternalReplayData) && neutralizeWristDofsInMirror && IsWristDof(dofIndex))
        {
            return wristNeutralRad;
        }

        string jointName = dofIndex >= 0 && dofIndex < CsvJointNames.Length ? CsvJointNames[dofIndex] : string.Empty;
        float defaultSign = dofIndex >= 0 && dofIndex < UnityDriveSigns.Length ? UnityDriveSigns[dofIndex] : 1f;
        float defaultOffset = dofIndex >= 0 && dofIndex < UnityDriveOffsetsRad.Length ? UnityDriveOffsetsRad[dofIndex] : 0f;
        UnityRetargetCalibrationEntry calibration = UnityRetargetCalibration.Resolve(
            dofIndex,
            jointName,
            UnityCalibration,
            defaultSign,
            defaultOffset,
            clampTargetsToDriveLimits);
        return applyMjcfAxisSignCorrection ? calibration.Apply(value) : value + calibration.offsetRad;
    }

    private static bool IsWristDof(int dofIndex)
    {
        return dofIndex == 6 || dofIndex == 7 || dofIndex == 8 ||
               dofIndex == 13 || dofIndex == 14 || dofIndex == 15;
    }

    private void LogRetargetCalibrationIfNeeded(float[] row, int dofIndex, float targetRad)
    {
        if (!logRetargetCalibrationDiagnostics)
        {
            return;
        }

        int interval = Mathf.Max(1, retargetCalibrationLogInterval);
        retargetCalibrationLogCounter++;
        if (retargetCalibrationLogCounter % interval != 0)
        {
            return;
        }

        UnityRetargetCalibrationEntry calibration = UnityRetargetCalibration.Resolve(
            dofIndex,
            dofIndex >= 0 && dofIndex < CsvJointNames.Length ? CsvJointNames[dofIndex] : string.Empty,
            UnityCalibration,
            dofIndex >= 0 && dofIndex < UnityDriveSigns.Length ? UnityDriveSigns[dofIndex] : 1f,
            dofIndex >= 0 && dofIndex < UnityDriveOffsetsRad.Length ? UnityDriveOffsetsRad[dofIndex] : 0f,
            clampTargetsToDriveLimits);
        ArticulationBody joint = dofIndex >= 0 && dofIndex < jh.Length ? jh[dofIndex] : null;
        float actualRad = 0f;
        if (joint != null && joint.jointPosition.dofCount > 0)
        {
            actualRad = joint.jointPosition[0];
        }

        float csvRad = row[RootPosCols + RootRotCols + dofIndex];
        Debug.Log(
            $"[OpenLoongCalib] frame={currentFrame} csvIndex={dofIndex} joint={calibration.jointName} " +
            $"csvRad={csvRad:F4} sign={calibration.sign:F1} offsetRad={calibration.offsetRad:F4} " +
            $"targetRad={targetRad:F4} actualRad={actualRad:F4} errorRad={(actualRad - targetRad):F4}");
    }

    private void LogAppliedRowDebug(float[] row)
    {
        if (appliedRowDebugCount >= 3)
        {
            return;
        }

        appliedRowDebugCount++;
        int boundCount = 0;
        int directJointStateCount = 0;
        int nonZeroDofCount = 0;
        float maxAbsDeg = 0f;
        string maxName = "<none>";
        for (int i = 0; i < DofCols; i++)
        {
            if (jh[i] != null)
            {
                boundCount++;
            }

            if (jh[i] != null && jh[i].jointPosition.dofCount > 0)
            {
                directJointStateCount++;
            }

            float absDeg = Mathf.Abs(uff[i]);
            if (Mathf.Abs(row[RootPosCols + RootRotCols + i]) > 1e-4f)
            {
                nonZeroDofCount++;
            }

            if (absDeg > maxAbsDeg)
            {
                maxAbsDeg = absDeg;
                maxName = CsvJointNames[i];
            }
        }

        Debug.Log(
            $"[OpenLoong] Applied CSV row: boundJoints={boundCount}/{DofCols}, directJointState={directJointStateCount}/{DofCols}, nonZeroDof={nonZeroDofCount}/{DofCols}, " +
            $"maxTarget={maxAbsDeg:F1}deg at {maxName}, " +
            $"J_arm_l_02 raw={row[RootPosCols + RootRotCols + 10] * Mathf.Rad2Deg:F1}deg applied={GetCalibratedCsvJointRadians(row, 10) * Mathf.Rad2Deg:F1}deg.");
    }

    private void ResolveCsvJointMap()
    {
        if (CsvJointNames.Length != DofCols || CsvBodyNames.Length != DofCols)
        {
            Debug.LogError($"[OpenLoong] Joint map length mismatch: joints={CsvJointNames.Length}, bodies={CsvBodyNames.Length}, expected={DofCols}.");
            return;
        }

        var byName = BuildArticulationAliasMap();
        var missing = new List<string>();
        for (int i = 0; i < CsvJointNames.Length; i++)
        {
            ArticulationBody joint = null;
            foreach (string alias in GetCsvJointAliases(i))
            {
                if (byName.TryGetValue(alias, out joint))
                {
                    break;
                }
            }

            jh[i] = joint;
            if (joint == null)
            {
                missing.Add($"{i}:{CsvJointNames[i]}/{CsvBodyNames[i]}");
                continue;
            }

            EnsureCsvJointIsMovable(i, joint);
        }

        if (missing.Count > 0)
        {
            Debug.LogError("[OpenLoong] Missing CSV joint bindings; DOF rows will not be applied to these joints: " + string.Join(", ", missing));
        }
        else
        {
            Debug.Log($"[OpenLoong] Bound {DofCols} CSV DOF joints by MuJoCo jointName/Unity link aliases.");
        }
    }

    private void DumpJointMapping()
    {
        var lines = new System.Text.StringBuilder();
        lines.AppendLine($"[OpenLoongMimicAgent:{name}] CSV joint mapping:");
        for (int i = 0; i < CsvJointNames.Length; i++)
        {
            string unityName = jh[i] != null ? jh[i].name : "<null>";
            string unityJointType = jh[i] != null ? jh[i].jointType.ToString() : "<null>";
            int reducedDof = jh[i] != null ? jh[i].jointPosition.dofCount : 0;
            float sign = applyMjcfAxisSignCorrection && i < UnityDriveSigns.Length ? UnityDriveSigns[i] : 1f;
            lines.AppendLine($"  CSV[{i,2}] {CsvJointNames[i]} ({CsvBodyNames[i]}) -> {unityName} [{unityJointType}], reducedDof={reducedDof}, sign={sign:+0;-0;+0}");
        }
        Debug.Log(lines.ToString());
    }

    private Dictionary<string, ArticulationBody> BuildArticulationAliasMap()
    {
        var map = new Dictionary<string, ArticulationBody>(StringComparer.OrdinalIgnoreCase);
        foreach (ArticulationBody art in arts)
        {
            if (art == null)
            {
                continue;
            }

            AddAlias(map, art.name, art);
            AddLinkJointAlias(map, art.name, art);
            foreach (string jointName in ReadUrdfJointNames(art))
            {
                AddAlias(map, jointName, art);
                AddLinkJointAlias(map, jointName, art);
            }
        }
        return map;
    }

    private static void EnsureCsvJointIsMovable(int index, ArticulationBody joint)
    {
        if (joint == null || joint.jointType != ArticulationJointType.FixedJoint)
        {
            return;
        }

        joint.jointType = ArticulationJointType.RevoluteJoint;
        Debug.LogWarning(
            $"[OpenLoong] Converted fixed Unity joint '{joint.name}' to RevoluteJoint for CSV[{index}] " +
            $"{CsvJointNames[index]} ({CsvBodyNames[index]}).");
    }

    private static IEnumerable<string> GetCsvJointAliases(int index)
    {
        yield return CsvJointNames[index];
        yield return CsvBodyNames[index];

        string jointSuffix = StripPrefix(CsvJointNames[index], "J_");
        string bodySuffix = StripPrefix(CsvBodyNames[index], "Link_");
        if (!string.IsNullOrWhiteSpace(jointSuffix))
        {
            yield return jointSuffix;
            yield return "Link_" + jointSuffix;
        }
        if (!string.IsNullOrWhiteSpace(bodySuffix))
        {
            yield return bodySuffix;
            yield return "J_" + bodySuffix;
        }
    }

    private static void AddLinkJointAlias(Dictionary<string, ArticulationBody> map, string name, ArticulationBody art)
    {
        string suffix = StripPrefix(name, "Link_");
        if (!string.IsNullOrWhiteSpace(suffix))
        {
            AddAlias(map, "J_" + suffix, art);
            AddAlias(map, suffix, art);
            return;
        }

        suffix = StripPrefix(name, "J_");
        if (!string.IsNullOrWhiteSpace(suffix))
        {
            AddAlias(map, "Link_" + suffix, art);
            AddAlias(map, suffix, art);
        }
    }

    private static void AddAlias(Dictionary<string, ArticulationBody> map, string name, ArticulationBody art)
    {
        if (string.IsNullOrWhiteSpace(name) || art == null)
        {
            return;
        }

        string key = name.Trim();
        if (!map.ContainsKey(key))
        {
            map.Add(key, art);
        }
    }

    private static string StripPrefix(string value, string prefix)
    {
        if (string.IsNullOrWhiteSpace(value) || !value.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }
        return value.Substring(prefix.Length);
    }

    private static IEnumerable<string> ReadUrdfJointNames(ArticulationBody art)
    {
        foreach (MonoBehaviour component in art.GetComponents<MonoBehaviour>())
        {
            if (component == null)
            {
                continue;
            }

            Type componentType = component.GetType();
            FieldInfo field = componentType.GetField("jointName", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (field != null && field.FieldType == typeof(string) && field.GetValue(component) is string fieldValue)
            {
                yield return fieldValue;
            }

            PropertyInfo property = componentType.GetProperty("jointName", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
            if (property != null && property.PropertyType == typeof(string) && property.GetIndexParameters().Length == 0 &&
                property.GetValue(component) is string propertyValue)
            {
                yield return propertyValue;
            }
        }
    }

    private string ResolveDatasetPath()
    {
        if (ImitationDatasetPaths.TryResolveRobotDatasetPath("openloong", datasetRelativePath, DefaultDatasetSearchPaths, out string resolved, out _))
        {
            return resolved;
        }

        return string.Empty;
    }
    private string ToAbsoluteProjectPath(string path)
    {
        string normalized = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(normalized)) return normalized;

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        return string.IsNullOrWhiteSpace(projectRoot)
            ? string.Empty
            : Path.GetFullPath(Path.Combine(projectRoot, normalized));
    }

    private void SafeSetJointPositions(List<float> positions)
    {
        if (rootArticulation == null) return;
        rootArticulation.SetJointPositions(AlignToCache(positions));
    }

    private void SafeSetJointVelocities(List<float> velocities)
    {
        if (rootArticulation == null) return;
        rootArticulation.SetJointVelocities(AlignToCache(velocities));
    }

    private List<float> AlignToCache(List<float> values)
    {
        positionCache.Clear();
        rootArticulation.GetJointPositions(positionCache);
        int cacheSize = positionCache.Count;
        if (values == null) values = new List<float>();
        if (values.Count == cacheSize) return values;

        var aligned = new List<float>(cacheSize);
        for (int i = 0; i < cacheSize; i++)
        {
            aligned.Add(i < values.Count ? values[i] : 0f);
        }
        return aligned;
    }

    private void SetJointTargetDeg(ArticulationBody joint, float value, bool clampToDriveLimits = true)
    {
        if (joint == null) return;

        ArticulationDrive drive = joint.xDrive;
        if (clampToDriveLimits && clampTargetsToDriveLimits && drive.lowerLimit < drive.upperLimit)
        {
            value = Mathf.Clamp(value, drive.lowerLimit, drive.upperLimit);
        }
        drive.stiffness = jointStiffness;
        drive.damping = jointDamping;
        if (replay || useExternalReplayData)
        {
            drive.forceLimit = Mathf.Max(drive.forceLimit, replayJointForceLimit);
        }
        drive.target = value;
        joint.xDrive = drive;
    }
}
