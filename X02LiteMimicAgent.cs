using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Linq;
using System;
using System.Globalization;
using Gewu.Imitation;

public class X02LiteMimicAgent : Agent, IMimicAgent, IRealtimeCsvMimicAgent, ISelectableMimicAgent
{
    private const int DofCount = 18;
    private const int CsvColumnCount = 7 + DofCount;

    public bool train = false;
    public bool replay = false;

    [Header("Multi-Robot Registry")]
    [Tooltip("Robot key used by the WHAM + GMR pipeline. Must match the name in StartInput.")]
    [SerializeField] private string robotKey = "x02lite";

    private static readonly string[] CsvJointNames =
    {
        "L_shoulder_pitch_Link",
        "L_shoulder_roll_Link",
        "L_shoulder_yaw_Link",
        "L_elbow_Link",
        "R_shoulder_pitch_Link",
        "R_shoulder_roll_Link",
        "R_shoulder_yaw_Link",
        "R_elbow_Link",
        "L_hip_yaw_Link",
        "L_hip_roll_Link",
        "L_hip_pitch_Link",
        "L_knee_Link",
        "L_ankle_pitch_Link",
        "R_hip_yaw_Link",
        "R_hip_roll_Link",
        "R_hip_pitch_Link",
        "R_knee_Link",
        "R_ankle_pitch_Link",
    };

    private static readonly string[][] CsvJointNameAliases =
    {
        new[] { "L_shoulder_pitch_Link", "Link_arm_l_01" },
        new[] { "L_shoulder_roll_Link", "Link_arm_l_02" },
        new[] { "L_shoulder_yaw_Link", "Link_arm_l_03" },
        new[] { "L_elbow_Link", "Link_arm_l_04" },
        new[] { "R_shoulder_pitch_Link", "Link_arm_r_01" },
        new[] { "R_shoulder_roll_Link", "Link_arm_r_02" },
        new[] { "R_shoulder_yaw_Link", "Link_arm_r_03" },
        new[] { "R_elbow_Link", "Link_arm_r_04" },
        new[] { "L_hip_yaw_Link", "Link_hip_l_yaw" },
        new[] { "L_hip_roll_Link", "Link_hip_l_roll" },
        new[] { "L_hip_pitch_Link", "Link_hip_l_pitch" },
        new[] { "L_knee_Link", "Link_knee_l_pitch" },
        new[] { "L_ankle_pitch_Link", "Link_ankle_l_pitch" },
        new[] { "R_hip_yaw_Link", "Link_hip_r_yaw" },
        new[] { "R_hip_roll_Link", "Link_hip_r_roll" },
        new[] { "R_hip_pitch_Link", "Link_hip_r_pitch" },
        new[] { "R_knee_Link", "Link_knee_r_pitch" },
        new[] { "R_ankle_pitch_Link", "Link_ankle_r_pitch" },
    };

    private static readonly string[] CsvJointDisplayNames =
    {
        "L_shoulder_pitch",
        "L_shoulder_roll",
        "L_shoulder_yaw",
        "L_elbow",
        "R_shoulder_pitch",
        "R_shoulder_roll",
        "R_shoulder_yaw",
        "R_elbow",
        "L_hip_yaw",
        "L_hip_roll",
        "L_hip_pitch",
        "L_knee_pitch",
        "L_ankle_pitch",
        "R_hip_yaw",
        "R_hip_roll",
        "R_hip_pitch",
        "R_knee_pitch",
        "R_ankle_pitch",
    };

    private static readonly float[] JointLowerLimitsDeg =
    {
        -90f, 0f, -130f, -145f,
        -90f, 0f, -130f, 0f,
        -45f, -15f, -30f, -140f, -80f,
        -45f, -15f, -30f, -140f, -80f,
    };

    private static readonly float[] JointUpperLimitsDeg =
    {
        180f, 180f, 130f, 0f,
        180f, 180f, 130f, 90f,
        45f, 45f, 115f, 5f, 60f,
        45f, 45f, 115f, 5f, 60f,
    };

    // CSV order stays MuJoCo/X02LiteV21. Unity's imported left elbow reduced-space
    // axis bends opposite to MuJoCo, so it uses a negative drive range below.
    private static readonly float[] DefaultUnityJointSigns =
    {
        1f, 1f, 1f, -1f,
        1f, 1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f,
    };

    // X02LiteV21 CSV/MuJoCo order:
    //   0..3   left  arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    //   4..7   right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
    //   8..12  left  leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch
    //   13..17 right leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch
    // Unity traversal may differ, so ResolveCsvJointMap maps by ArticulationBody name.

    [Tooltip("If ON, print every Unity revolute joint's GameObject name on Initialize().")]
    [SerializeField] private bool logJointMappingOnStart = true;
    [SerializeField] private bool requireCompleteRevoluteJointMap = true;
    [SerializeField] private bool writeReplayJointPositionsDirectly = true;
    [SerializeField] private bool writeLiveJointPositionsDirectly = false;
    [SerializeField] private bool zeroReplayJointVelocitiesOnDirectWrite = true;
    [SerializeField] private bool preferNeutralStandForDefaultReplay = true;
    [SerializeField] private string neutralStandCsvFileName = "neutral_stand.csv";
    [SerializeField] private float replayJointForceLimit = 300f;
    [SerializeField] private bool clampTargetsToDriveLimits = true;
    [SerializeField] private bool showGroundedNeutralPoseWhenSelected = true;
    [SerializeField] private bool applyUnityJointSignCorrection = true;
    [SerializeField] private bool logUnityJointSignsOnStart = true;

    private void DumpJointMapping()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"[X02LiteMimicAgent:{name}] Joint mapping:");
        for (int i = 0; i < DofCount && i < jh.Length; i++)
        {
            string jointName = (jh[i] != null) ? $"{jh[i].name} ({jh[i].jointType})" : "<missing>";
            sb.AppendLine($"  CSV[{i,2}] {CsvJointDisplayNames[i],-18} -> Unity '{jointName}'");
        }
        sb.AppendLine("Expected X02LiteV21 order (18 DOF):");
        sb.AppendLine("  0..3 left arm, 4..7 right arm, 8..12 left leg, 13..17 right leg");
        sb.AppendLine("Missing joints are skipped; they are never mapped by traversal fallback.");
        if (logUnityJointSignsOnStart)
        {
            sb.AppendLine("Unity joint signs:");
            for (int i = 0; i < DofCount; i++)
            {
                ArticulationDrive drive = jh[i] != null ? jh[i].xDrive : default(ArticulationDrive);
                string limits = jh[i] != null
                    ? $"limits=[{drive.lowerLimit:F1},{drive.upperLimit:F1}]"
                    : "limits=<missing>";
                sb.AppendLine($"  CSV[{i,2}] {CsvJointDisplayNames[i],-18} sign={GetUnityJointSign(i),4:F1} {limits}");
            }
        }
        UnityEngine.Debug.Log(sb.ToString());
    }

    float[] uff = new float[DofCount];
    float[] u = new float[DofCount];
    float[] utotal = new float[DofCount];

    public bool useExternalReplayData = false;

    [Tooltip("Primary dataset folder. If missing, the script tries the built-in search list.")]
    [SerializeField] private string datasetRelativePath = "Assets/Gewu/Imitation/dataset/x02lite";

    private static readonly string[] DefaultDatasetSearchPaths =
    {
        "Assets/Gewu/Imitation/dataset/x02lite",
        "Assets/Imitation/dataset/x02lite",
    };

    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();

    private int currentFrame;
    private int realtimePreviousMaxStep = -1;
    private float realtimeFrameCursor;
    private float realtimePlaybackFps = ReplayCsvUtility.SourceFps;
    private float realtimePlaybackBufferSeconds = 0.2f;
    private bool hasValidJointMap = true;
    private bool holdSelectionNeutralPose;
    private bool isLiveRealtimeCsv;
    private bool hasLoggedDirectJointStateError;
    private bool hasLoggedEmptyReplayData;
    private float[] neutralPoseFrame;

    // X02Lite CSV: 3 root pos + 4 root quat + 18 DOF = 25 cols
    float[] currentData = new float[CsvColumnCount];
    float[] realtimeSampledData = new float[CsvColumnCount];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[DofCount];

    Transform body;

    private float[] restPositions;
    private float[] restVelocities;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();

    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[DofCount];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;
    public int frame0 = 0;

    public float positionKp = 1000f;
    public float positionKd = 50f;
    public float rotationKp = 1000f;
    public float rotationKd = 50f;

    [Header("Root Height")]
    [Tooltip("MuJoCo X02LiteV21.xml qpos0 pelvis height. Used only for legacy CSV rows whose root z is zero.")]
    [SerializeField] private float nominalRootHeight = 0.962f;
    [Tooltip("If the scene stores the pelvis root at ground height, lift it to nominalRootHeight on Initialize/Reset.")]
    [SerializeField] private bool normalizeInitialRootHeight = true;
    [SerializeField] private bool replaceZeroCsvRootHeight = true;
    [SerializeField] private float zeroRootHeightEpsilon = 0.05f;

    private bool _isClone = false;
    private bool isRobotSelectedInScene = true;

    // IMimicAgent surface
    public string RobotKey => string.IsNullOrWhiteSpace(robotKey) ? "x02lite" : robotKey.Trim();
    public GameObject AgentGameObject => gameObject;
    public bool UseExternalReplayData
    {
        get => useExternalReplayData;
        set
        {
            useExternalReplayData = value;
            if (!value)
            {
                isLiveRealtimeCsv = false;
            }
        }
    }
    public bool ReplayMode { get => replay; set => replay = value; }
    public int MotionId { get; set; }
    public void RequestEndEpisode()
    {
        if (ReplayMode || UseExternalReplayData)
        {
            TryApplyReplayStartFrame(logIfEmpty: true);
            return;
        }

        EndEpisode();
    }
    public int ExpectedCsvColumns => CsvColumnCount;

    public void SetRobotSelectedInScene(bool isSelected)
    {
        isRobotSelectedInScene = isSelected;
        if (isSelected)
        {
            if (showGroundedNeutralPoseWhenSelected && !ReplayMode && !UseExternalReplayData)
            {
                UseExternalReplayData = false;
                ReplayMode = false;
                isLiveRealtimeCsv = false;
                holdSelectionNeutralPose = true;
                ClearReplayBuffers();
                ApplyGroundedNeutralPose();
            }
            else
            {
                holdSelectionNeutralPose = false;
            }
            return;
        }

        holdSelectionNeutralPose = false;
        UseExternalReplayData = false;
        ReplayMode = false;
        isLiveRealtimeCsv = false;
        ClearReplayBuffers();
        if (art0 != null)
        {
            art0.velocity = Vector3.zero;
            art0.angularVelocity = Vector3.zero;
            art0.immovable = true;
        }
    }

    public bool BeginRealtimeCsv()
    {
        if (!hasValidJointMap)
        {
            UnityEngine.Debug.LogError("[X02Lite] BeginRealtimeCsv aborted: incomplete or invalid revolute joint mapping.");
            return false;
        }

        if (realtimePreviousMaxStep < 0)
        {
            realtimePreviousMaxStep = MaxStep;
        }

        holdSelectionNeutralPose = false;
        MaxStep = 0;
        ClearReplayBuffers();
        currentFrame = 0;
        realtimeFrameCursor = 0f;
        tt = 0;
        isLiveRealtimeCsv = true;
        return true;
    }

    public void SetRealtimePlaybackRate(float framesPerSecond, float bufferSeconds)
    {
        realtimePlaybackFps = ReplayCsvUtility.ClampRealtimeFps(framesPerSecond);
        realtimePlaybackBufferSeconds = Mathf.Max(0f, bufferSeconds);
    }

    public bool AppendRealtimeCsvRows(IReadOnlyList<float[]> rows)
    {
        if (!hasValidJointMap)
        {
            return false;
        }

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
        isLiveRealtimeCsv = false;
    }

    public void ResetToInitialState()
    {
        isLiveRealtimeCsv = false;
        RestoreInitialRootPose();

        if (restPositions != null)
        {
            SafeSetJointPositions(new List<float>(restPositions));
        }
        if (restVelocities != null)
        {
            SafeSetJointVelocities(new List<float>(restVelocities));
        }

        for (int i = 0; i < DofCount; i++)
        {
            if (jh[i] != null) SetJointTargetDeg(jh[i], 0f);
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
        }
        currentFrame = useExternalReplayData ? 0 : frame0;
        if (useExternalReplayData && isLiveRealtimeCsv)
        {
            realtimeFrameCursor = 0f;
        }
        tt = 0;

        if (isRobotSelectedInScene && !ReplayMode && !UseExternalReplayData && showGroundedNeutralPoseWhenSelected)
        {
            holdSelectionNeutralPose = true;
            ClearReplayBuffers();
            ApplyGroundedNeutralPose();
        }
    }

    void Start()
    {
        Time.fixedDeltaTime = 0.02f;

        if (train && !_isClone)
        {
            for (int i = 1; i < 10; i++)
            {
                GameObject clone = Instantiate(gameObject);
                clone.transform.position = transform.position + new Vector3(i * 2f, 0, 0);
                clone.name = $"{name}_Clone_{i}";
                clone.GetComponent<X02LiteMimicAgent>()._isClone = true;
            }
        }
    }

    public override void Initialize()
    {
        arts = this.GetComponentsInChildren<ArticulationBody>();
        art0 = ResolveRootArticulation();
        body = art0 != null ? art0.transform : null;
        hasValidJointMap = art0 != null && ResolveCsvJointMap();

        if (art0 == null)
        {
            UnityEngine.Debug.LogError("[X02Lite] No ArticulationBody root found. Replay/live retargeting is disabled.");
        }

        if (restPositions == null && art0 != null && body != null)
        {
            pos0 = body.position;
            rot0 = body.rotation;
            NormalizeInitialRootPose();
            art0.TeleportRoot(pos0, rot0);
            art0.velocity = Vector3.zero;
            art0.angularVelocity = Vector3.zero;

            art0.GetJointPositions(P0);
            art0.GetJointVelocities(W0);

            restPositions  = P0.ToArray();
            restVelocities = W0.ToArray();
        }

        TryCacheNeutralPoseFrame();
        if (replay)
        {
            TryLoadCurrentMotionData(keepProgress: false);
        }
        else
        {
            ClearReplayBuffers();
        }

        if (logJointMappingOnStart) DumpJointMapping();

        if (!hasValidJointMap)
        {
            UnityEngine.Debug.LogError("[X02Lite] Incomplete joint map. Replay/live retargeting will stay disabled until every expected arm/leg joint is authored as a RevoluteJoint.");
        }

        MimicAgentRegistry.Instance.Register(this);
    }

    private ArticulationBody ResolveRootArticulation()
    {
        if (arts == null || arts.Length == 0)
        {
            return null;
        }

        ArticulationBody root = arts.FirstOrDefault(art => art != null && art.isRoot);
        if (root != null)
        {
            return root;
        }

        root = arts.FirstOrDefault(art => art != null && string.Equals(art.name, "pelvis", System.StringComparison.OrdinalIgnoreCase));
        if (root != null)
        {
            UnityEngine.Debug.LogWarning("[X02Lite] No articulation isRoot flag found; using pelvis as root.");
            return root;
        }

        root = arts.FirstOrDefault(art => art != null);
        if (root != null)
        {
            UnityEngine.Debug.LogWarning($"[X02Lite] No articulation root/pelvis found; falling back to first ArticulationBody '{root.name}'.");
        }
        return root;
    }

    private bool ResolveCsvJointMap()
    {
        var byName = arts
            .Where(art => art != null)
            .GroupBy(art => art.name, System.StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.First(), System.StringComparer.OrdinalIgnoreCase);

        bool mappingValid = true;

        for (int i = 0; i < CsvJointNames.Length; i++)
        {
            ArticulationBody joint = ResolveCsvJointByAliases(i, byName);
            if (joint != null)
            {
                bool jointValid = EnsureCsvJointIsDriveable(i, joint);
                jh[i] = jointValid ? joint : null;
                mappingValid &= jointValid;
                continue;
            }

            jh[i] = null;
            mappingValid = false;
            UnityEngine.Debug.LogError($"[X02Lite] Missing expected joint for CSV[{i}] {CsvJointDisplayNames[i]}. Tried: {DescribeJointAliases(i)}.");
        }

        return !requireCompleteRevoluteJointMap || mappingValid;
    }

    private ArticulationBody ResolveCsvJointByAliases(int csvIndex, Dictionary<string, ArticulationBody> byName)
    {
        string[] aliases = CsvJointNameAliases[csvIndex];
        for (int i = 0; i < aliases.Length; i++)
        {
            if (byName.TryGetValue(aliases[i], out ArticulationBody joint))
            {
                return joint;
            }
        }
        return null;
    }

    private static string DescribeJointAliases(int csvIndex)
    {
        return string.Join(", ", CsvJointNameAliases[csvIndex]);
    }

    private bool EnsureCsvJointIsDriveable(int csvIndex, ArticulationBody joint)
    {
        if (joint == null)
        {
            return false;
        }

        if (joint.jointType != ArticulationJointType.RevoluteJoint)
        {
            UnityEngine.Debug.LogError($"[X02Lite] Joint '{joint.name}' is {joint.jointType}, not RevoluteJoint. CSV[{csvIndex}] {CsvJointDisplayNames[csvIndex]} is invalid.");
            return false;
        }

        return true;
    }

    List<string> GetCsvFileNames(string directoryPath)
    {
        List<string> csvFiles = new List<string>();
        try
        {
            if (Directory.Exists(directoryPath))
            {
                foreach (string file in Directory.GetFiles(directoryPath, "*.csv", SearchOption.AllDirectories)
                    .OrderBy(file => file, StringComparer.OrdinalIgnoreCase))
                {
                    if (Path.GetExtension(file).ToLower() == ".csv")
                        csvFiles.Add(file);
                }
            }
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError("Error accessing directory: " + e.Message);
        }
        return csvFiles;
    }

    List<float[]> LoadDataFromFile(string filePath)
    {
        List<float[]> dataList = new List<float[]>();
        bool loggedColumnMismatch = false;
        try
        {
            using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (StreamReader reader = new StreamReader(fileStream))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    string[] values = line.Split(',');
                    List<float> frameData = new List<float>();
                    foreach (string value in values)
                    {
                        string trimmed = value.Trim();
                        if (float.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) ||
                            float.TryParse(trimmed, out v))
                        {
                            frameData.Add(v);
                        }
                    }

                    if (frameData.Count != CsvColumnCount)
                    {
                        if (!loggedColumnMismatch)
                        {
                            UnityEngine.Debug.LogWarning(
                                $"[X02Lite] Skipping CSV row with {frameData.Count} columns in '{filePath}'. Expected exactly {CsvColumnCount} columns.");
                            loggedColumnMismatch = true;
                        }
                        continue;
                    }

                    dataList.Add(frameData.ToArray());
                }
            }
        }
        catch (System.Exception e)
        {
            print("Error loading data from file " + filePath + ": " + e.Message);
        }
        return dataList;
    }

    public bool LoadReplayCsvFromPath(string filePath, bool keepProgress = true)
    {
        if (!hasValidJointMap)
        {
            UnityEngine.Debug.LogError("[X02Lite] LoadReplayCsvFromPath aborted: incomplete or invalid revolute joint mapping.");
            return false;
        }

        if (string.IsNullOrWhiteSpace(filePath))
        {
            UnityEngine.Debug.LogWarning("[X02Lite] LoadReplayCsvFromPath failed: filePath is empty.");
            return false;
        }
        if (!File.Exists(filePath))
        {
            UnityEngine.Debug.LogWarning("[X02Lite] LoadReplayCsvFromPath failed: file does not exist: " + filePath);
            return false;
        }

        List<float[]> data = LoadDataFromFile(filePath);
        if (data == null || data.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[X02Lite] No valid data in " + filePath);
            return false;
        }

        holdSelectionNeutralPose = false;
        isLiveRealtimeCsv = false;
        ReplayMode = true;
        UnityEngine.Debug.Log($"[X02Lite] Loaded replay CSV '{filePath}' rows={data.Count}.");
        return ApplyReplayData(data, keepProgress);
    }

    private bool TryLoadCurrentMotionData(bool keepProgress)
    {
        string datasetPath = ResolveDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath)) return false;

        List<string> csvFileNames = GetCsvFileNames(datasetPath);
        if (csvFileNames.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[X02Lite] No csv files found in dataset path: " + datasetPath);
            return false;
        }

        int mid = Mathf.Clamp(MotionId, 0, csvFileNames.Count - 1);
        string selectedCsv = PickDefaultReplayCsv(csvFileNames, mid);

        List<float[]> data = LoadDataFromFile(selectedCsv);
        if (data == null || data.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[X02Lite] CSV contains no valid frame data: " + selectedCsv);
            return false;
        }

        isLiveRealtimeCsv = false;
        return ApplyReplayData(data, keepProgress);
    }

    private string PickDefaultReplayCsv(List<string> csvFileNames, int fallbackIndex)
    {
        if (csvFileNames == null || csvFileNames.Count == 0)
        {
            return string.Empty;
        }

        if (preferNeutralStandForDefaultReplay)
        {
            string neutralName = string.IsNullOrWhiteSpace(neutralStandCsvFileName)
                ? "neutral_stand.csv"
                : neutralStandCsvFileName.Trim();

            string neutral = csvFileNames.FirstOrDefault(file =>
                string.Equals(Path.GetFileName(file), neutralName, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(Path.GetFileNameWithoutExtension(file), Path.GetFileNameWithoutExtension(neutralName), StringComparison.OrdinalIgnoreCase));

            if (!string.IsNullOrWhiteSpace(neutral))
            {
                return neutral;
            }
        }

        int index = Mathf.Clamp(fallbackIndex, 0, csvFileNames.Count - 1);
        return csvFileNames[index];
    }

    private bool ApplyReplayData(List<float[]> data, bool keepProgress)
    {
        if (data == null || data.Count == 0) return false;

        int oldFrame = currentFrame;
        isLiveRealtimeCsv = false;
        refData = data;

        if (refData.Count <= 1)
        {
            itpData = new List<float[]>(refData);
        }
        else
        {
            float[] refT = new float[refData.Count];
            for (int i = 0; i < refT.Length; i++) refT[i] = i / 30f;

            int newFrameCount = Mathf.Max(1, (int)(refData.Count * 50f / 30f) - 1);
            float[] newT = new float[newFrameCount];
            for (int i = 0; i < newT.Length; i++) newT[i] = i / 50f;

            List<float[]> interpolated = Interpolate(refT, refData, newT);
            itpData = interpolated ?? new List<float[]>(refData);
        }

        if (itpData.Count == 0) return false;

        holdSelectionNeutralPose = false;
        hasLoggedDirectJointStateError = false;
        currentFrame = keepProgress
            ? Mathf.Clamp(oldFrame, 0, itpData.Count - 1)
            : 0;

        return true;
    }

    private string ResolveDatasetPath()
    {
        if (ImitationDatasetPaths.TryResolveRobotDatasetPath("x02lite", datasetRelativePath, DefaultDatasetSearchPaths, out string resolved, out _))
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
        if (string.IsNullOrWhiteSpace(projectRoot)) return string.Empty;

        return Path.GetFullPath(Path.Combine(projectRoot, normalized));
    }

    // Safe articulation helpers

    private readonly List<float> _cacheProbe = new List<float>();

    private int ArticulationCacheSize()
    {
        if (art0 == null) return 0;
        _cacheProbe.Clear();
        art0.GetJointPositions(_cacheProbe);
        return _cacheProbe.Count;
    }

    private void SafeSetJointPositions(List<float> positions)
    {
        if (art0 == null) return;
        int cacheSize = ArticulationCacheSize();
        List<float> safe = AlignToCache(positions, cacheSize);
        art0.SetJointPositions(safe);
    }

    private void SafeSetJointVelocities(List<float> velocities)
    {
        if (art0 == null) return;
        int cacheSize = ArticulationCacheSize();
        List<float> safe = AlignToCache(velocities, cacheSize);
        art0.SetJointVelocities(safe);
    }

    private List<float> AlignToCache(List<float> source, int cacheSize)
    {
        if (source == null) source = new List<float>();
        if (source.Count == cacheSize) return source;

        const int RootDofCount = 6;

        if (source.Count == cacheSize + RootDofCount)
            return source.GetRange(RootDofCount, cacheSize);

        if (source.Count + RootDofCount == cacheSize)
        {
            _cacheProbe.Clear();
            art0.GetJointPositions(_cacheProbe);
            var safe = new List<float>(cacheSize);
            for (int i = 0; i < RootDofCount && i < _cacheProbe.Count; i++)
                safe.Add(_cacheProbe[i]);
            while (safe.Count < RootDofCount) safe.Add(0f);
            safe.AddRange(source);
            return safe;
        }

        UnityEngine.Debug.LogWarning(
            $"[X02Lite] Unexpected cache mismatch: source={source.Count}, cache={cacheSize}.");
        var result = new List<float>(cacheSize);
        for (int i = 0; i < cacheSize; i++)
            result.Add(i < source.Count ? source[i] : 0f);
        return result;
    }

    // Episode lifecycle

    public override void OnEpisodeBegin()
    {
        if (!isRobotSelectedInScene)
        {
            return;
        }

        if (!hasValidJointMap)
        {
            return;
        }

        if (holdSelectionNeutralPose && !ReplayMode && !useExternalReplayData)
        {
            ApplyGroundedNeutralPose();
            return;
        }

        TryApplyReplayStartFrame(logIfEmpty: true);
    }

    private bool TryApplyReplayStartFrame(bool logIfEmpty)
    {
        if (!isRobotSelectedInScene || !hasValidJointMap || art0 == null)
        {
            return false;
        }

        art0.immovable       = false;
        RestoreInitialRootPose();

        if (restPositions != null)
        {
            SafeSetJointPositions(new List<float>(restPositions));
        }
        if (restVelocities != null)
        {
            SafeSetJointVelocities(new List<float>(restVelocities));
        }

        for (int i = 0; i < DofCount; i++) u[i]   = 0;
        for (int i = 0; i < DofCount; i++) uff[i] = 0;
        currentFrame = useExternalReplayData ? 0 : frame0;
        if (useExternalReplayData)
        {
            realtimeFrameCursor = 0f;
        }

        if (replay && !useExternalReplayData)
        {
            TryLoadCurrentMotionData(keepProgress: false);
        }

        List<float[]> episodeData = useExternalReplayData ? itpData : refData;
        if (episodeData == null || episodeData.Count == 0)
        {
            if (logIfEmpty && !hasLoggedEmptyReplayData)
            {
                UnityEngine.Debug.LogWarning("[X02Lite] replay data is empty, skip replay start.");
                hasLoggedEmptyReplayData = true;
            }
            return false;
        }

        hasLoggedEmptyReplayData = false;
        currentFrame = Mathf.Clamp(currentFrame, 0, episodeData.Count - 1);
        tt = 0;
        currentData = episodeData[currentFrame];
        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, DofCount);

        ApplyCurrentDofToJoints(directWrite: replay && (!isLiveRealtimeCsv || writeLiveJointPositionsDirectly));

        Vector3 newPosition = BuildUnityRootPosition(currentPos);
        Quaternion newRotation = new Quaternion(
            -currentRot[1],
             currentRot[2],
             currentRot[0],
            -currentRot[3]
        );
        art0.TeleportRoot(newPosition, newRotation);
        art0.velocity        = Vector3.zero;
        art0.angularVelocity = Vector3.zero;
        art0.immovable       = true;
        return true;
    }

    List<float[]> Interpolate(float[] t, List<float[]> posList, float[] targetT)
    {
        if (t.Length != posList.Count) { UnityEngine.Debug.LogError("t and posList must have the same length"); return null; }
        int dimension = posList[0].Length;
        foreach (float[] arr in posList)
        {
            if (arr.Length != dimension) { UnityEngine.Debug.LogError("All arrays in posList must have the same length"); return null; }
        }
        List<float[]> result = new List<float[]>();
        for (int i = 0; i < targetT.Length; i++)
        {
            float tValue = targetT[i];
            if (tValue < t[0] || tValue > t[t.Length - 1]) return null;
            int index = 0;
            while (index < t.Length - 1 && t[index + 1] < tValue) index++;
            float ratio = (tValue - t[index]) / (t[index + 1] - t[index]);
            float[] interpolatedPos = new float[dimension];
            for (int j = 0; j < dimension; j++)
                interpolatedPos[j] = Mathf.Lerp(posList[index][j], posList[index + 1][j], ratio);
            result.Add(interpolatedPos);
        }
        return result;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        if (body != null) sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * 3.14f / 180f);
        else              sensor.AddObservation(0f);
        if (body != null) sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * 3.14f / 180f);
        else              sensor.AddObservation(0f);
        if (body != null && art0 != null)
            sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        else
            sensor.AddObservation(Vector3.zero);

        for (int i = 0; i < DofCount; i++)
        {
            float pos = 0f, vel = 0f;
            if (jh[i] != null)
            {
                var jp = jh[i].jointPosition;
                var jv = jh[i].jointVelocity;
                if (jp.dofCount > 0) pos = jp[0];
                if (jv.dofCount > 0) vel = jv[0];
            }
            sensor.AddObservation(pos);
            sensor.AddObservation(vel);
        }
    }

    float EulerTrans(float eulerAngle)
    {
        if (eulerAngle <= 180) return eulerAngle;
        return eulerAngle - 360f;
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (!isRobotSelectedInScene)
        {
            return;
        }

        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        float kb = 50;
        if (replay) kb = 0;
        for (int i = 0; i < DofCount; i++)
        {
            float action = i < continuousActions.Length ? continuousActions[i] : 0f;
            u[i] = u[i] * kk + (1 - kk) * action;
            utotal[i] = kb * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }

    void FixedUpdate()
    {
        if (!isRobotSelectedInScene)
        {
            return;
        }

        if (!hasValidJointMap)
        {
            return;
        }

        if (holdSelectionNeutralPose && !ReplayMode && !useExternalReplayData)
        {
            return;
        }

        bool hasMotionData = itpData != null && itpData.Count > 0;
        if (hasMotionData)
        {
            if (useExternalReplayData && isLiveRealtimeCsv)
            {
                realtimeFrameCursor = Mathf.Clamp(realtimeFrameCursor, 0f, itpData.Count - 1);
                if (ReplayCsvUtility.SampleRowsAtFrame(itpData, realtimeFrameCursor, ExpectedCsvColumns, realtimeSampledData))
                {
                    currentData = realtimeSampledData;
                    currentFrame = Mathf.Clamp(Mathf.FloorToInt(realtimeFrameCursor), 0, itpData.Count - 1);
                }
                else
                {
                    currentFrame = Mathf.Clamp(currentFrame, 0, itpData.Count - 1);
                    currentData = itpData[currentFrame];
                }
            }
            else
            {
                currentFrame = Mathf.Clamp(currentFrame, 0, itpData.Count - 1);
                currentData = itpData[currentFrame];
            }
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, DofCount);
            ApplyCurrentDofToJoints(directWrite: replay && (!isLiveRealtimeCsv || writeLiveJointPositionsDirectly));

            newPosition = BuildUnityRootPosition(currentPos);
            newRotation = new Quaternion(
                -currentRot[1],
                 currentRot[2],
                 currentRot[0],
                -currentRot[3]
            );
            if (replay)
            {
                Physics.gravity = Vector3.zero;
                art0.immovable = true;
                art0.TeleportRoot(newPosition, newRotation);
            }
            else
            {
                if (tt > 3)
                {
                    art0.immovable = false;

                    Vector3 positionError = newPosition - body.position;
                    Vector3 velocityError = -art0.velocity;
                    Vector3 positionForce = positionKp * positionError + positionKd * velocityError;
                    art0.AddForce(positionForce);

                    Quaternion rotationError = newRotation * Quaternion.Inverse(body.rotation);
                    rotationError.ToAngleAxis(out float angle, out Vector3 axis);
                    if (angle > 180f) angle -= 360f;
                    Vector3 rotationErrorVector = (angle * Mathf.Deg2Rad) * axis.normalized;
                    Vector3 angularVelocityError = -art0.angularVelocity;
                    Vector3 rotationTorque = rotationKp * rotationErrorVector + rotationKd * angularVelocityError;
                    art0.AddTorque(0 * rotationTorque);
                }
            }
        }

        tt++;
        if (train && hasMotionData && tt > 3 && !replay)
        {
            art0.immovable = false;
            float rot_reward = -0.01f * Quaternion.Angle(body.rotation, newRotation);
            float pos_reward = -1f * (body.position - newPosition).magnitude;

            if (!replay && (Quaternion.Angle(body.rotation, newRotation) > 40f || (body.position - newPosition).magnitude > 0.5f))
            {
                EndEpisode();
            }

            float dof_reward = 0f;
            for (int i = 0; i < DofCount; i++)
            {
                if (jh[i] == null || jh[i].jointPosition.dofCount == 0)
                {
                    continue;
                }

                dof_reward += -0.1f * Mathf.Abs(jh[i].jointPosition[0] - ToUnityJointRadians(i, currentDof[i]));
            }

            AddReward(1f + (rot_reward + pos_reward) * 1f + dof_reward);
        }

        if (itpData != null && currentFrame < itpData.Count - 1)
        {
            if (useExternalReplayData && isLiveRealtimeCsv)
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
                currentFrame = currentFrame + 1;
            }
        }
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        if (joint == null) return;
        var drive = joint.xDrive;
        if (clampTargetsToDriveLimits)
        {
            x = Mathf.Clamp(x, drive.lowerLimit, drive.upperLimit);
        }
        drive.stiffness = 180f;
        drive.damping = 8f;
        drive.forceLimit = Mathf.Max(drive.forceLimit, replayJointForceLimit);
        drive.target = x;
        joint.xDrive = drive;
    }

    private void ApplyCurrentDofToJoints(bool directWrite)
    {
        int directWriteCount = 0;
        for (int i = 0; i < DofCount; i++)
        {
            float targetRad = ToUnityJointRadians(i, currentDof[i]);
            float targetDeg = targetRad * Mathf.Rad2Deg;
            uff[i] = targetDeg;
            SetJointTargetDeg(jh[i], targetDeg);

            if (directWrite && writeReplayJointPositionsDirectly)
            {
                if (SetJointPositionRad(jh[i], targetRad))
                {
                    directWriteCount++;
                }
            }
        }

        if (directWrite && writeReplayJointPositionsDirectly && directWriteCount == 0 && !hasLoggedDirectJointStateError)
        {
            UnityEngine.Debug.LogError("[X02Lite] Direct replay could not write any joint positions; all mapped joints reported empty reduced-space state.");
            hasLoggedDirectJointStateError = true;
        }
    }

    private float ToUnityJointRadians(int csvIndex, float csvRadians)
    {
        return csvRadians * GetUnityJointSign(csvIndex);
    }

    private float GetUnityJointSign(int csvIndex)
    {
        if (!applyUnityJointSignCorrection || csvIndex < 0 || csvIndex >= DefaultUnityJointSigns.Length)
        {
            return 1f;
        }

        return DefaultUnityJointSigns[csvIndex] < 0f ? -1f : 1f;
    }

    private bool SetJointPositionRad(ArticulationBody joint, float radians)
    {
        if (joint == null || joint.jointType != ArticulationJointType.RevoluteJoint)
        {
            return false;
        }

        try
        {
            ArticulationReducedSpace jointPosition = joint.jointPosition;
            if (jointPosition.dofCount <= 0)
            {
                return false;
            }

            jointPosition[0] = ClampJointRadiansToDrive(joint, radians);
            joint.jointPosition = jointPosition;

            if (zeroReplayJointVelocitiesOnDirectWrite)
            {
                ArticulationReducedSpace jointVelocity = joint.jointVelocity;
                if (jointVelocity.dofCount > 0)
                {
                    jointVelocity[0] = 0f;
                    joint.jointVelocity = jointVelocity;
                }
            }
            return true;
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogWarning($"[X02Lite] Failed to write joint position for '{joint.name}': {e.Message}");
            return false;
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

    private Vector3 BuildUnityRootPosition(float[] rootPos)
    {
        float y = rootPos[2];
        if (replaceZeroCsvRootHeight && Mathf.Abs(y) <= zeroRootHeightEpsilon)
        {
            y = nominalRootHeight;
        }

        Vector3 position = new Vector3(-rootPos[1], y, rootPos[0]);
        position.x += pos0.x;
        position.z += pos0.z;
        return position;
    }

    private void NormalizeInitialRootPose()
    {
        if (normalizeInitialRootHeight && Mathf.Abs(pos0.y) <= zeroRootHeightEpsilon)
        {
            pos0.y = nominalRootHeight;
        }
    }

    private void ClearReplayBuffers()
    {
        refData = new List<float[]>();
        itpData = new List<float[]>();
        currentFrame = 0;
        realtimeFrameCursor = 0f;
    }

    private bool TryCacheNeutralPoseFrame()
    {
        string datasetPath = ResolveDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath))
        {
            return false;
        }

        List<string> csvFileNames = GetCsvFileNames(datasetPath);
        string neutralPath = FindNeutralCsv(csvFileNames);
        if (string.IsNullOrWhiteSpace(neutralPath))
        {
            return false;
        }

        List<float[]> data = LoadDataFromFile(neutralPath);
        if (data == null || data.Count == 0)
        {
            return false;
        }

        neutralPoseFrame = data[0];
        return neutralPoseFrame != null && neutralPoseFrame.Length >= CsvColumnCount;
    }

    private string FindNeutralCsv(List<string> csvFileNames)
    {
        if (csvFileNames == null || csvFileNames.Count == 0)
        {
            return string.Empty;
        }

        string neutralName = string.IsNullOrWhiteSpace(neutralStandCsvFileName)
            ? "neutral_stand.csv"
            : neutralStandCsvFileName.Trim();

        return csvFileNames.FirstOrDefault(file =>
            string.Equals(Path.GetFileName(file), neutralName, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(Path.GetFileNameWithoutExtension(file), Path.GetFileNameWithoutExtension(neutralName), StringComparison.OrdinalIgnoreCase))
            ?? string.Empty;
    }

    private void ApplyGroundedNeutralPose()
    {
        if (arts == null || arts.Length == 0 || art0 == null)
        {
            return;
        }

        if ((neutralPoseFrame == null || neutralPoseFrame.Length < CsvColumnCount) && !TryCacheNeutralPoseFrame())
        {
            UnityEngine.Debug.LogWarning("[X02Lite] neutral_stand.csv is unavailable, falling back to zero joint targets for selection pose.");
        }

        RestoreInitialRootPose();
        art0.immovable = true;
        art0.velocity = Vector3.zero;
        art0.angularVelocity = Vector3.zero;

        for (int i = 0; i < DofCount; i++)
        {
            float targetRad = 0f;
            if (neutralPoseFrame != null && neutralPoseFrame.Length >= CsvColumnCount)
            {
                targetRad = neutralPoseFrame[7 + i];
            }

            currentDof[i] = targetRad;
            float unityTargetRad = ToUnityJointRadians(i, targetRad);
            uff[i] = unityTargetRad * Mathf.Rad2Deg;
            SetJointTargetDeg(jh[i], uff[i]);
            if (writeReplayJointPositionsDirectly)
            {
                SetJointPositionRad(jh[i], unityTargetRad);
            }
        }

        tt = 0;
    }

    private void RestoreInitialRootPose()
    {
        if (art0 == null) return;

        NormalizeInitialRootPose();
        art0.TeleportRoot(pos0, rot0);
        art0.velocity = Vector3.zero;
        art0.angularVelocity = Vector3.zero;
    }

    public override void Heuristic(in ActionBuffers actionsOut) { }
}
