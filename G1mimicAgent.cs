using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgentsExamples;
using Unity.MLAgents.Sensors;
using Random = UnityEngine.Random;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Linq;
using System;
using System.Globalization;
using System.Reflection;
using Gewu.Imitation;

public class G1mimicAgent : Agent, IMimicAgent, IRealtimeCsvMimicAgent, ISelectableMimicAgent
{
    public bool train = false;
    public bool replay = false;

    [Header("Multi-Robot Registry")]
    [Tooltip("Robot key used by the WHAM + GMR pipeline. Must match one of the supported names " +
             "in StartInput (e.g. unitree_g1, unitree_g1_with_hands).")]
    [SerializeField] private string robotKey = "unitree_g1";

    // ── Why no CSV→Unity permutation table here ──────────────────────────────
    // The current G1 prefab's ArticulationBody depth-first traversal yields
    // joints in the same order that WHAM + GMR writes to the CSV — URDF order:
    //   0..5   left  leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    //   6..11  right leg: same suborder
    //   12..14 waist:  yaw, roll, pitch
    //   15..21 left  arm: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
    //   22..28 right arm: same suborder
    //
    // So a 1:1 identity mapping (`currentDof[i] → jh[i]`) is correct. An
    // earlier prefab needed a non-trivial permutation; copying that table onto
    // this prefab is what was causing the "twitching" (every joint received a
    // random other joint's angle). The field is deliberately omitted so Unity
    // can't deserialize a stale wrong permutation from the scene.

    [Tooltip("If ON, print every Unity revolute joint's GameObject name on Initialize() so " +
             "the user can verify the G1 prefab still yields URDF order (left_hip_pitch → " +
             "right_hip_pitch → waist → left_arm → right_arm). If the order ever changes " +
             "(swapped left/right traversal, wrist after shoulder, etc.), retargeting will " +
             "look wrong even though no other code changed.")]
    [SerializeField] private bool logJointMappingOnStart = true;
    [SerializeField] private bool logReplayMirrorDiagnostics = false;
    [SerializeField] private int replayMirrorDiagnosticIntervalFrames = 120;
    [Tooltip("G1-only Unity articulation axis signs. Keep CSV order unchanged; adjust only entries proven by mirror diagnostics.")]
    [SerializeField] private float[] unityJointSigns = new float[29]
    {
        1f, 1f, 1f, 1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f, 1f,
        1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f, 1f, 1f,
        1f, 1f, 1f, 1f, 1f, 1f, 1f
    };

    private static readonly string[][] G1CsvJointAliases =
    {
        new[] { "left_hip_pitch_joint", "left_hip_pitch_link" },
        new[] { "left_hip_roll_joint", "left_hip_roll_link" },
        new[] { "left_hip_yaw_joint", "left_hip_yaw_link" },
        new[] { "left_knee_joint", "left_knee_link" },
        new[] { "left_ankle_pitch_joint", "left_ankle_pitch_link", "left_ankle_joint", "left_ankle_link" },
        new[] { "left_ankle_roll_joint", "left_ankle_roll_link" },

        new[] { "right_hip_pitch_joint", "right_hip_pitch_link" },
        new[] { "right_hip_roll_joint", "right_hip_roll_link" },
        new[] { "right_hip_yaw_joint", "right_hip_yaw_link" },
        new[] { "right_knee_joint", "right_knee_link" },
        new[] { "right_ankle_pitch_joint", "right_ankle_pitch_link", "right_ankle_joint", "right_ankle_link" },
        new[] { "right_ankle_roll_joint", "right_ankle_roll_link" },

        new[] { "waist_yaw_joint", "waist_yaw_link" },
        new[] { "waist_roll_joint", "waist_roll_link" },
        new[] { "waist_pitch_joint", "waist_pitch_link", "torso_link" },

        new[] { "left_shoulder_pitch_joint", "left_shoulder_pitch_link" },
        new[] { "left_shoulder_roll_joint", "left_shoulder_roll_link" },
        new[] { "left_shoulder_yaw_joint", "left_shoulder_yaw_link" },
        new[] { "left_elbow_joint", "left_elbow_link" },
        new[] { "left_wrist_roll_joint", "left_wrist_roll_link" },
        new[] { "left_wrist_pitch_joint", "left_wrist_pitch_link" },
        new[] { "left_wrist_yaw_joint", "left_wrist_yaw_link" },

        new[] { "right_shoulder_pitch_joint", "right_shoulder_pitch_link" },
        new[] { "right_shoulder_roll_joint", "right_shoulder_roll_link" },
        new[] { "right_shoulder_yaw_joint", "right_shoulder_yaw_link" },
        new[] { "right_elbow_joint", "right_elbow_link" },
        new[] { "right_wrist_roll_joint", "right_wrist_roll_link" },
        new[] { "right_wrist_pitch_joint", "right_wrist_pitch_link" },
        new[] { "right_wrist_yaw_joint", "right_wrist_yaw_link" },
    };

    private bool g1JointMapValid;
    private bool hasLoggedInvalidJointMap;
    private bool hasLoggedDirectJointStateError;
    private bool hasLoggedPartialDirectJointStateError;
    private readonly List<Collider> replayMirrorColliders = new List<Collider>();
    private readonly List<bool> replayMirrorColliderStates = new List<bool>();
    private bool replayMirrorCollidersDisabled;
    private static readonly PropertyInfo ArticulationJointNameProperty =
        typeof(ArticulationBody).GetProperty("jointName", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);

    /// <summary>
    /// Dump the Unity-joint-index → joint-name correspondence to the Console.
    /// Compare against expected URDF order to verify identity mapping is still
    /// correct for the current G1 prefab.
    /// </summary>
    private void DumpJointMapping()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"[G1mimicAgent:{name}] Joint mapping (explicit CSV index -> Unity joint, valid={g1JointMapValid}):");
        for (int i = 0; i < 29 && i < jh.Length; i++)
        {
            string jointName = (jh[i] != null) ? jh[i].name : "<null>";
            sb.AppendLine($"  Unity jh[{i,2}] = '{jointName}'   ←  CSV[{i}]");
        }
        sb.AppendLine("Expected URDF order (29 DOF G1):");
        sb.AppendLine("  0..5  left  leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll");
        sb.AppendLine("  6..11 right leg: same suborder");
        sb.AppendLine("  12..14 waist: yaw, roll, pitch");
        sb.AppendLine("  15..21 left  arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw");
        sb.AppendLine("  22..28 right arm: same suborder");
        sb.AppendLine("If 15..21 are RIGHT arm names instead of left, the prefab's traversal swapped sides; needs a permutation.");
        UnityEngine.Debug.Log(sb.ToString());
    }

    private void BuildDeterministicG1JointMap(IReadOnlyList<ArticulationBody> revoluteJoints)
    {
        Array.Clear(jh, 0, jh.Length);
        g1JointMapValid = false;
        hasLoggedInvalidJointMap = false;
        hasLoggedDirectJointStateError = false;
        hasLoggedPartialDirectJointStateError = false;

        var byName = new Dictionary<string, ArticulationBody>(StringComparer.Ordinal);
        for (int i = 0; i < revoluteJoints.Count; i++)
        {
            ArticulationBody joint = revoluteJoints[i];
            if (joint == null)
            {
                continue;
            }

            AddJointName(byName, GetSerializedJointName(joint), joint);
            AddJointName(byName, joint.name, joint);
            AddJointName(byName, joint.gameObject.name, joint);
        }

        var missing = new List<string>();
        for (int i = 0; i < G1CsvJointAliases.Length && i < jh.Length; i++)
        {
            ArticulationBody joint = FindJointByAlias(byName, G1CsvJointAliases[i]);
            if (joint != null)
            {
                jh[i] = joint;
                continue;
            }

            missing.Add($"CSV[{i}] expected one of [{string.Join(", ", G1CsvJointAliases[i])}]");
        }

        if (missing.Count == 0)
        {
            g1JointMapValid = true;
            return;
        }

        // Keep traversal bindings for non-replay code paths, but refuse qpos mirroring.
        for (int i = 0; i < jh.Length && i < revoluteJoints.Count; i++)
        {
            if (jh[i] == null)
            {
                jh[i] = revoluteJoints[i];
            }
        }

        UnityEngine.Debug.LogError(
            $"[G1mimicAgent:{name}] Explicit G1 joint map is incomplete. " +
            "Replay/live qpos mirroring will be refused to avoid applying CSV values to wrong joints. " +
            string.Join("; ", missing));
    }

    private static ArticulationBody FindJointByAlias(Dictionary<string, ArticulationBody> byName, string[] aliases)
    {
        if (aliases == null)
        {
            return null;
        }

        for (int i = 0; i < aliases.Length; i++)
        {
            string key = NormalizeJointName(aliases[i]);
            if (!string.IsNullOrEmpty(key) && byName.TryGetValue(key, out ArticulationBody joint))
            {
                return joint;
            }
        }

        return null;
    }

    private static void AddJointName(Dictionary<string, ArticulationBody> byName, string rawName, ArticulationBody joint)
    {
        string key = NormalizeJointName(rawName);
        if (string.IsNullOrEmpty(key) || byName.ContainsKey(key))
        {
            return;
        }

        byName.Add(key, joint);
    }

    private static string GetSerializedJointName(ArticulationBody joint)
    {
        if (joint == null || ArticulationJointNameProperty == null)
        {
            return string.Empty;
        }

        try
        {
            return ArticulationJointNameProperty.GetValue(joint) as string ?? string.Empty;
        }
        catch
        {
            return string.Empty;
        }
    }

    private static string NormalizeJointName(string rawName)
    {
        if (string.IsNullOrWhiteSpace(rawName))
        {
            return string.Empty;
        }

        return rawName.Trim().ToLowerInvariant();
    }

    float[] uff = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] u = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] utotal = new float[29] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    int[] trainid=new int[13]{5,5,13,13,   6,14,15,   5,5,13,13,   16,17};

    private List<string> actionFolders = new List<string>();
    public int motion_id;
    public string motion_name;
    [Tooltip("Primary Unitree G1 dataset folder. If missing, the script tries robot-specific fallbacks only. " +
             "Relative paths resolve from the project root; absolute paths are used as-is.")]
    [SerializeField] private string datasetRelativePath = "Assets/Gewu/Imitation/dataset/unitree_g1";
    [SerializeField] private string datasetFallbackRelativePath = "Assets/Imitation/dataset/unitree_g1";
    public bool useExternalReplayData = false;

    private string dofFilePath;
    private string rotFilePath;
    private string posFilePath;

    // Robot-specific paths only. The dataset root now contains per-robot
    // subfolders, so using it as a replay source produces no CSV rows.
    private static readonly string[] DefaultDatasetSearchPaths =
    {
        "Assets/Gewu/Imitation/dataset/unitree_g1",
        "Assets/Imitation/dataset/unitree_g1",
    };

    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();

    private enum ReplayDataMode
    {
        DatasetReplay,
        ExternalCsvReplay,
        LiveRealtimeCsv
    }

    private ReplayDataMode replayDataMode = ReplayDataMode.DatasetReplay;

    private int currentFrame;
    private int realtimePreviousMaxStep = -1;
    private float realtimeFrameCursor;
    private float realtimePlaybackFps = ReplayCsvUtility.SourceFps;
    private float realtimePlaybackBufferSeconds = 0.2f;

    float[] currentData = new float[36];
    float[] realtimeSampledData = new float[36];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[29];

    Transform body;

    // Snapshot of the articulation's rest pose, captured once in Initialize().
    // Stored as plain arrays so their size is immutable — no risk of them growing
    // to 35 entries if GetJointPositions is called again later on a dirty state.
    private float[] restPositions;
    private float[] restVelocities;

    // Legacy List<float> kept for the initial GetJointPositions / GetJointVelocities call.
    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();

    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[29];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    private readonly List<float> replayZeroVelocities = new List<float>();
    int tt = 0;
    public int frame0 = 100;

    public float positionKp = 1000f;
    public float positionKd = 50f;
    public float rotationKp = 1000f;
    public float rotationKd = 50f;

    private bool _isClone = false;
    private bool isRobotSelectedInScene = true;

    // ── IMimicAgent surface ───────────────────────────────────────────────────
    public string RobotKey => string.IsNullOrWhiteSpace(robotKey) ? "unitree_g1" : robotKey.Trim();
    public GameObject AgentGameObject => gameObject;
    public bool UseExternalReplayData
    {
        get => useExternalReplayData;
        set
        {
            useExternalReplayData = value;
            if (!value)
            {
                replayDataMode = ReplayDataMode.DatasetReplay;
                realtimeFrameCursor = 0f;
                return;
            }

            if (replayDataMode != ReplayDataMode.LiveRealtimeCsv)
            {
                replayDataMode = ReplayDataMode.ExternalCsvReplay;
            }
        }
    }
    public bool ReplayMode { get => replay; set => replay = value; }
    public int MotionId { get => motion_id; set => motion_id = value; }
    public void RequestEndEpisode() => EndEpisode();
    public int ExpectedCsvColumns => 36;
    private bool IsLiveRealtimeCsv => useExternalReplayData && replayDataMode == ReplayDataMode.LiveRealtimeCsv;

    public void SetRobotSelectedInScene(bool isSelected)
    {
        isRobotSelectedInScene = isSelected;
        if (isSelected) return;

        UseExternalReplayData = false;
        ReplayMode = false;
        RestoreReplayMirrorColliders();
        if (art0 != null)
        {
            art0.velocity = Vector3.zero;
            art0.angularVelocity = Vector3.zero;
            art0.immovable = true;
        }
    }

    public bool BeginRealtimeCsv()
    {
        if (realtimePreviousMaxStep < 0)
        {
            realtimePreviousMaxStep = MaxStep;
        }

        MaxStep = 0;
        refData = new List<float[]>();
        itpData = new List<float[]>();
        motion_name = "live_motion";
        currentFrame = 0;
        realtimeFrameCursor = 0f;
        useExternalReplayData = true;
        replayDataMode = ReplayDataMode.LiveRealtimeCsv;
        tt = 0;
        DisableReplayMirrorColliders();
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
        RestoreReplayMirrorColliders();
    }

    /// <summary>
    /// Imperative reset of the agent's PD targets and replay bookkeeping.
    ///
    /// IMPORTANT — what this DOES NOT do, and why:
    /// We deliberately do NOT call <c>TeleportRoot</c>, <c>SetJointPositions</c>,
    /// or <c>SetJointVelocities</c> here. Those operations write the engine's
    /// articulation cache, and they are UNSAFE in the frame immediately after
    /// <c>SetActive(true)</c>: Unity defers the actual ArticulationBody
    /// rebuild to the next FixedUpdate, so on the same frame the cache slots
    /// don't yet correspond to the final joint indices. Writing
    /// <c>restPositions</c> at that moment ends up placing values into the
    /// wrong slots (elbow slot gets a wrist value, etc.), which is exactly
    /// the bug that produced "G1's forearm + hand reversed front-to-back
    /// after switching robots".
    ///
    /// What we DO here is safe regardless of active state:
    ///   - Zero every revolute joint's xDrive.target. xDrive is a per-joint
    ///     configuration property (not a per-step physics state), so writes
    ///     persist across active/inactive cycles and don't depend on the
    ///     articulation rebuild. This is the actual fix for "stale PD target
    ///     dragging joints toward the last replay pose" — that was the only
    ///     thing ResetToInitialState ever needed to address.
    ///   - Clear the C# bookkeeping arrays (u/uff/utotal/currentFrame/tt).
    ///
    /// The actual pose restoration (root TeleportRoot + SetJointPositions of
    /// restPositions + per-frame uff write) happens in <c>OnEpisodeBegin</c>,
    /// which is queued by <c>RequestEndEpisode</c> in StartInput and fires
    /// on the NEXT FixedUpdate — at which point the articulation has been
    /// fully rebuilt and the cache writes are safe.
    /// </summary>
    public void ResetToInitialState()
    {
        // Clear all xDrive targets so the PD controller can't drag joints
        // toward stale targets in the one-frame gap between SetActive(true)
        // and OnEpisodeBegin firing.
        for (int i = 0; i < 29; i++)
        {
            if (jh[i] != null) SetJointTargetDeg(jh[i], 0f);
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
        }

        // Replay bookkeeping: rewind to frame0 and zero the step counter so
        // the next OnEpisodeBegin starts cleanly.
        currentFrame = useExternalReplayData ? 0 : frame0;
        if (useExternalReplayData)
        {
            realtimeFrameCursor = 0f;
        }
        tt = 0;
    }

    // Registration with the scene-wide registry happens at the end of the
    // ML-Agents Initialize() override below — NOT in OnEnable / OnDisable.
    // Reason: ML-Agents Agent uses its own non-virtual OnEnable() to drive
    // LazyInitialize(); shadowing it here would prevent Initialize() from
    // ever running. The registry detects stale entries via AgentGameObject
    // == null, so explicit Unregister on disable is not required.

    void Start()
    {
        Time.fixedDeltaTime = 0.02f;

        if (train && !_isClone)
        {
            for (int i = 1; i < 34; i++)
            {
                GameObject clone = Instantiate(gameObject);
                clone.transform.position = transform.position + new Vector3(i * 2f, 0, 0);
                clone.name = $"{name}_Clone_{i}";
                clone.GetComponent<G1mimicAgent>()._isClone = true;
            }
        }
    }

    void ChangeLayerRecursively(GameObject obj, int targetLayer)
    {
        obj.layer = targetLayer;
        foreach (Transform child in obj.transform) ChangeLayerRecursively(child.gameObject, targetLayer);
    }

    public override void Initialize()
    {
        arts = this.GetComponentsInChildren<ArticulationBody>();
        var revoluteJoints = new List<ArticulationBody>(29);
        for (int k = 0; k < arts.Length; k++)
        {
            if (arts[k].jointType == ArticulationJointType.RevoluteJoint)
            {
                revoluteJoints.Add(arts[k]);
            }
        }
        BuildDeterministicG1JointMap(revoluteJoints);

        body = arts[0].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        // ── CRITICAL: capture pos0/rot0/restPositions only ONCE, ever. ─────
        // ML-Agents' Agent.OnEnable re-fires LazyInitialize each time the
        // GameObject is SetActive(true). The user's stack trace confirms
        // Initialize being invoked from ApplyRobotVisibility → SetActive(true)
        // during a dropdown switch. On THAT call:
        //   - arts[0].immovable is still true (set at the end of the previous
        //     OnEpisodeBegin), so the articulation cache size is 29.
        //   - The articulation hasn't been rebuilt yet from the SetActive flip.
        //   - GetJointPositions returns whatever junk was in the cache when
        //     the body was deactivated — i.e. the last replay frame's joint
        //     angles, not the bind pose.
        // If we let those values overwrite restPositions every cycle, the
        // next OnEpisodeBegin's SafeSetJointPositions(restPositions) writes
        // the previous-frame replay angles back into the articulation as if
        // they were the rest pose, and the corruption compounds each switch
        // (= "arms twisted into body after repeated switches").
        //
        // Guard with a null check so subsequent Initialize() calls are
        // effectively no-ops for the captured state. The first call (at
        // scene load when the body is in its true bind pose) is the only
        // one that captures.
        if (restPositions == null)
        {
            pos0 = body.position;
            rot0 = body.rotation;

            // Capture the rest pose into Lists first (Unity API requires List<float>).
            art0.GetJointPositions(P0);
            art0.GetJointVelocities(W0);

            // Snapshot into fixed-size arrays so subsequent GetJointPositions
            // calls (which mutate the same list in place) can never change
            // the size of our baseline.
            restPositions  = P0.ToArray();
            restVelocities = W0.ToArray();
        }

        TryLoadCurrentMotionData(keepProgress: false);

        // Print Unity joint name → CSV index correspondence so the user can
        // verify the G1 prefab traversal still matches URDF order.
        if (logJointMappingOnStart) DumpJointMapping();

        // Register with the scene-wide IMimicAgent registry so that StartInput,
        // Stop and Replay can route operations to this specific robot by its
        // RobotKey string. Done at the end of Initialize() so it only happens
        // once Agent.LazyInitialize has finished (otherwise registering during
        // an in-progress init could expose a half-constructed agent to UI
        // scripts that try to use it immediately).
        MimicAgentRegistry.Instance.Register(this);
    }

    List<string> GetCsvFileNames(string directoryPath)
    {
        List<string> csvFiles = new List<string>();

        try
        {
            if (Directory.Exists(directoryPath))
            {
                string[] allFiles = Directory.GetFiles(directoryPath);
                foreach (string file in allFiles)
                {
                    if (Path.GetExtension(file).ToLower() == ".csv")
                    {
                        string fileName = Path.GetFileName(file);
                        csvFiles.Add(Path.Combine(directoryPath, fileName));
                    }
                }
            }
            else
            {
                UnityEngine.Debug.LogError("Directory does not exist: " + directoryPath);
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
                        if (float.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsedValue) ||
                            float.TryParse(trimmed, out parsedValue))
                        {
                            frameData.Add(parsedValue);
                        }
                    }

                    if (frameData.Count < 36) continue;
                    if (frameData.Count > 36) frameData = frameData.Take(36).ToList();

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
        if (string.IsNullOrWhiteSpace(filePath))
        {
            UnityEngine.Debug.LogWarning("LoadReplayCsvFromPath failed: filePath is empty.");
            return false;
        }

        if (!File.Exists(filePath))
        {
            UnityEngine.Debug.LogWarning("LoadReplayCsvFromPath failed: file does not exist: " + filePath);
            return false;
        }

        List<float[]> data = LoadDataFromFile(filePath);
        if (data == null || data.Count == 0)
        {
            UnityEngine.Debug.LogWarning("LoadReplayCsvFromPath failed: no valid data in " + filePath);
            return false;
        }

        motion_name = Path.GetFileNameWithoutExtension(filePath);
        replayDataMode = ReplayDataMode.ExternalCsvReplay;
        useExternalReplayData = true;
        realtimeFrameCursor = 0f;
        DisableReplayMirrorColliders();
        return ApplyReplayData(data, keepProgress, resample30FpsToFixed50Hz: true, startAtFrameZero: true);
    }

    private bool TryLoadCurrentMotionData(bool keepProgress)
    {
        string datasetPath = ResolveDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath)) return false;

        List<string> csvFileNames = GetCsvFileNames(datasetPath);
        if (csvFileNames.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[G1mimicAgent] No csv files found in dataset path: " + datasetPath);
            return false;
        }

        motion_id = Mathf.Clamp(motion_id, 0, csvFileNames.Count - 1);
        string selectedCsv = csvFileNames[motion_id];
        motion_name = Path.GetFileNameWithoutExtension(selectedCsv);

        List<float[]> data = LoadDataFromFile(selectedCsv);
        if (data == null || data.Count == 0)
        {
            UnityEngine.Debug.LogError("CSV contains no valid frame data: " + selectedCsv);
            return false;
        }

        replayDataMode = ReplayDataMode.DatasetReplay;
        return ApplyReplayData(data, keepProgress, resample30FpsToFixed50Hz: true, startAtFrameZero: false);
    }

    private bool ApplyReplayData(
        List<float[]> data,
        bool keepProgress,
        bool resample30FpsToFixed50Hz,
        bool startAtFrameZero)
    {
        if (data == null || data.Count == 0) return false;

        int oldFrame = currentFrame;
        refData = data;

        itpData = resample30FpsToFixed50Hz
            ? ReplayCsvUtility.Resample30FpsToFixed50Hz(refData)
            : ReplayCsvUtility.CopyRows(refData, ExpectedCsvColumns);

        if (itpData.Count == 0) return false;

        int defaultStartFrame = startAtFrameZero ? 0 : frame0;
        currentFrame = keepProgress
            ? Mathf.Clamp(oldFrame, 0, itpData.Count - 1)
            : Mathf.Clamp(defaultStartFrame, 0, itpData.Count - 1);
        realtimeFrameCursor = currentFrame;

        return true;
    }

    private string ResolveDatasetPath()
    {
        List<string> candidates = new List<string>();
        List<string> existingWithoutCsv = new List<string>();

        // 1) Inspector-configured paths first (so user overrides win).
        AddDatasetPathCandidate(candidates, datasetRelativePath);
        AddDatasetPathCandidate(candidates, datasetFallbackRelativePath);

        // 2) Built-in fallback list — catches scenes whose serialized
        //    relative-path fields are stale / contain typos (e.g. the
        //    historical "Imatation" misspelling, or absolute paths from a
        //    different machine that no longer exist).
        foreach (string fallback in DefaultDatasetSearchPaths)
        {
            AddDatasetPathCandidate(candidates, fallback);
        }

        foreach (string candidate in candidates)
        {
            if (!Directory.Exists(candidate))
            {
                continue;
            }

            if (GetCsvFileNames(candidate).Count > 0)
            {
                return candidate;
            }

            existingWithoutCsv.Add(candidate);
        }

        UnityEngine.Debug.LogWarning(
            "[G1mimicAgent] No usable G1 replay dataset found. Tried: " + string.Join(" | ", candidates) +
            (existingWithoutCsv.Count > 0 ? "\nExisting folders without CSV: " + string.Join(" | ", existingWithoutCsv) : string.Empty) +
            "\nLive retargeting still works once a CSV arrives via StartInput.");
        return string.Empty;
    }

    private void AddDatasetPathCandidate(List<string> candidates, string configuredPath)
    {
        if (string.IsNullOrWhiteSpace(configuredPath)) return;

        string absolute = ToAbsoluteProjectPath(configuredPath);
        if (!string.IsNullOrWhiteSpace(absolute) && !candidates.Contains(absolute))
            candidates.Add(absolute);
    }

    private string ToAbsoluteProjectPath(string path)
    {
        string normalized = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(normalized)) return normalized;

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (string.IsNullOrWhiteSpace(projectRoot)) return string.Empty;

        return Path.GetFullPath(Path.Combine(projectRoot, normalized));
    }

    // ── safe articulation helpers ─────────────────────────────────────────────

    // Per-instance scratch list for the cache-size probe. Avoids allocating a
    // fresh List every OnEpisodeBegin / FixedUpdate while keeping each clone's
    // probe independent.
    private readonly List<float> _cacheProbe = new List<float>();

    /// <summary>
    /// Returns the articulation cache size that <c>SetJointPositions</c> /
    /// <c>SetJointVelocities</c> expects right now. The naive approach of
    /// summing <c>ab.dofCount</c> across <c>arts</c> is unreliable because
    /// <c>dofCount</c> can be temporarily out of sync with the engine right
    /// after the root's <c>immovable</c> flag changes — that mismatch was the
    /// root cause of the "Articulation cache size (35) does not match supplied
    /// list size (29)" error.
    ///
    /// Instead, probe the engine directly: <c>GetJointPositions</c> always
    /// returns a list sized to whatever cache <c>SetJointPositions</c> will
    /// accept in the same frame, so we use the result's <c>Count</c> as the
    /// authoritative target size.
    /// </summary>
    private int ArticulationCacheSize()
    {
        if (arts == null || arts.Length == 0 || arts[0] == null) return 0;
        _cacheProbe.Clear();
        arts[0].GetJointPositions(_cacheProbe);
        return _cacheProbe.Count;
    }

    /// <summary>
    /// Calls SetJointPositions, handling the cache-size shift that happens
    /// when <c>arts[0].immovable</c> toggles between captures and writes.
    ///
    /// Why this is non-trivial: restPositions is captured at Initialize when
    /// immovable=false (cache = 6 root DOFs + 29 joint DOFs = 35). After the
    /// first OnEpisodeBegin sets immovable=true the cache shrinks to 29
    /// (joints only — root slots are removed). A naive "trim to first N"
    /// approach would keep restPositions[0..28] = "6 root values + 23 joint
    /// values" and write those into the 29 joint cache slots — every joint
    /// receives the wrong value (left_hip_pitch gets a root quaternion
    /// component, etc.). On repeated robot switches the corruption compounds
    /// until arms end up twisted into the body.
    ///
    /// The fix: when the source list is LONGER than the current cache, take
    /// the TAIL of the source (those are the joint values, with the root
    /// portion stripped off). When the source is SHORTER, prepend the
    /// current root cache values to it.
    /// </summary>
    private void SafeSetJointPositions(List<float> positions)
    {
        int cacheSize = ArticulationCacheSize();
        List<float> safe = AlignToCache(positions, cacheSize);
        arts[0].SetJointPositions(safe);
    }

    /// <summary>
    /// Same cache-alignment wrapper for SetJointVelocities.
    /// </summary>
    private void SafeSetJointVelocities(List<float> velocities)
    {
        int cacheSize = ArticulationCacheSize();
        List<float> safe = AlignToCache(velocities, cacheSize);
        arts[0].SetJointVelocities(safe);
    }

    /// <summary>
    /// Reshape <paramref name="source"/> to exactly <paramref name="cacheSize"/>
    /// entries, preserving joint-slot alignment when the difference is the
    /// articulation root's 6 DOFs (the only difference we expect in this
    /// project, caused by toggling <c>arts[0].immovable</c>).
    ///
    /// - source.Count == cacheSize: pass through unchanged.
    /// - source.Count == cacheSize + 6: drop the first 6 entries (root slots
    ///   that aren't present in the current immovable=true cache).
    /// - source.Count + 6 == cacheSize: prepend the current cache's first 6
    ///   entries (so we keep whatever root values are currently in the
    ///   cache, and tack on the source joint values).
    /// - any other mismatch: fall back to trimming/padding so we don't throw,
    ///   but log a warning because this is unexpected territory.
    /// </summary>
    private List<float> AlignToCache(List<float> source, int cacheSize)
    {
        if (source == null) source = new List<float>();
        if (source.Count == cacheSize) return source;

        const int RootDofCount = 6; // free articulation root: 3 pos + 3 rot DOFs.

        if (source.Count == cacheSize + RootDofCount)
        {
            // Source was captured with immovable=false (root present), current
            // cache has immovable=true (root stripped). Take the joint tail.
            return source.GetRange(RootDofCount, cacheSize);
        }

        if (source.Count + RootDofCount == cacheSize)
        {
            // Source is joint-only, current cache includes root. Preserve
            // current root values and append source joints.
            _cacheProbe.Clear();
            arts[0].GetJointPositions(_cacheProbe);
            var safe = new List<float>(cacheSize);
            for (int i = 0; i < RootDofCount && i < _cacheProbe.Count; i++)
                safe.Add(_cacheProbe[i]);
            // Pad in case probe was shorter than expected.
            while (safe.Count < RootDofCount) safe.Add(0f);
            safe.AddRange(source);
            return safe;
        }

        UnityEngine.Debug.LogWarning(
            $"[G1mimicAgent] Unexpected cache mismatch: source={source.Count}, cache={cacheSize}. " +
            "Falling back to trim/pad — joint values may be misaligned.");
        var result = new List<float>(cacheSize);
        for (int i = 0; i < cacheSize; i++)
            result.Add(i < source.Count ? source[i] : 0f);
        return result;
    }

    // ── episode lifecycle ─────────────────────────────────────────────────────

    public override void OnEpisodeBegin()
    {
        if (!isRobotSelectedInScene)
        {
            return;
        }

        // ── CRITICAL: restore immovable=false BEFORE any articulation-cache
        // writes. The previous OnEpisodeBegin (last line below) sets
        // arts[0].immovable = true so the root stays put during replay. That
        // shrinks the articulation cache from 35 slots (6 root + 29 joints)
        // down to 29 (joints only). On the NEXT OnEpisodeBegin call,
        // restPositions still has 35 entries (captured at Initialize when
        // immovable=false), but the current cache is 29 — EnsureListSize then
        // trims restPositions to its FIRST 29 entries, which are the 6 root
        // values followed by only the first 23 joint values. Those get
        // written into the 29 joint slots, so every joint receives the
        // wrong value (left_hip_pitch gets a root pos float, etc.), and on
        // repeated switches the joints drift further and further into
        // garbage poses ("arms twisted into body" symptom).
        //
        // Flipping immovable=false first restores cache size to 35 so the
        // restPositions write lands on the right slots.
        if (replay || useExternalReplayData)
        {
            DisableReplayMirrorColliders();
            for (int i = 0; i < 29; i++)
            {
                u[i] = 0f;
                uff[i] = 0f;
                utotal[i] = 0f;
            }

            currentFrame = useExternalReplayData ? 0 : frame0;
            if (useExternalReplayData)
            {
                realtimeFrameCursor = 0f;
            }

            if (replay && !useExternalReplayData)
            {
                TryLoadCurrentMotionData(keepProgress: false);
            }

            List<float[]> replayData = useExternalReplayData ? itpData : refData;
            if (replayData == null || replayData.Count == 0)
            {
                UnityEngine.Debug.LogError("replay data is empty, skip OnEpisodeBegin.");
                return;
            }

            currentFrame = Mathf.Clamp(currentFrame, 0, replayData.Count - 1);
            tt = 0;
            currentData = replayData[currentFrame];
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, 29);

            for (int i = 0; i < 29; i++)
            {
                float targetRad = ToUnityJointRadians(i, currentDof[i]);
                uff[i] = targetRad * Mathf.Rad2Deg;
                SetJointTargetDeg(jh[i], uff[i]);
            }

            this.newPosition = new Vector3(-currentPos[1], currentPos[2] + 0.04f, currentPos[0]);
            this.newRotation = new Quaternion(
                -currentRot[1],
                 currentRot[2],
                 currentRot[0],
                -currentRot[3]
            );
            this.newPosition.x += pos0.x;
            this.newPosition.z += pos0.z;

            Physics.gravity = Vector3.zero;
            arts[0].immovable = true;
            arts[0].TeleportRoot(this.newPosition, this.newRotation);
            arts[0].velocity = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
            ApplyReplayFrameToArticulation();
            return;
        }

        RestoreReplayMirrorColliders();
        arts[0].immovable       = false;
        arts[0].TeleportRoot(pos0, rot0);
        arts[0].velocity        = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;

        // Use the immutable rest-pose arrays captured at Initialize() time.
        // Now that immovable=false, cache size matches restPositions length.
        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        for (int i = 0; i < 29; i++) u[i]   = 0;
        for (int i = 0; i < 29; i++) uff[i] = 0;
        currentFrame = useExternalReplayData ? 0 : frame0;
        if (useExternalReplayData)
        {
            realtimeFrameCursor = 0f;
        }

        // Standard replay: reload from the dataset by motion_id.
        // Live mode: data is injected externally by StartInput; don't reload here.
        if (replay && !useExternalReplayData)
        {
            TryLoadCurrentMotionData(keepProgress: false);
        }

        List<float[]> episodeData = useExternalReplayData ? itpData : refData;
        if (episodeData == null || episodeData.Count == 0)
        {
            UnityEngine.Debug.LogError("replay data is empty, skip OnEpisodeBegin.");
            return;
        }

        currentFrame = Mathf.Clamp(currentFrame, 0, episodeData.Count - 1);
        tt = 0;
        currentData = episodeData[currentFrame];
        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, 29);

        // 1:1 identity mapping — currentDof[i] drives jh[i]. The current G1
        // prefab's joint hierarchy already matches URDF order; no permutation.
        for (int i = 0; i < 29; i++)
        {
            uff[i] = currentDof[i] * 180f / 3.14f;
            SetJointTargetDeg(jh[i], uff[i]);
        }

        Vector3 newPosition = new Vector3(-currentPos[1], currentPos[2] + 0.04f, currentPos[0]);
        Quaternion newRotation = new Quaternion(
            -currentRot[1],
             currentRot[2],
             currentRot[0],
            -currentRot[3]
        );
        newPosition.x += pos0.x;
        newPosition.z += pos0.z;

        if (replay || useExternalReplayData)
        {
            ApplyReplayFrameToArticulation();
        }
        else
        {
            arts[0].TeleportRoot(newPosition, newRotation);
            arts[0].velocity        = Vector3.zero;
            arts[0].angularVelocity = Vector3.zero;
            arts[0].immovable       = true;
        }
    }

    List<float[]> Interpolate(float[] t, List<float[]> posList, float[] targetT)
    {
        if (t.Length != posList.Count)
        {
            UnityEngine.Debug.LogError("t and posList must have the same length");
            return null;
        }
        int dimension = posList[0].Length;
        foreach (float[] arr in posList)
        {
            if (arr.Length != dimension)
            {
                UnityEngine.Debug.LogError("All arrays in posList must have the same length");
                return null;
            }
        }
        List<float[]> result = new List<float[]>();
        for (int i = 0; i < targetT.Length; i++)
        {
            float tValue = targetT[i];
            if (tValue < t[0] || tValue > t[t.Length - 1])
            {
                UnityEngine.Debug.LogError("tValue is out of range");
                return null;
            }
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
        // Guard against the half-torn-down articulation state. CollectObservations
        // is called from ML-Agents' Agent.OnDisable() → NotifyAgentDone() path
        // when the GameObject is SetActive(false), and at that point each
        // ArticulationBody's `jointPosition` / `jointVelocity` ReducedSpace can
        // have a dofCount of 0 (joints aren't simulated while inactive). The
        // original `jh[i].jointPosition[0]` indexing then throws
        // IndexOutOfRangeException. We must still emit the same NUMBER of
        // observations or ML-Agents' VectorSensor shape check will fail, so
        // substitute zeros when the cache slot isn't available.
        if (body != null) sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * 3.14f / 180f);
        else              sensor.AddObservation(0f);
        if (body != null) sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * 3.14f / 180f);
        else              sensor.AddObservation(0f);
        if (body != null && art0 != null)
            sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        else
            sensor.AddObservation(Vector3.zero);

        for (int i = 0; i < 29; i++)
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

        if (replay || useExternalReplayData)
        {
            return;
        }

        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        float kb = 50;
        for (int i = 0; i < 29; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
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

        if (itpData != null && itpData.Count > 0)
        {
            if (useExternalReplayData)
            {
                if (IsLiveRealtimeCsv)
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
            }
            else
            {
                currentFrame = Mathf.Clamp(currentFrame, 0, itpData.Count - 1);
                currentData = itpData[currentFrame];
            }
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, 29);
            // 1:1 identity mapping — radians → degrees, no joint permutation.
            for (int i = 0; i < 29; i++) uff[i] = currentDof[i] * 180f / 3.14f;

            newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
            newRotation = new Quaternion(
                -currentRot[1],
                 currentRot[2],
                 currentRot[0],
                -currentRot[3]
            );
            newPosition.x += pos0.x;
            newPosition.z += pos0.z;

            if (replay || useExternalReplayData)
            {
                ApplyReplayFrameToArticulation();
            }
            else
            {
                if (tt > 3)
                {
                    arts[0].immovable = false;

                    Vector3 positionError = newPosition - body.position;
                    Vector3 velocityError = -art0.velocity;
                    Vector3 positionForce = positionKp * positionError + positionKd * velocityError;
                    arts[0].AddForce(positionForce);

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

        if (replay || useExternalReplayData)
        {
            tt++;
            AdvanceMotionFrameIfNeeded();
            return;
        }

        tt++;
        var vel = body.InverseTransformDirection(art0.velocity);
        var wel = body.InverseTransformDirection(art0.angularVelocity);

        var live_reward = 1f;
        float rot_reward = 0;
        float pos_reward = 0;
        if (tt > 3)
        {
            arts[0].immovable = false;
            rot_reward = -0.01f * Quaternion.Angle(body.rotation, newRotation);
            pos_reward = -1f * (body.position - newPosition).magnitude;

            if (!replay && (Quaternion.Angle(body.rotation, newRotation) > 40f || (body.position - newPosition).magnitude > 0.5f))
            {
                EndEpisode();
            }
        }
        var dof_reward = 0f;
        for (int i = 0; i < 29; i++) dof_reward += -0.1f * Mathf.Abs(jh[i].jointPosition[0] - currentDof[i]);
        var reward = live_reward + (rot_reward + pos_reward) * 1f + dof_reward;
        AddReward(reward);

        AdvanceMotionFrameIfNeeded();
    }

    private void ApplyReplayFrameToArticulation()
    {
        if (arts == null || arts.Length == 0 || arts[0] == null)
        {
            return;
        }

        if (!g1JointMapValid)
        {
            if (!hasLoggedInvalidJointMap)
            {
                UnityEngine.Debug.LogError($"[G1mimicAgent:{name}] Replay/live qpos mirror skipped because the explicit G1 joint map is invalid.");
                hasLoggedInvalidJointMap = true;
            }
            return;
        }

        Physics.gravity = Vector3.zero;
        arts[0].immovable = true;
        arts[0].TeleportRoot(newPosition, newRotation);
        arts[0].velocity = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;

        int directWriteCount = 0;
        for (int i = 0; i < 29; i++)
        {
            ArticulationBody joint = jh[i];
            float targetRad = ToUnityJointRadians(i, currentDof[i]);
            float targetDeg = targetRad * Mathf.Rad2Deg;
            u[i] = 0f;
            uff[i] = targetDeg;
            utotal[i] = targetDeg;
            SetJointTargetDeg(joint, targetDeg);
            if (SetJointPositionRad(joint, targetRad))
            {
                directWriteCount++;
            }
        }

        ZeroArticulationJointVelocities();
        ZeroArticulationBodyVelocities();

        if (directWriteCount == 0 && !hasLoggedDirectJointStateError)
        {
            UnityEngine.Debug.LogError($"[G1mimicAgent:{name}] Direct qpos mirror could not write any joint positions; all mapped joints reported empty reduced-space state.");
            hasLoggedDirectJointStateError = true;
        }
        else if (directWriteCount != 29 && !hasLoggedPartialDirectJointStateError)
        {
            UnityEngine.Debug.LogWarning($"[G1mimicAgent:{name}] Direct qpos mirror wrote {directWriteCount}/29 joints. Partial reduced-space writes can cause localized twitching.");
            hasLoggedPartialDirectJointStateError = true;
        }
        LogReplayMirrorDiagnosticsIfNeeded();
    }

    private float ToUnityJointRadians(int jointIndex, float csvRadians)
    {
        if (unityJointSigns == null || jointIndex < 0 || jointIndex >= unityJointSigns.Length)
        {
            return csvRadians;
        }

        float sign = Mathf.Approximately(unityJointSigns[jointIndex], 0f)
            ? 1f
            : Mathf.Sign(unityJointSigns[jointIndex]);
        return csvRadians * sign;
    }

    private void ZeroArticulationJointVelocities()
    {
        if (arts == null || arts.Length == 0 || arts[0] == null)
        {
            return;
        }

        replayZeroVelocities.Clear();
        arts[0].GetJointVelocities(replayZeroVelocities);
        for (int i = 0; i < replayZeroVelocities.Count; i++)
        {
            replayZeroVelocities[i] = 0f;
        }

        SafeSetJointVelocities(replayZeroVelocities);
    }

    private void ZeroArticulationBodyVelocities()
    {
        if (arts == null)
        {
            return;
        }

        for (int i = 0; i < arts.Length; i++)
        {
            ArticulationBody bodyPart = arts[i];
            if (bodyPart == null)
            {
                continue;
            }

            bodyPart.velocity = Vector3.zero;
            bodyPart.angularVelocity = Vector3.zero;
        }
    }

    private void DisableReplayMirrorColliders()
    {
        if (replayMirrorCollidersDisabled)
        {
            return;
        }

        replayMirrorColliders.Clear();
        replayMirrorColliderStates.Clear();
        Collider[] colliders = GetComponentsInChildren<Collider>(true);
        for (int i = 0; i < colliders.Length; i++)
        {
            Collider collider = colliders[i];
            if (collider == null)
            {
                continue;
            }

            replayMirrorColliders.Add(collider);
            replayMirrorColliderStates.Add(collider.enabled);
            collider.enabled = false;
        }

        replayMirrorCollidersDisabled = true;
    }

    private void RestoreReplayMirrorColliders()
    {
        if (!replayMirrorCollidersDisabled)
        {
            return;
        }

        int count = Mathf.Min(replayMirrorColliders.Count, replayMirrorColliderStates.Count);
        for (int i = 0; i < count; i++)
        {
            Collider collider = replayMirrorColliders[i];
            if (collider != null)
            {
                collider.enabled = replayMirrorColliderStates[i];
            }
        }

        replayMirrorColliders.Clear();
        replayMirrorColliderStates.Clear();
        replayMirrorCollidersDisabled = false;
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

            // Replay/live is a visual qpos mirror of GMR. Do not clamp to Unity
            // drive limits here, or poses near the robot limits diverge from the
            // MuJoCo preview and can look like localized twitching.
            jointPosition[0] = radians;
            joint.jointPosition = jointPosition;

            ArticulationReducedSpace jointVelocity = joint.jointVelocity;
            if (jointVelocity.dofCount > 0)
            {
                jointVelocity[0] = 0f;
                joint.jointVelocity = jointVelocity;
            }

            joint.velocity = Vector3.zero;
            joint.angularVelocity = Vector3.zero;
            return true;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning($"[G1mimicAgent:{name}] Failed to write joint position for '{joint.name}': {e.Message}");
            return false;
        }
    }

    private float ClampJointRadiansToDrive(ArticulationBody joint, float radians)
    {
        if (joint == null)
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

    private void LogReplayMirrorDiagnosticsIfNeeded()
    {
        if (!logReplayMirrorDiagnostics)
        {
            return;
        }

        int interval = Mathf.Max(1, replayMirrorDiagnosticIntervalFrames);
        if (Time.frameCount % interval != 0)
        {
            return;
        }

        int[] watch = { 0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21 };
        var sb = new System.Text.StringBuilder();
        sb.Append($"[G1mimicAgent:{name}] replay mirror joint errors rad:");
        for (int i = 0; i < watch.Length; i++)
        {
            int jointIndex = watch[i];
            ArticulationBody joint = jh[jointIndex];
            if (joint == null || joint.jointPosition.dofCount == 0)
            {
                sb.Append($" jh[{jointIndex}]=missing");
                continue;
            }

            float target = ToUnityJointRadians(jointIndex, currentDof[jointIndex]);
            float actual = joint.jointPosition[0];
            float error = Mathf.Abs(Mathf.DeltaAngle(actual * Mathf.Rad2Deg, target * Mathf.Rad2Deg)) * Mathf.Deg2Rad;
            sb.Append($" jh[{jointIndex}]({joint.name}) target={target:F3} actual={actual:F3} err={error:F3}");
        }

        UnityEngine.Debug.Log(sb.ToString());
    }

    private void AdvanceMotionFrameIfNeeded()
    {
        if (itpData != null && currentFrame < itpData.Count - 1)
        {
            if (useExternalReplayData)
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
                    currentFrame = currentFrame + 1;
                    realtimeFrameCursor = currentFrame;
                }
            }
            else
            {
                currentFrame = currentFrame + 1;
            }
        }
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        if (joint == null)
        {
            return;
        }

        var drive = joint.xDrive;
        drive.stiffness = 180f;
        drive.damping = 8f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut) { }
}
