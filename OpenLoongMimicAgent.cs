using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gewu.Imitation;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using UnityEngine;

public class OpenLoongMimicAgent : Agent, IMimicAgent, IRealtimeCsvMimicAgent
{
    private const int RootPosCols = 3;
    private const int RootRotCols = 4;
    private const int DofCols = 31;
    private const int ExpectedCols = RootPosCols + RootRotCols + DofCols;

    private static readonly string[] CsvJointNames =
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

    private static readonly float[] CsvJointSigns =
    {
         1f, -1f,
        -1f, -1f, -1f, -1f, -1f,  1f,  1f,
         1f,  1f,  1f,  1f,  1f, -1f,  1f,
        -1f,  1f,  1f,
         1f,  1f, -1f, -1f, -1f,  1f,
         1f,  1f, -1f, -1f, -1f,  1f,
    };

    [SerializeField] private string robotKey = "openloong";
    [SerializeField] private bool replay;
    [SerializeField] private bool useExternalReplayData;
    [SerializeField] private string datasetRelativePath = "Assets/Gewu/Imitation/dataset/openloong";
    [SerializeField] private int frame0;
    [SerializeField] private float jointStiffness = 300f;
    [SerializeField] private float jointDamping = 20f;
    [SerializeField] private bool logJointMappingOnStart = true;
    [SerializeField] private bool applyMjcfAxisSignCorrection = true;
    [SerializeField] private bool clampTargetsToDriveLimits = true;

    private static readonly string[] DefaultDatasetSearchPaths =
    {
        "Assets/Gewu/Imitation/dataset/openloong",
        "Assets/Imitation/dataset/openloong",
        "Assets/Gewu/Imitation/dataset",
        "Assets/Imitation/dataset",
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
    private int currentFrame;
    private int realtimePreviousMaxStep = -1;
    private int tt;
    private Vector3 newPosition;
    private Quaternion newRotation;

    public string RobotKey => string.IsNullOrWhiteSpace(robotKey) ? "openloong" : robotKey.Trim();
    public GameObject AgentGameObject => gameObject;
    public bool UseExternalReplayData { get => useExternalReplayData; set => useExternalReplayData = value; }
    public bool ReplayMode { get => replay; set => replay = value; }
    public int MotionId { get; set; }
    public void RequestEndEpisode() => EndEpisode();
    public int ExpectedCsvColumns => ExpectedCols;

    public void BeginRealtimeCsv()
    {
        if (realtimePreviousMaxStep < 0)
        {
            realtimePreviousMaxStep = MaxStep;
        }

        MaxStep = 0;
        refData = new List<float[]>();
        itpData = new List<float[]>();
        currentFrame = 0;
        tt = 0;
        UseExternalReplayData = true;
        ReplayMode = true;
    }

    public bool AppendRealtimeCsvRows(IReadOnlyList<float[]> rows)
    {
        return ReplayCsvUtility.AppendResampled30FpsToFixed50Hz(refData, itpData, rows, ExpectedCsvColumns) > 0;
    }

    public void EndRealtimeCsv()
    {
        if (realtimePreviousMaxStep >= 0)
        {
            MaxStep = realtimePreviousMaxStep;
            realtimePreviousMaxStep = -1;
        }

        UseExternalReplayData = false;
    }

    public override void Initialize()
    {
        Time.fixedDeltaTime = 0.02f;
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

        positionCache.Clear();
        velocityCache.Clear();
        rootArticulation.GetJointPositions(positionCache);
        rootArticulation.GetJointVelocities(velocityCache);
        restPositions = positionCache.ToArray();
        restVelocities = velocityCache.ToArray();

        ResolveCsvJointMap();
        if (logJointMappingOnStart) DumpJointMapping();
        TryLoadCurrentMotionData(keepProgress: false);

        MimicAgentRegistry.Instance.Register(this);
    }

    public override void OnEpisodeBegin()
    {
        if (rootArticulation == null) return;

        rootArticulation.immovable = false;
        rootArticulation.TeleportRoot(pos0, rot0);
        rootArticulation.velocity = Vector3.zero;
        rootArticulation.angularVelocity = Vector3.zero;
        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        Array.Clear(u, 0, u.Length);
        Array.Clear(uff, 0, uff.Length);
        Array.Clear(utotal, 0, utotal.Length);
        currentFrame = frame0;
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
        if (rootArticulation == null || itpData == null || itpData.Count == 0)
        {
            return;
        }

        ApplyFrame(Mathf.Clamp(currentFrame, 0, itpData.Count - 1), teleportRoot: replay);

        tt++;
        if (tt > 3)
        {
            rootArticulation.immovable = false;
        }

        if (currentFrame < itpData.Count - 1)
        {
            currentFrame++;
        }
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

        return ApplyReplayData(data, keepProgress);
    }

    public void ResetToInitialState()
    {
        if (rootArticulation == null) return;

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
    }

    private bool TryLoadCurrentMotionData(bool keepProgress)
    {
        string datasetPath = ResolveDatasetPath();
        if (string.IsNullOrWhiteSpace(datasetPath)) return false;

        List<string> csvFiles = Directory.GetFiles(datasetPath, "*.csv", SearchOption.AllDirectories)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
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
        float[] pos = new float[RootPosCols];
        float[] rot = new float[RootRotCols];
        Array.Copy(row, 0, pos, 0, RootPosCols);
        Array.Copy(row, RootPosCols, rot, 0, RootRotCols);

        for (int i = 0; i < DofCols; i++)
        {
            float sign = applyMjcfAxisSignCorrection ? CsvJointSigns[i] : 1f;
            uff[i] = row[RootPosCols + RootRotCols + i] * Mathf.Rad2Deg * sign;
            SetJointTargetDeg(jh[i], uff[i]);
        }

        newPosition = new Vector3(-pos[1], pos[2], pos[0]);
        newRotation = new Quaternion(-rot[1], rot[2], rot[0], -rot[3]);
        newPosition.x += pos0.x;
        newPosition.z += pos0.z;

        if (teleportRoot)
        {
            Physics.gravity = Vector3.zero;
            rootArticulation.TeleportRoot(newPosition, newRotation);
            rootArticulation.velocity = Vector3.zero;
            rootArticulation.angularVelocity = Vector3.zero;
        }
    }

    private void ResolveCsvJointMap()
    {
        var byName = arts
            .Where(art => art != null && art.jointType == ArticulationJointType.RevoluteJoint)
            .GroupBy(art => art.name, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.First(), StringComparer.OrdinalIgnoreCase);

        var fallback = arts
            .Where(art => art != null && art.jointType == ArticulationJointType.RevoluteJoint)
            .ToList();

        for (int i = 0; i < CsvJointNames.Length; i++)
        {
            if (byName.TryGetValue(CsvJointNames[i], out ArticulationBody joint))
            {
                jh[i] = joint;
                continue;
            }

            jh[i] = i < fallback.Count ? fallback[i] : null;
            Debug.LogWarning($"[OpenLoong] Missing expected joint '{CsvJointNames[i]}'; using traversal fallback at index {i}.");
        }
    }

    private void DumpJointMapping()
    {
        var lines = new System.Text.StringBuilder();
        lines.AppendLine($"[OpenLoongMimicAgent:{name}] CSV joint mapping:");
        for (int i = 0; i < CsvJointNames.Length; i++)
        {
            string unityName = jh[i] != null ? jh[i].name : "<null>";
            lines.AppendLine($"  CSV[{i,2}] {CsvJointNames[i]} -> {unityName}");
        }
        Debug.Log(lines.ToString());
    }

    private string ResolveDatasetPath()
    {
        var candidates = new List<string>();
        if (!string.IsNullOrWhiteSpace(datasetRelativePath))
        {
            candidates.Add(ToAbsoluteProjectPath(datasetRelativePath));
        }

        foreach (string fallback in DefaultDatasetSearchPaths)
        {
            string abs = ToAbsoluteProjectPath(fallback);
            if (!string.IsNullOrWhiteSpace(abs) && !candidates.Contains(abs))
            {
                candidates.Add(abs);
            }
        }

        return candidates.FirstOrDefault(Directory.Exists) ?? string.Empty;
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

    private void SetJointTargetDeg(ArticulationBody joint, float value)
    {
        if (joint == null) return;

        ArticulationDrive drive = joint.xDrive;
        if (clampTargetsToDriveLimits && drive.lowerLimit < drive.upperLimit)
        {
            value = Mathf.Clamp(value, drive.lowerLimit, drive.upperLimit);
        }
        drive.stiffness = jointStiffness;
        drive.damping = jointDamping;
        drive.target = value;
        joint.xDrive = drive;
    }
}
