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

public class X02LiteMimicAgent : Agent, IMimicAgent
{
    public bool train = false;
    public bool replay = false;

    [Header("Multi-Robot Registry")]
    [Tooltip("Robot key used by the WHAM + GMR pipeline. Must match the name in StartInput.")]
    [SerializeField] private string robotKey = "x02lite";

    // X02Lite has 10 DOF (legs only, 5 per leg). The URDF joint order is:
    //   0..4  left  leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch
    //   5..9  right leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch
    // Unity's GetComponentsInChildren<ArticulationBody>() traversal may yield a
    // different order — DumpJointMapping() will print the actual order at startup.

    [Tooltip("If ON, print every Unity revolute joint's GameObject name on Initialize().")]
    [SerializeField] private bool logJointMappingOnStart = true;

    private void DumpJointMapping()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"[X02LiteMimicAgent:{name}] Joint mapping (identity assumed):");
        for (int i = 0; i < 10 && i < jh.Length; i++)
        {
            string jointName = (jh[i] != null) ? jh[i].name : "<null>";
            sb.AppendLine($"  Unity jh[{i,2}] = '{jointName}'   ←  CSV[{i}]");
        }
        sb.AppendLine("Expected URDF order (10 DOF X02Lite):");
        sb.AppendLine("  0..4 left  leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch");
        sb.AppendLine("  5..9 right leg: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch");
        UnityEngine.Debug.Log(sb.ToString());
    }

    float[] uff = new float[10];
    float[] u = new float[10];
    float[] utotal = new float[10];

    public bool useExternalReplayData = false;

    [Tooltip("Primary dataset folder. If missing, the script tries the built-in search list.")]
    [SerializeField] private string datasetRelativePath = "Assets/Gewu/Imitation/dataset/x02lite";

    private static readonly string[] DefaultDatasetSearchPaths =
    {
        "Assets/Gewu/Imitation/dataset/x02lite",
        "Assets/Imitation/dataset/x02lite",
        "Assets/Gewu/Imitation/dataset",
        "Assets/Imitation/dataset",
    };

    private List<float[]> refData = new List<float[]>();
    private List<float[]> itpData = new List<float[]>();

    private int currentFrame;

    // X02Lite CSV: 3 root pos + 4 root quat + 10 DOF = 17 cols
    float[] currentData = new float[17];
    float[] currentPos = new float[3];
    float[] currentRot = new float[4];
    float[] currentDof = new float[10];

    Transform body;

    private float[] restPositions;
    private float[] restVelocities;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();

    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[10];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;
    public int frame0 = 100;

    public float positionKp = 1000f;
    public float positionKd = 50f;
    public float rotationKp = 1000f;
    public float rotationKd = 50f;

    private bool _isClone = false;

    // ── IMimicAgent surface ───────────────────────────────────────────────────
    public string RobotKey => string.IsNullOrWhiteSpace(robotKey) ? "x02lite" : robotKey.Trim();
    public GameObject AgentGameObject => gameObject;
    public bool UseExternalReplayData { get => useExternalReplayData; set => useExternalReplayData = value; }
    public bool ReplayMode { get => replay; set => replay = value; }
    public int MotionId { get; set; }
    public void RequestEndEpisode() => EndEpisode();

    public void ResetToInitialState()
    {
        for (int i = 0; i < 10; i++)
        {
            if (jh[i] != null) SetJointTargetDeg(jh[i], 0f);
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
        }
        currentFrame = frame0;
        tt = 0;
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
        int ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if (arts[k].jointType.ToString() == "RevoluteJoint")
            {
                jh[ActionNum] = arts[k];
                ActionNum++;
            }
        }

        body = arts[0].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        if (restPositions == null)
        {
            pos0 = body.position;
            rot0 = body.rotation;

            art0.GetJointPositions(P0);
            art0.GetJointVelocities(W0);

            restPositions  = P0.ToArray();
            restVelocities = W0.ToArray();
        }

        TryLoadCurrentMotionData(keepProgress: false);

        if (logJointMappingOnStart) DumpJointMapping();

        MimicAgentRegistry.Instance.Register(this);
    }

    List<string> GetCsvFileNames(string directoryPath)
    {
        List<string> csvFiles = new List<string>();
        try
        {
            if (Directory.Exists(directoryPath))
            {
                foreach (string file in Directory.GetFiles(directoryPath))
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

                    if (frameData.Count < 17) continue;
                    if (frameData.Count > 17) frameData = frameData.Take(17).ToList();

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
        string selectedCsv = csvFileNames[mid];

        List<float[]> data = LoadDataFromFile(selectedCsv);
        if (data == null || data.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[X02Lite] CSV contains no valid frame data: " + selectedCsv);
            return false;
        }

        return ApplyReplayData(data, keepProgress);
    }

    private bool ApplyReplayData(List<float[]> data, bool keepProgress)
    {
        if (data == null || data.Count == 0) return false;

        int oldFrame = currentFrame;
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

        currentFrame = keepProgress
            ? Mathf.Clamp(oldFrame, 0, itpData.Count - 1)
            : Mathf.Clamp(frame0, 0, itpData.Count - 1);

        return true;
    }

    private string ResolveDatasetPath()
    {
        List<string> candidates = new List<string>();

        if (!string.IsNullOrWhiteSpace(datasetRelativePath))
            candidates.Add(ToAbsoluteProjectPath(datasetRelativePath));

        foreach (string fallback in DefaultDatasetSearchPaths)
        {
            string abs = ToAbsoluteProjectPath(fallback);
            if (!string.IsNullOrWhiteSpace(abs) && !candidates.Contains(abs))
                candidates.Add(abs);
        }

        foreach (string candidate in candidates)
        {
            if (Directory.Exists(candidate)) return candidate;
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

    // ── safe articulation helpers ─────────────────────────────────────────────

    private readonly List<float> _cacheProbe = new List<float>();

    private int ArticulationCacheSize()
    {
        if (arts == null || arts.Length == 0 || arts[0] == null) return 0;
        _cacheProbe.Clear();
        arts[0].GetJointPositions(_cacheProbe);
        return _cacheProbe.Count;
    }

    private void SafeSetJointPositions(List<float> positions)
    {
        int cacheSize = ArticulationCacheSize();
        List<float> safe = AlignToCache(positions, cacheSize);
        arts[0].SetJointPositions(safe);
    }

    private void SafeSetJointVelocities(List<float> velocities)
    {
        int cacheSize = ArticulationCacheSize();
        List<float> safe = AlignToCache(velocities, cacheSize);
        arts[0].SetJointVelocities(safe);
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
            arts[0].GetJointPositions(_cacheProbe);
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

    // ── episode lifecycle ─────────────────────────────────────────────────────

    public override void OnEpisodeBegin()
    {
        arts[0].immovable       = false;
        arts[0].TeleportRoot(pos0, rot0);
        arts[0].velocity        = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;

        SafeSetJointPositions(new List<float>(restPositions));
        SafeSetJointVelocities(new List<float>(restVelocities));

        for (int i = 0; i < 10; i++) u[i]   = 0;
        for (int i = 0; i < 10; i++) uff[i] = 0;
        currentFrame = frame0;

        if (replay && !useExternalReplayData)
        {
            TryLoadCurrentMotionData(keepProgress: false);
        }

        if (refData == null || refData.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[X02Lite] refData is empty, skip OnEpisodeBegin.");
            return;
        }

        currentFrame = Mathf.Clamp(currentFrame, 0, refData.Count - 1);
        tt = 0;
        currentData = refData[currentFrame];
        Array.Copy(currentData, 0, currentPos, 0, 3);
        Array.Copy(currentData, 3, currentRot, 0, 4);
        Array.Copy(currentData, 7, currentDof, 0, 10);

        for (int i = 0; i < 10; i++)
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

        arts[0].TeleportRoot(newPosition, newRotation);
        arts[0].velocity        = Vector3.zero;
        arts[0].angularVelocity = Vector3.zero;
        arts[0].immovable       = true;
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

        for (int i = 0; i < 10; i++)
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
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        float kb = 50;
        if (replay) kb = 0;
        for (int i = 0; i < 10; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            utotal[i] = kb * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
    }

    void FixedUpdate()
    {
        if (itpData != null && itpData.Count > 0)
        {
            currentFrame = Mathf.Clamp(currentFrame, 0, itpData.Count - 1);
            currentData = itpData[currentFrame];
            Array.Copy(currentData, 0, currentPos, 0, 3);
            Array.Copy(currentData, 3, currentRot, 0, 4);
            Array.Copy(currentData, 7, currentDof, 0, 10);
            for (int i = 0; i < 10; i++) uff[i] = currentDof[i] * 180f / 3.14f;

            newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
            newRotation = new Quaternion(
                -currentRot[1],
                 currentRot[2],
                 currentRot[0],
                -currentRot[3]
            );
            newPosition.x += pos0.x;
            newPosition.z += pos0.z;

            if (replay)
            {
                Physics.gravity = Vector3.zero;
                arts[0].TeleportRoot(newPosition, newRotation);
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

        tt++;
        if (tt > 3)
        {
            arts[0].immovable = false;
            float rot_reward = -0.01f * Quaternion.Angle(body.rotation, newRotation);
            float pos_reward = -1f * (body.position - newPosition).magnitude;

            if (!replay && (Quaternion.Angle(body.rotation, newRotation) > 40f || (body.position - newPosition).magnitude > 0.5f))
            {
                EndEpisode();
            }

            float dof_reward = 0f;
            for (int i = 0; i < 10; i++)
                dof_reward += -0.1f * Mathf.Abs(jh[i].jointPosition[0] - currentDof[i]);

            AddReward(1f + (rot_reward + pos_reward) * 1f + dof_reward);
        }

        if (itpData != null && currentFrame < itpData.Count - 1)
        {
            currentFrame = currentFrame + 1;
        }
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 180f;
        drive.damping = 8f;
        drive.target = x;
        joint.xDrive = drive;
    }

    public override void Heuristic(in ActionBuffers actionsOut) { }
}
