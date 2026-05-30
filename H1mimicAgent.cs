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
using System.Globalization;
#if UNITY_EDITOR
using UnityEditor;
#endif
using Gewu.Imitation;

public class H1mimicAgent : Agent, IMimicAgent
{
    public bool train = false;
    public bool replay = false;

    [Header("Multi-Robot Registry")]
    [Tooltip("Robot key used by the WHAM + GMR pipeline. Must match one of the supported names " +
             "in StartInput (e.g. unitree_h1, unitree_h1_2).")]
    [SerializeField] private string robotKey = "unitree_h1";

    // Joint mapping note: the H1 prefab's GetComponentsInChildren<ArticulationBody>()
    // traversal yields joints in URDF order (left leg → right leg → torso →
    // left arm → right arm), matching what GMR writes to live_motion_h*.csv.
    // So a 1:1 identity mapping (currentDof[i] → jh[i]) is correct, and we
    // deliberately don't expose a SerializedField permutation table (the same
    // mistake we made on G1 — Unity then serialized a wrong table into the
    // scene that survived every code edit). DumpJointMapping below remains so
    // a future prefab variant can be verified.

    [Tooltip("If ON, print every Unity revolute joint's GameObject name on Initialize() so the " +
             "user can verify the H1 prefab traversal still yields URDF order.")]
    [SerializeField] private bool logJointMappingOnStart = true;

    /// <summary>
    /// Dump the Unity-joint-index → joint-name → CSV-index correspondence to
    /// the Console. The GMR pipeline writes the 19 DOFs in URDF order:
    ///   0..4  left  leg (hip_yaw, hip_roll, hip_pitch, knee, ankle)
    ///   5..9  right leg (same suborder)
    ///   10    torso
    ///   11..14 left  arm (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow)
    ///   15..18 right arm (same suborder)
    /// If Unity's traversal doesn't match, the CSV DOF at index i drives the
    /// wrong joint and the motion looks wrong even though the data is clean.
    /// </summary>
    private void DumpJointMapping()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"[H1mimicAgent:{name}] Joint mapping (identity assumed):");
        for (int i = 0; i < 19 && i < jh.Length; i++)
        {
            string jointName = (jh[i] != null) ? jh[i].name : "<null>";
            sb.AppendLine($"  Unity jh[{i,2}] = '{jointName}'   ←  CSV[{i}]");
        }
        sb.AppendLine("Expected URDF order (19 DOF H1):");
        sb.AppendLine("  0..4  left  leg: hip_yaw, hip_roll, hip_pitch, knee, ankle");
        sb.AppendLine("  5..9  right leg: same suborder");
        sb.AppendLine("  10    torso");
        sb.AppendLine("  11..14 left  arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow");
        sb.AppendLine("  15..18 right arm: same suborder");
        sb.AppendLine("If names don't line up, fill in a permutation table or fix the prefab's transform hierarchy.");
        UnityEngine.Debug.Log(sb.ToString());
    }

    [Header("Live CSV (set when retargeted from WHAM+GMR)")]
    public bool useExternalReplayData = false;

    float[] uff = new float[19] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] u = new float[19] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    float[] utotal = new float[19] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,0, 0, 0, 0, 0, 0, 0, 0, 0};
    //int[] trainid=new int[13]{3,9, 5,13,6,14,15,   3,9, 5,13,16,17};
    int[] trainid=new int[13]{5,5,13,13,   6,14,15,   5,5,13,13,   16,17};
    
    private List<string> actionFolders = new List<string>();
    public int motion_idx = 0;
    public string motion_name;
    int idx = 0;

    private string dofFilePath;
    private string rotFilePath;
    private string posFilePath;

    private List<float[]> allDofData = new List<float[]>();
    private List<float[]> allPosData = new List<float[]>();
    private List<float[]> allRotData = new List<float[]>();
    
    private int currentFrame = 0;  
    private bool isEndEpisode = false;
    
    Transform body;

    List<float> P0 = new List<float>();
    List<float> W0 = new List<float>();
    List<Transform> bodypart = new List<Transform>();
    Vector3 pos0;
    Quaternion rot0;
    Quaternion newRotation;
    Vector3 newPosition;
    ArticulationBody[] jh = new ArticulationBody[19];
    ArticulationBody[] arts = new ArticulationBody[40];
    ArticulationBody art0;
    int tt = 0;

    private bool _isClone = false;

    // ── IMimicAgent surface ───────────────────────────────────────────────────
    public string RobotKey => string.IsNullOrWhiteSpace(robotKey) ? "unitree_h1" : robotKey.Trim();
    public GameObject AgentGameObject => gameObject;
    public bool UseExternalReplayData { get => useExternalReplayData; set => useExternalReplayData = value; }
    public bool ReplayMode { get => replay; set => replay = value; }
    public int MotionId { get => motion_idx; set => motion_idx = value; }
    public void RequestEndEpisode() => EndEpisode();

    /// <summary>
    /// Imperative reset of PD targets and replay bookkeeping. See the long
    /// comment on G1mimicAgent.ResetToInitialState for why we deliberately
    /// DO NOT touch the articulation cache (TeleportRoot / SetJointPositions
    /// / SetJointVelocities) here: those operations are unsafe in the same
    /// frame as SetActive(true), because Unity hasn't rebuilt the
    /// ArticulationBody yet and the cache slot ↔ joint mapping is
    /// undefined. The cache restore happens in OnEpisodeBegin on the next
    /// FixedUpdate, after the rebuild is done.
    /// </summary>
    public void ResetToInitialState()
    {
        // Clear every revolute xDrive target so the PD controller can't drag
        // joints toward stale "last replay frame" angles in the one-frame
        // gap between SetActive(true) and OnEpisodeBegin firing. xDrive is a
        // joint-configuration property and writes persist safely even when
        // the body is inactive or mid-rebuild.
        for (int i = 0; i < 19; i++)
        {
            if (jh[i] != null) SetJointTargetDeg(jh[i], 0f);
            u[i] = 0f;
            uff[i] = 0f;
            utotal[i] = 0f;
        }

        currentFrame = 0;
        tt = 0;
        isEndEpisode = false;
    }

    // Registration with the scene-wide registry happens at the end of the
    // ML-Agents Initialize() override below — see the comment in G1mimicAgent
    // for the rationale (we must not shadow Agent.OnEnable).

    /// <summary>
    /// Load a flat 26-column-per-row CSV produced by the WHAM + GMR pipeline
    /// (3 root pos + 4 root quat + 19 DOF) and feed it to the H1 replay path
    /// the same way the on-disk h1_dataset is consumed. Returns true when the
    /// CSV parsed at least one usable frame.
    /// </summary>
    public bool LoadReplayCsvFromPath(string filePath, bool keepProgress)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            UnityEngine.Debug.LogWarning("[H1] LoadReplayCsvFromPath failed: filePath is empty.");
            return false;
        }
        if (!File.Exists(filePath))
        {
            UnityEngine.Debug.LogWarning("[H1] LoadReplayCsvFromPath failed: file does not exist: " + filePath);
            return false;
        }

        const int RootPosCols = 3;
        const int RootRotCols = 4;
        const int DofCols     = 19;
        const int ExpectedCols = RootPosCols + RootRotCols + DofCols; // 26

        var newPos = new List<float[]>();
        var newRot = new List<float[]>();
        var newDof = new List<float[]>();

        try
        {
            using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (StreamReader r = new StreamReader(fs))
            {
                string line;
                while ((line = r.ReadLine()) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] tokens = line.Split(',');
                    if (tokens.Length < ExpectedCols) continue;

                    float[] row = new float[ExpectedCols];
                    bool ok = true;
                    for (int i = 0; i < ExpectedCols; i++)
                    {
                        string t = tokens[i].Trim();
                        if (!float.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out float v) &&
                            !float.TryParse(t, out v))
                        {
                            ok = false; break;
                        }
                        row[i] = v;
                    }
                    if (!ok) continue;

                    float[] pos = new float[RootPosCols]; System.Array.Copy(row, 0, pos, 0, RootPosCols);
                    float[] rot = new float[RootRotCols]; System.Array.Copy(row, RootPosCols, rot, 0, RootRotCols);
                    float[] dof = new float[DofCols];     System.Array.Copy(row, RootPosCols + RootRotCols, dof, 0, DofCols);

                    newPos.Add(pos); newRot.Add(rot); newDof.Add(dof);
                }
            }
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogWarning("[H1] LoadReplayCsvFromPath read error: " + e.Message);
            return false;
        }

        if (newDof.Count == 0)
        {
            UnityEngine.Debug.LogWarning("[H1] LoadReplayCsvFromPath: no usable frames in " + filePath);
            return false;
        }

        int oldFrame = currentFrame;
        allPosData = newPos;
        allRotData = newRot;
        allDofData = newDof;
        motion_name = Path.GetFileNameWithoutExtension(filePath);

        if (keepProgress)
            currentFrame = Mathf.Clamp(oldFrame, 0, allDofData.Count - 1);
        else
            currentFrame = 0;

        return true;
    }

    void Start()
    {
        Time.fixedDeltaTime = 0.02f;

        /*SerializedObject tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset"));
        SerializedProperty layers = tagManager.FindProperty("layers");
        SerializedProperty layer = layers.GetArrayElementAtIndex(15);
        int targetLayer = LayerMask.NameToLayer("robot");
        layer.stringValue = "robot";
        tagManager.ApplyModifiedProperties();
        Physics.IgnoreLayerCollision(15, 15, true);
        ChangeLayerRecursively(gameObject, 15);*/

        if (train && !_isClone) 
        {
            for (int i = 1; i < 24; i++)
            {
                GameObject clone = Instantiate(gameObject); 
                clone.transform.position = transform.position + new Vector3(i * 2f, 0, 0);
                clone.name = $"{name}_Clone_{i}"; 
                clone.GetComponent<H1mimicAgent>()._isClone = true; 
            }
        }
    }

    void ChangeLayerRecursively(GameObject obj, int targetLayer)
    {
        obj.layer = targetLayer;
        foreach (Transform child in obj.transform)ChangeLayerRecursively(child.gameObject, targetLayer);
    }

    // Tracks whether the spawn-pose snapshot has been captured. Without this
    // guard, ML-Agents' Agent.OnEnable re-fires LazyInitialize each time the
    // GameObject is SetActive(true), and the second-and-later captures grab
    // whatever junk was in the articulation cache from the previous replay
    // frame — silently corrupting P0/W0 for every subsequent OnEpisodeBegin.
    private bool _restPoseCaptured = false;

    public override void Initialize()
    {

        arts = this.GetComponentsInChildren<ArticulationBody>();
        int ActionNum = 0;
        for (int k = 0; k < arts.Length; k++)
        {
            if(arts[k].jointType.ToString() == "RevoluteJoint")
            {
                jh[ActionNum] = arts[k];
                ActionNum++;
            }
        }
        body = arts[0].GetComponent<Transform>();
        art0 = body.GetComponent<ArticulationBody>();

        // ── CRITICAL: capture pos0/rot0/P0/W0 only ONCE. See the long comment
        // on G1mimicAgent.Initialize for the full explanation. TL;DR:
        // ML-Agents re-triggers Initialize on every SetActive(true), and the
        // second-and-later calls capture corrupted (post-replay) cache state
        // unless we guard against it.
        if (!_restPoseCaptured)
        {
            _restPoseCaptured = true;
            pos0 = body.position;
            rot0 = body.rotation;
            art0.GetJointPositions(P0);
            art0.GetJointVelocities(W0);
        }
                
        string streamingAssetsPath = Path.Combine(Application.streamingAssetsPath, "h1_dataset");
        LoadActionFolders(streamingAssetsPath);
        if (actionFolders != null && actionFolders.Count > 1)
        {
            print(actionFolders[1]);
            LoadDataForAction(actionFolders[1]);
        }

        // Print Unity joint name → CSV index correspondence so the user can
        // verify the H1 prefab's joint hierarchy actually matches URDF order.
        if (logJointMappingOnStart) DumpJointMapping();

        // Register with the scene-wide IMimicAgent registry — see G1mimicAgent
        // for why this happens here rather than in OnEnable.
        MimicAgentRegistry.Instance.Register(this);
    }
    void LoadActionFolders(string basePath)
    {
        string[] folders = Directory.GetDirectories(basePath);
        actionFolders = folders.ToList();  
    }
    void LoadDataForAction(string actionFolder)
    {
        dofFilePath = Path.Combine(actionFolder, "dof.csv");
        rotFilePath = Path.Combine(actionFolder, "root_rot.csv");
        posFilePath = Path.Combine(actionFolder, "root_trans_offset.csv");
        allDofData = LoadDataFromFile(dofFilePath);
        allRotData = LoadDataFromFile(rotFilePath);
        allPosData = LoadDataFromFile(posFilePath);
    }

    List<float[]> LoadDataFromFile(string filePath)
    {
        List<float[]> dataList = new List<float[]>();
        try
        {
            string[] lines = File.ReadAllLines(filePath);
            foreach (string line in lines)
            {
                string[] values = line.Split(',');
                List<float> frameData = new List<float>();
                foreach (string value in values)
                {
                    if (float.TryParse(value.Trim(), out float parsedValue))frameData.Add(parsedValue);
                }
                dataList.Add(frameData.ToArray());
            }
        }
        catch (System.Exception e)
        {
            print("Error loading data from file " + filePath + ": " + e.Message);
        }

        return dataList;
    }
    // Per-instance scratch buffer used to probe the actual articulation cache
    // size before calling SetJointPositions / SetJointVelocities. dofCount can
    // be out of sync with the engine right after immovable toggles, so we use
    // GetJointPositions(probe).Count as the authoritative size — same fix as
    // G1mimicAgent.ArticulationCacheSize.
    private readonly List<float> _cacheProbe = new List<float>();

    private void SafeSetJointPositions(List<float> positions)
    {
        if (arts == null || arts.Length == 0 || arts[0] == null) return;
        _cacheProbe.Clear();
        arts[0].GetJointPositions(_cacheProbe);
        int target = _cacheProbe.Count;
        if (target <= 0) return;

        List<float> safe = AlignToCache(positions, target);
        arts[0].SetJointPositions(safe);
    }

    private void SafeSetJointVelocities(List<float> velocities)
    {
        if (arts == null || arts.Length == 0 || arts[0] == null) return;
        _cacheProbe.Clear();
        arts[0].GetJointPositions(_cacheProbe);
        int target = _cacheProbe.Count;
        if (target <= 0) return;

        List<float> safe = AlignToCache(velocities, target);
        arts[0].SetJointVelocities(safe);
    }

    /// <summary>
    /// Reshape <paramref name="source"/> so that joint-slot alignment is
    /// preserved when the articulation root's immovable toggle has shifted
    /// the cache by 6 DOFs. See G1mimicAgent.AlignToCache for the full
    /// rationale — same root-cause bug, same fix.
    /// </summary>
    private List<float> AlignToCache(List<float> source, int cacheSize)
    {
        if (source == null) source = new List<float>();
        if (source.Count == cacheSize) return source;

        const int RootDofCount = 6;

        if (source.Count == cacheSize + RootDofCount)
        {
            // Source captured at immovable=false (root present); cache is
            // immovable=true (root stripped). Take the joint tail.
            return source.GetRange(RootDofCount, cacheSize);
        }

        if (source.Count + RootDofCount == cacheSize)
        {
            // Source is joint-only, cache includes root. Preserve current
            // root values, append source joints.
            var safe = new List<float>(cacheSize);
            for (int i = 0; i < RootDofCount && i < _cacheProbe.Count; i++)
                safe.Add(_cacheProbe[i]);
            while (safe.Count < RootDofCount) safe.Add(0f);
            safe.AddRange(source);
            return safe;
        }

        UnityEngine.Debug.LogWarning(
            $"[H1mimicAgent] Unexpected cache mismatch: source={source.Count}, cache={cacheSize}. " +
            "Falling back to trim/pad — joint values may be misaligned.");
        var result = new List<float>(cacheSize);
        for (int i = 0; i < cacheSize; i++)
            result.Add(i < source.Count ? source[i] : 0f);
        return result;
    }

    public override void OnEpisodeBegin()
    {
        // ── CRITICAL: clear immovable BEFORE cache writes (see the long comment
        // in G1mimicAgent.OnEpisodeBegin). Previous OnEpisodeBegin set
        // arts[0].immovable = true to keep root pinned during replay; that
        // shrinks the articulation cache to joint-only size. If we then write
        // P0 (captured at Initialize with immovable=false, so includes 6 root
        // slots), EnsureListSize trims off the joint tail and shifts every
        // surviving value 6 slots left → joints get root values, arms end up
        // twisted into the body. Restoring immovable=false here resizes the
        // cache to match P0 before the SetJointPositions call.
        art0.immovable = false;
        art0.TeleportRoot(pos0, rot0);
        art0.velocity = Vector3.zero;
        art0.angularVelocity = Vector3.zero;
        SafeSetJointPositions(P0);
        SafeSetJointVelocities(W0);
        for (int i = 0; i < 19; i++) u[i] = 0;
        for (int i = 0; i < 19; i++) uff[i] = 0;

        // Live retargeting (StartInput) and replay-from-disk (Replay button) both
        // populate the motion buffers externally before invoking OnEpisodeBegin;
        // in those cases the on-disk h1_dataset reload below must be skipped or
        // it would clobber the user-picked motion. Only fall back to the
        // training-cycle behaviour (trainid[idx] → actionFolders) when this is
        // a true training episode.
        if (!useExternalReplayData && !replay)
        {
            motion_idx = trainid[idx];
            if (actionFolders != null && actionFolders.Count > 0)
            {
                int clamped = Mathf.Clamp(motion_idx, 0, actionFolders.Count - 1);
                LoadDataForAction(actionFolders[clamped]);
                motion_name = actionFolders[clamped].Replace("./Assets/Imitation/H1/dataset\\", "");
            }
        }
        else if (!useExternalReplayData && replay)
        {
            // Replay clicked without an external CSV load: honour the
            // motion_idx that was set externally (Replay.cs) and pull that
            // specific h1_dataset folder. Don't advance the training cycle.
            if (actionFolders != null && actionFolders.Count > 0)
            {
                int clamped = Mathf.Clamp(motion_idx, 0, actionFolders.Count - 1);
                LoadDataForAction(actionFolders[clamped]);
                motion_name = actionFolders[clamped];
            }
        }

        currentFrame = 0;
        // Advance the training-cycle pointer only on actual training episodes;
        // replay clicks must not change which motion training would next see.
        if(!train && !replay && !useExternalReplayData)
        {
            idx++;
            if(idx==13)idx=0;
        }
        if(train)idx=(int)Mathf.Floor(((Time.time%3900)/300));//train for 10 million steps first
        //if(train)idx=(int)Mathf.Floor(((Time.time%390)/30));//train for 2 million steps after

        tt=0;
        // Guard against empty data — e.g. live retargeting before the first
        // CSV frame has arrived. The original code accessed index [1]
        // unconditionally; pick a safe in-range index instead.
        if (allDofData == null || allDofData.Count == 0 ||
            allPosData == null || allPosData.Count == 0 ||
            allRotData == null || allRotData.Count == 0)
        {
            return;
        }
        int seedFrame = Mathf.Min(1, allDofData.Count - 1);
        float[] currentDof = allDofData[seedFrame];
        float[] currentPos = allPosData[Mathf.Min(seedFrame, allPosData.Count - 1)];
        float[] currentRot = allRotData[Mathf.Min(seedFrame, allRotData.Count - 1)];
        Vector3 newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        Quaternion newRotation = new Quaternion(
            currentRot[1], 
            -currentRot[2], 
            currentRot[0], 
            currentRot[3]
        );
        newPosition.x+=pos0.x;
        newPosition.z+=pos0.z;
        
        art0.TeleportRoot(newPosition, newRotation);
        art0.velocity = Vector3.zero;
        art0.angularVelocity = Vector3.zero;
        art0.immovable = true;
        for (int i = 0; i < 19; i++)
        {
            uff[i] = currentDof[i]* 180f / 3.14f;
            SetJointTargetDeg(jh[i], uff[i]);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Defensive shape: CollectObservations is invoked from Agent.OnDisable
        // → NotifyAgentDone when the GameObject is SetActive(false). At that
        // moment the ArticulationBody's ReducedSpace caches have dofCount==0
        // and indexing them throws. We must still emit the same NUMBER of
        // observations or VectorSensor's shape check trips, so substitute
        // zeros when the cache slot isn't simulated.
        if (body != null) sensor.AddObservation(EulerTrans(body.eulerAngles[0]) * 3.14f / 180f);
        else              sensor.AddObservation(0f);
        if (body != null) sensor.AddObservation(EulerTrans(body.eulerAngles[2]) * 3.14f / 180f);
        else              sensor.AddObservation(0f);
        if (body != null && art0 != null)
        {
            sensor.AddObservation(body.InverseTransformDirection(art0.velocity));
            sensor.AddObservation(body.InverseTransformDirection(art0.angularVelocity));
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(Vector3.zero);
        }

        for (int i = 0; i < 19; i++)
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
        if (eulerAngle <= 180)
            return eulerAngle;
        else
            return eulerAngle - 360f;
    }
    
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var continuousActions = actionBuffers.ContinuousActions;
        var kk = 0.9f;
        
        float kb = 30;
        if(replay)kb = 0;
        for (int i = 0; i < 19; i++)
        {
            u[i] = u[i] * kk + (1 - kk) * continuousActions[i];
            utotal[i] = kb * u[i] + uff[i];
            SetJointTargetDeg(jh[i], utotal[i]);
        }
       
    }

    void FixedUpdate()
    {
	///////////////feedforward///////////////////////////////////////////////////////////
	if (allDofData.Count > 0)
	{
		float[] currentDof = allDofData[currentFrame];
		float[] currentPos = allPosData[currentFrame];
        float[] currentRot = allRotData[currentFrame];
		for (int i = 0; i < 19; i++)uff[i] = currentDof[i]* 180f / 3.14f;

        Quaternion gymQuat = new Quaternion(
            currentRot[0], 
            currentRot[1], 
            currentRot[2], 
            currentRot[3]
        );
        Quaternion conversionQ = new Quaternion(0.5f, -0.5f, -0.5f, 0.5f);
        newRotation = conversionQ * gymQuat * Quaternion.Inverse(conversionQ);
		newPosition = new Vector3(-currentPos[1], currentPos[2], currentPos[0]);
        newRotation = new Quaternion(
            currentRot[1], 
            -currentRot[2], 
            currentRot[0], 
            currentRot[3]
        );

        newPosition.x+=pos0.x;
        newPosition.z+=pos0.z;

    	if(replay)
    	{
    		// Plain TeleportRoot — same minimal path the working G1 baseline
    		// uses. Earlier this branch also refreshed every xDrive every
    		// FixedUpdate and did a kinematic snap; that combination corrupts
    		// the articulation cache and was the root cause of the persistent
    		// H1 misalignment. xDrive targets are set in OnEpisodeBegin and
    		// OnActionReceived (replay mode forces kb=0 so utotal=uff), which
    		// is enough as long as the DecisionRequester period is 1.
    		Physics.gravity = Vector3.zero;
    		art0.TeleportRoot(newPosition, newRotation);
    	}
   		currentFrame = (currentFrame + 1) % allDofData.Count;

		if (currentFrame == 0)  // wrapped past last frame
		{
			// Don't EndEpisode in external (live) replay — the live CSV is
			// authoritative and EndEpisode would wipe it; just loop the
			// playhead and let the next CSV poll bring in fresh data.
			if (!useExternalReplayData) EndEpisode();
			currentFrame = 0;
		}
		    
		if (isEndEpisode)
		{
	            currentFrame = 0;
	            isEndEpisode = false; 
		}
	}
	
	/////////////////rewards/////////////////////////////////////////////////////////
        tt++;
        var vel = body.InverseTransformDirection(art0.velocity);
        var wel = body.InverseTransformDirection(art0.angularVelocity);
        
        var live_reward = 1f;
        float rot_reward = 0;
        float pos_reward = 0;
        if(tt>3)
        {
            art0.immovable = false;
            rot_reward = - 0.01f * Quaternion.Angle(body.rotation, newRotation);
            pos_reward = - 1f * (body.position - newPosition).magnitude;
            if (Quaternion.Angle(body.rotation, newRotation)>30f || (body.position - newPosition).magnitude>0.3f)EndEpisode();
        }  
        var reward = live_reward + (rot_reward + pos_reward)*1;
        AddReward(reward);
    
    }

    void SetJointTargetDeg(ArticulationBody joint, float x)
    {
        var drive = joint.xDrive;
        drive.stiffness = 2000f;//180f;
        drive.damping = 200f;//8f;
        drive.forceLimit = 300f;//250f;// 33.5f;// 50f;
        drive.target = x;
        joint.xDrive = drive;
    }

    // KinematicApplyRevoluteRadians removed. Earlier we did a per-FixedUpdate
    // kinematic cache write on top of the PD drive in replay mode, modeled on
    // an unsuccessful experiment with G1. That combination ends up corrupting
    // the articulation cache (the PD drive and the cache write fight each
    // other) and was the source of the visible H1 misalignment. Replay now
    // uses plain TeleportRoot + PD targets set once per OnEpisodeBegin /
    // OnActionReceived, same as the working G1 baseline.

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        
    }
    
}



