using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Gewu.Imitation;

using Debug = UnityEngine.Debug;

public class StartInput : MonoBehaviour
{
    [Header("Bash Command")]
    [SerializeField] private bool runBashOnClick = true;
    [Tooltip("Executable to run. Default: powershell on Windows. Prefix args are auto-split (e.g. \"powershell -ExecutionPolicy Bypass -File\").")]
    [SerializeField] private string bashExecutable = "powershell -ExecutionPolicy Bypass -File";
    [Tooltip("Script to run. run.ps1 (PowerShell) reads all config from environment variables.")]
    [SerializeField] private string bashScriptPath = "run.ps1";
    [TextArea(2, 6)]
    [SerializeField] private string bashArguments = "";
    [Tooltip("WHAM repo root (= Robot-imitation-learning/). Use an absolute path, and all relative paths below will be resolved from here.")]
    [SerializeField] private string bashWorkingDirectory = "Assets/Gewu/Imitation/Robot-imitation-learning/";
    [SerializeField] private bool restartBashIfRunning = false;
    [SerializeField] private bool logBashOutput = true;

    [Header("Runtime Env")]
    [SerializeField] private string outputRoot = "output/my_run";
    [SerializeField] private string outputCsvFileName = "csv/live_motion.csv";
    [SerializeField] private string defaultRobotName = "unitree_g1";
    [SerializeField] private bool recordGmrVideo = true;
    [SerializeField] private bool recordWhamVideo = true;
    [SerializeField] private bool track = true;  // Alias for --camera_follow
    [SerializeField] private string videoPath = "examples/IMG_9732.mov";

    [Header("WHAM Performance")]
    [Tooltip("Enable AMP (mixed precision) inference for WHAM (CUDA only).")]
    [SerializeField] private bool whamUseAmp = false;
    [Tooltip("Run full detection every N frames; intermediate frames reuse tracking result.")]
    [SerializeField] private int whamDetectInterval = 1;
    [Tooltip("Run full WHAM inference every N frames; intermediate frames reuse last result.")]
    [SerializeField] private int whamInferInterval = 1;
    [Tooltip("WHAM temporal window length (frames). Smaller = faster, less stable.")]
    [SerializeField] private int whamStreamSeqLen = 16;
    [Tooltip("WHAM input resize scale (0.1–1.0). Smaller = faster.")]
    [Range(0.1f, 1.0f)]
    [SerializeField] private float whamInputScale = 1.0f;
    [Tooltip("GMR postprocessing torch device (cpu / cuda / auto).")]
    [SerializeField] private string gmrTorchDevice = "cpu";

    [Header("CSV Lifecycle")]
    [Tooltip("Clear live_motion.csv before launching the bash process to avoid replaying stale data from a previous run.")]
    [SerializeField] public bool clearCsvOnStart = true;

    [Tooltip("Clear live_motion.csv when the application exits or the bash process is stopped. " +
             "Enable this to ensure the file is never left in a partially-written (corrupted) state " +
             "that would confuse the next startup.")]
    [SerializeField] public bool clearCsvOnExit = true;

    [Header("RoboList")]
    [SerializeField] private TMP_Dropdown roboListDropdown;
    [SerializeField] private FileBrowser roboListFileBrowser;
    [SerializeField] private string roboListObjectName = "RoboList";

    [Tooltip("When the RoboList dropdown selection changes at runtime, stop the running " +
             "WHAM/GMR pipeline (the CSV format is robot-specific so it must be restarted) " +
             "and switch the live retargeting target to the newly selected robot. The user " +
             "then presses Start again to relaunch the pipeline with the new ROBOT env var.")]
    [SerializeField] private bool switchActiveRobotOnDropdownChange = true;

    [Tooltip("If true, also auto-press Start after the robot switch so the user doesn't have " +
             "to. Off by default because relaunching the Python pipeline is heavy and you " +
             "usually want to confirm before paying that cost.")]
    [SerializeField] private bool autoStartOnRobotSwitch = false;

    [Tooltip("When ON, only the selected robot's GameObject stays active in the scene; the " +
             "others get SetActive(false). Lets the dropdown act as a visual switch even " +
             "when a robot has no IMimicAgent attached yet. Resolution order: 1) match in " +
             "sceneRobots list, 2) RobotKey of a registered IMimicAgent.")]
    [SerializeField] private bool hideInactiveRobotsOnSwitch = true;

    [System.Serializable]
    public class RobotSceneEntry
    {
        [Tooltip("Dropdown label or robot key this entry matches against (case-insensitive). " +
                 "Examples: \"G1\", \"unitree_g1\", \"H1\", \"X02Lite\", \"openloong\". " +
                 "If multiple labels should map to the same GameObject, add multiple entries.")]
        public string label;
        [Tooltip("Scene GameObject to keep active when this label is selected. Usually the " +
                 "robot's root transform.")]
        public GameObject robotRoot;
    }

    [Tooltip("Manual mapping from RoboList labels to scene GameObjects. Lets you switch " +
             "visibility of robots that don't have an IMimicAgent yet (e.g. X02Lite, openloong) " +
             "and overrides the registry's lookup when both are present. Leave empty to rely " +
             "entirely on the IMimicAgent registry.")]
    [SerializeField] private List<RobotSceneEntry> sceneRobots = new List<RobotSceneEntry>();

    [Header("Realtime CSV Replay")]
    [SerializeField] private bool monitorCsvOnClick = true;
    [SerializeField] private float csvPollInterval = 0.1f;
    [SerializeField] private bool restartEpisodeOnFirstCsv = true;
    [SerializeField] private bool keepReplayProgressOnCsvUpdate = true;

    [Tooltip("Optional explicit target. When left empty, the agent matching the currently selected " +
             "robot in the RoboList dropdown is looked up from MimicAgentRegistry at runtime.")]
    [SerializeField] private MonoBehaviour targetAgentBehaviour; // must implement IMimicAgent
    private IMimicAgent targetAgent;
    private string lastResolvedRobotKey = string.Empty;

    private Button startButton;
    private bool addedRuntimeListener;
    private Process pythonProcess;
    private Coroutine monitorCoroutine;
    private long lastCsvLength = -1;
    private System.DateTime lastCsvWriteTimeUtc = System.DateTime.MinValue;
    private bool replayBootstrapped;
    private string resolvedCsvPath = string.Empty;
    private string resolvedOutputRootPath = string.Empty;
    private bool csvMissingLogged;

    private static readonly HashSet<string> SupportedRobotNames = new HashSet<string>
    {
        "unitree_g1",
        "unitree_g1_with_hands",
        "unitree_h1",
        "unitree_h1_2",
        "booster_t1",
        "booster_t1_29dof",
        "stanford_toddy",
        "fourier_n1",
        "engineai_pm01",
        "kuavo_s45",
        "hightorque_hi",
        "galaxea_r1pro",
        "berkeley_humanoid_lite",
        "booster_k1",
        "pnd_adam_lite",
        "openloong",
        "tienkung",
        "fourier_gr3",
    };

    private static readonly Dictionary<string, string> RobotAliases = new Dictionary<string, string>(System.StringComparer.OrdinalIgnoreCase)
    {
        { "fourier_gr3v2_1_1", "fourier_gr3" },
        // Short labels people commonly use in UI dropdowns.
        { "G1",   "unitree_g1" },
        { "G1H",  "unitree_g1_with_hands" },
        { "H1",   "unitree_h1" },
        { "H1_2", "unitree_h1_2" },
        { "T1",   "booster_t1" },
    };

    public string BashWorkingDirectory
    {
        get => bashWorkingDirectory;
        set => bashWorkingDirectory = value ?? string.Empty;
    }

    void Awake()
    {
        startButton = GetComponent<Button>();
        if (startButton != null && !HasPersistentStartHandler(startButton))
        {
            startButton.onClick.AddListener(OnStartButtonClicked);
            addedRuntimeListener = true;
        }

        // Attach onValueChanged on the RoboList dropdown so switching the
        // selection mid-session retargets to the newly chosen robot.
        ResolveRoboListReferences();
        if (roboListDropdown != null && switchActiveRobotOnDropdownChange)
        {
            roboListDropdown.onValueChanged.RemoveListener(OnRoboListChanged);
            roboListDropdown.onValueChanged.AddListener(OnRoboListChanged);
        }
    }

    void OnDestroy()
    {
        if (startButton != null && addedRuntimeListener)
        {
            startButton.onClick.RemoveListener(OnStartButtonClicked);
        }

        if (roboListDropdown != null)
        {
            roboListDropdown.onValueChanged.RemoveListener(OnRoboListChanged);
        }

        StopCsvMonitor();
        StopBashProcess();

        if (clearCsvOnExit)
        {
            TryClearCsv("OnDestroy");
        }
    }

    // -------------------------------------------------------------------------
    // RoboList live-switching
    // -------------------------------------------------------------------------

    /// <summary>
    /// Called whenever the user picks a new entry in the RoboList dropdown.
    /// Stops the running pipeline (CSV format is robot-specific, so the
    /// Python side has to be relaunched), resets every registered IMimicAgent
    /// to a clean state, and clears <c>lastResolvedRobotKey</c> so the next
    /// <c>ResolveActiveAgent</c> call logs the switch.
    /// </summary>
    private void OnRoboListChanged(int newIndex)
    {
        if (!switchActiveRobotOnDropdownChange) return;
        if (roboListDropdown == null || roboListDropdown.options == null) return;
        if (newIndex < 0 || newIndex >= roboListDropdown.options.Count) return;

        string newLabel = roboListDropdown.options[newIndex].text;
        string newKey   = TryResolveRobotKeyQuiet(newLabel);
        if (string.IsNullOrWhiteSpace(newKey))
        {
            // Label is not a WHAM/GMR-supported robot key. We still apply the
            // visual switch (so the user can use the dropdown as a pure
            // visibility toggle for robots like X02Lite that don't have an
            // IMimicAgent / aren't supported by the Python pipeline yet) and
            // bail out — without touching the running pipeline.
            Debug.LogWarning($"[StartInput] '{newLabel}' 不在 WHAM/GMR 支持列表，按纯可见性切换处理（不停管线、不复位）。" +
                             $"\n支持列表: {string.Join(", ", SupportedRobotNames)}");
            ApplyRobotVisibility(newLabel, newLabel);
            return;
        }

        Debug.Log($"[StartInput] RoboList 已切换到 '{newLabel}' → ROBOT={newKey}。" +
                  (pythonProcess != null ? " 正在停止当前 WHAM/GMR 流水线..." : ""));

        // 1) Stop the running pipeline + CSV monitor (if any).
        bool wasRunning = pythonProcess != null && !pythonProcess.HasExited;
        StopStartPipeline();

        // 2) Restore Physics.gravity. G1mimicAgent.FixedUpdate's replay branch
        //    sets `Physics.gravity = Vector3.zero` (so the kinematic root
        //    teleport doesn't fight gravity), but that's a GLOBAL setting —
        //    leaving it zeroed when we switch to another robot makes that
        //    robot float weirdly. Reset to standard Earth gravity here so the
        //    newly-activated robot starts in a clean physics environment.
        Physics.gravity = new Vector3(0f, -9.81f, 0f);

        // 3) Invalidate cached resolution so the next ResolveActiveAgent picks
        //    the new key freshly and logs the change.
        lastResolvedRobotKey = string.Empty;
        targetAgent = null;
        replayBootstrapped = false;

        // 4) Visual switch FIRST. The original order (reset → SetActive(false))
        //    queued OnEpisodeBegin via RequestEndEpisode but the agent then got
        //    deactivated before ML-Agents could fire it — so the soon-to-be-
        //    hidden robot's reset never ran, and the next time it became
        //    visible it picked up frozen state ("hands reversed" symptom).
        //    Switching visibility first means only currently-active agents
        //    take the reset, which is exactly what we need.
        ApplyRobotVisibility(newLabel, newKey);

        // 5) HARD-RESET every registered agent imperatively via
        //    ResetToInitialState. Only act on the agent matching the newly-
        //    selected RobotKey; the hidden ones had ReplayMode/UseExternalReplayData
        //    cleared in ApplyRobotVisibility above and we deliberately leave
        //    them alone so we don't restart their articulation simulation.
        if (MimicAgentRegistry.Instance != null)
        {
            IMimicAgent selected = MimicAgentRegistry.Instance.FindByKey(newKey);
            if (selected != null && selected.AgentGameObject != null)
            {
                selected.UseExternalReplayData = false;
                selected.ReplayMode = true;

                try { selected.ResetToInitialState(); }
                catch (System.Exception e)
                {
                    Debug.LogWarning($"[StartInput] {selected.RobotKey} ResetToInitialState threw: {e.Message}");
                }

                selected.RequestEndEpisode();
                Debug.Log($"[StartInput] 选中 '{selected.RobotKey}'：已复位 articulation，已排队 OnEpisodeBegin。");
            }
            else
            {
                Debug.Log($"[StartInput] 没有 RobotKey='{newKey}' 的 IMimicAgent，跳过 articulation 复位（纯可见性切换）。");
            }
        }

        // 5) Pre-verify that an agent with this RobotKey exists in the scene.
        //    If not, warn the user — the live CSV will fall back to the first
        //    registered robot (probably G1).
        if (MimicAgentRegistry.Instance != null && MimicAgentRegistry.Instance.FindByKey(newKey) == null)
        {
            Debug.LogWarning($"[StartInput] 场景里没有 RobotKey='{newKey}' 的 IMimicAgent。" +
                             $"\n• 如果该机器人在 sceneRobots 映射表里，只切了可见性，没有数据流。" +
                             $"\n• 否则按 Start 后实时 CSV 会被路由到第一个已注册机器人（通常是 G1）。" +
                             $"\n要让它真正接收实时数据，给它挂上对应的 *mimicAgent 脚本并把 RobotKey 设成 '{newKey}'。");
        }

        // 6) Optionally auto-relaunch the pipeline with the new ROBOT env var.
        if (autoStartOnRobotSwitch && wasRunning)
        {
            Debug.Log("[StartInput] autoStartOnRobotSwitch=true，自动重启流水线。");
            OnStartButtonClicked();
        }
    }

    // Cached child renderers per robot root. We disable/enable Renderer
    // components instead of toggling GameObject.SetActive because SetActive
    // on a hierarchy that contains ArticulationBody bodies forces Unity to
    // tear down and rebuild the articulation simulation. The rebuild seeds
    // from the LAST cache state (which can be a mid-replay frame) rather
    // than the prefab bind pose, and the C# side ML-Agents Agent ends up
    // re-running Initialize against a partially-rebuilt articulation. Both
    // effects compound across switches — see imitation_robot_switch_ordering
    // and articulation_cache_probe memories. The user confirmed even
    // X02Lite (no script) is corrupted by SetActive cycles, ruling out
    // any agent-side fix and pointing at engine behaviour.
    //
    // Renderer-based hiding leaves the GameObject active throughout: the
    // articulation never rebuilds, agents never re-Initialize, cache values
    // never shift, and the visibility toggle has zero physics side-effects.
    // Cached per-root so we don't allocate every dropdown change.
    private readonly Dictionary<GameObject, Renderer[]> _cachedRobotRenderers =
        new Dictionary<GameObject, Renderer[]>();

    private Renderer[] GetOrCacheRenderers(GameObject root)
    {
        if (root == null) return System.Array.Empty<Renderer>();
        if (_cachedRobotRenderers.TryGetValue(root, out var cached) && cached != null) return cached;
        var fresh = root.GetComponentsInChildren<Renderer>(includeInactive: true);
        _cachedRobotRenderers[root] = fresh;
        return fresh;
    }

    /// <summary>
    /// Toggle visibility of the registered/listed robots by enabling or
    /// disabling their <see cref="Renderer"/> components. The matching
    /// robot becomes visible (renderers on); every other tracked robot
    /// becomes invisible (renderers off). GameObjects stay active in the
    /// scene so their ArticulationBody simulation, ML-Agents Agent
    /// initialization, and replay logic are NOT disturbed — which is the
    /// whole point of switching off SetActive (see the long comment on
    /// _cachedRobotRenderers above).
    ///
    /// Side-effect: also pauses the replay loop on hidden IMimicAgent
    /// instances by clearing their ReplayMode and UseExternalReplayData,
    /// so they don't keep running TeleportRoot every FixedUpdate while
    /// invisible. The newly-visible robot's replay state is set up by the
    /// rest of OnRoboListChanged after this function returns.
    ///
    /// Resolution order: (1) sceneRobots Inspector mapping (matches against
    /// label OR RobotKey), (2) registered IMimicAgent.AgentGameObject.
    /// Robots that are neither listed nor registered are left untouched.
    /// </summary>
    private void ApplyRobotVisibility(string selectedLabel, string selectedRobotKey)
    {
        if (!hideInactiveRobotsOnSwitch) return;

        // Collect every (GameObject, isSelected) pair we know about.
        // Using a dictionary keyed by GameObject so a robot listed in BOTH
        // sceneRobots and the registry is processed only once.
        var roster = new Dictionary<GameObject, bool>();

        // 1) sceneRobots — match against either label or robot key.
        if (sceneRobots != null)
        {
            foreach (RobotSceneEntry entry in sceneRobots)
            {
                if (entry == null || entry.robotRoot == null) continue;
                string entryLabel = (entry.label ?? string.Empty).Trim();

                bool match =
                    !string.IsNullOrEmpty(entryLabel) && (
                        string.Equals(entryLabel, selectedLabel, System.StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(entryLabel, selectedRobotKey, System.StringComparison.OrdinalIgnoreCase));

                // Or match by the resolved key going through the alias table
                // (so "G1" label still matches a "unitree_g1" entry).
                if (!match && !string.IsNullOrEmpty(entryLabel))
                {
                    string entryKey = TryResolveRobotKeyQuiet(entryLabel);
                    if (!string.IsNullOrEmpty(entryKey) &&
                        string.Equals(entryKey, selectedRobotKey, System.StringComparison.OrdinalIgnoreCase))
                    {
                        match = true;
                    }
                }

                if (!roster.ContainsKey(entry.robotRoot) || match)
                    roster[entry.robotRoot] = match;
            }
        }

        // 2) Registered IMimicAgents — match against RobotKey.
        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                GameObject go = agent.AgentGameObject;
                if (go == null) continue;
                bool match = string.Equals(agent.RobotKey, selectedRobotKey, System.StringComparison.OrdinalIgnoreCase);
                if (!roster.ContainsKey(go) || match)
                    roster[go] = match;
            }
        }

        if (roster.Count == 0)
        {
            Debug.LogWarning("[StartInput] 没有可见性映射可用（sceneRobots 为空且没有 IMimicAgent 注册）—— 跳过可见性切换。");
            return;
        }

        // Build a set of agent GameObjects so we can also pause the matching
        // IMimicAgent's replay loop when its robot is being hidden.
        var agentsByGo = new Dictionary<GameObject, IMimicAgent>();
        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent != null && agent.AgentGameObject != null)
                    agentsByGo[agent.AgentGameObject] = agent;
            }
        }

        int shown = 0, hidden = 0, bootstrapped = 0;
        foreach (var kv in roster)
        {
            GameObject root = kv.Key;
            bool isSelected = kv.Value;
            if (root == null) continue;

            // ── One-time SetActive(true) bootstrap for never-shown robots ─
            // If the scene shipped this robot SetActive(false) (typical for
            // H1, X02Lite, etc. when G1 is the default), it has never run
            // Awake / Initialize, so neither its IMimicAgent script nor its
            // Renderer components are alive — Renderer-disable on a
            // never-activated hierarchy can't make it visible, and a
            // toggle-active mode change isn't possible because activeSelf
            // is false. Activate it exactly once here when the user first
            // selects it. Crucially, we NEVER SetActive(false) anything in
            // this function — so the full SetActive(false)→(true) cycle
            // that corrupts ArticulationBody state can never occur. Each
            // robot's first (and only) SetActive(true) runs Initialize on a
            // clean bind pose; afterwards Renderer.enabled handles visibility.
            if (isSelected && !root.activeSelf)
            {
                root.SetActive(true);
                bootstrapped++;
                // Invalidate the renderer cache for this root — the Renderer
                // components were null before activation, so the cached
                // array (if any) is full of nulls. Rebuild now.
                _cachedRobotRenderers.Remove(root);
            }

            // Renderer toggle — visibility-only, no SetActive(false) on the
            // hierarchy. Leaving GameObjects active is what prevents the
            // ArticulationBody rebuild that corrupted joint poses on every
            // dropdown switch in the SetActive-based implementation.
            Renderer[] renderers = GetOrCacheRenderers(root);
            for (int i = 0; i < renderers.Length; i++)
            {
                if (renderers[i] != null) renderers[i].enabled = isSelected;
            }

            // Pause hidden agents' replay loop so they don't keep
            // TeleportRoot-ing themselves around while invisible. The
            // newly-visible robot's replay state is set fresh later in
            // OnRoboListChanged (RequestEndEpisode → OnEpisodeBegin).
            if (!isSelected && agentsByGo.TryGetValue(root, out IMimicAgent hiddenAgent))
            {
                hiddenAgent.ReplayMode = false;
                hiddenAgent.UseExternalReplayData = false;
            }

            if (isSelected) shown++; else hidden++;
        }

        // If we just activated a robot for the first time, give the IMimicAgent
        // registry a chance to pick up its newly-registered entry before the
        // caller (OnRoboListChanged) tries to FindByKey.
        if (bootstrapped > 0)
        {
            Debug.Log($"[StartInput] 首次激活 {bootstrapped} 个机器人（SetActive(true) bootstrap）。");
        }
        Debug.Log($"[StartInput] 可见性切换 (Renderer mode)：显示 {shown} 个，隐藏 {hidden} 个 → 当前 '{selectedLabel}'。");
    }

    // Called by Unity when the application is quitting (covers editor Stop, build exit, and
    // OS-level termination signals that Unity intercepts).
    void OnApplicationQuit()
    {
        StopCsvMonitor();
        StopBashProcess();

        if (clearCsvOnExit)
        {
            TryClearCsv("OnApplicationQuit");
        }
    }

    public void OnStartButtonClicked()
    {
        if (runBashOnClick)
        {
            StartOrRestartBashProcess();
        }

        if (monitorCsvOnClick)
        {
            StartCsvMonitor();
        }
    }

    public void StopStartPipeline()
    {
        StopCsvMonitor();
        StopBashProcess();

        if (clearCsvOnExit)
        {
            TryClearCsv("StopStartPipeline");
        }
    }

    // -------------------------------------------------------------------------
    // CSV helpers
    // -------------------------------------------------------------------------

    /// <summary>
    /// Resolves the CSV path using the current inspector settings and deletes the file if it
    /// exists. Safe to call at any point; failures are logged as warnings, never thrown.
    /// </summary>
    private void TryClearCsv(string caller)
    {
        string csvPath = ResolveCsvAbsolutePath();
        if (string.IsNullOrWhiteSpace(csvPath))
        {
            return;
        }

        if (!File.Exists(csvPath))
        {
            return;
        }

        try
        {
            File.Delete(csvPath);
            Debug.Log($"[{caller}] live_motion.csv cleared: {csvPath}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[{caller}] Failed to clear live_motion.csv ({csvPath}): {e.Message}");
        }
    }

    /// <summary>
    /// Resolves the absolute path of the CSV output file from current inspector settings.
    /// Returns empty string when the path cannot be determined.
    /// </summary>
    private string ResolveCsvAbsolutePath()
    {
        // Use already-resolved path when available (set during StartOrRestartBashProcess).
        if (!string.IsNullOrWhiteSpace(resolvedCsvPath))
        {
            return resolvedCsvPath;
        }

        // Fall back to computing the path from scratch (useful when called before bash launch).
        string baseDir = ResolveWorkingDirectoryPath();
        string outputRoot = ResolveOutputRootAbsolutePath(baseDir);
        if (string.IsNullOrWhiteSpace(outputRoot))
        {
            return string.Empty;
        }

        string csvRelative = ResolveCsvRelativePath();
        return ResolvePathFromBaseDirectory(csvRelative, outputRoot);
    }

    // -------------------------------------------------------------------------
    // Existing private methods (unchanged except StartOrRestartBashProcess which
    // now calls TryClearCsv before launching).
    // -------------------------------------------------------------------------

    private bool HasPersistentStartHandler(Button button)
    {
        int eventCount = button.onClick.GetPersistentEventCount();
        for (int i = 0; i < eventCount; i++)
        {
            if (button.onClick.GetPersistentTarget(i) == this &&
                button.onClick.GetPersistentMethodName(i) == nameof(OnStartButtonClicked))
            {
                return true;
            }
        }

        return false;
    }

    private void StartOrRestartBashProcess()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            if (!restartBashIfRunning)
            {
                Debug.Log("Bash 进程已在运行，跳过重复启动。");
                return;
            }

            StopBashProcess();
        }

        string resolvedWorkingDir = ResolveWorkingDirectoryPath();
        if (string.IsNullOrWhiteSpace(resolvedWorkingDir) || !Directory.Exists(resolvedWorkingDir))
        {
            Debug.LogError($"Bash 工作目录无效: {resolvedWorkingDir}");
            return;
        }

        string executable = (bashExecutable ?? string.Empty).Trim();
        string executablePrefixArgs = string.Empty;
        ParseExecutable(executable, out executable, out executablePrefixArgs);

        if (string.IsNullOrWhiteSpace(executable))
        {
            Debug.LogError("bashExecutable 为空，无法启动 Bash 命令。");
            return;
        }

        string selectedRobot = ResolveSelectedRobotName();
        if (string.IsNullOrWhiteSpace(selectedRobot))
        {
            selectedRobot = defaultRobotName.Trim();
        }

        if (string.IsNullOrWhiteSpace(selectedRobot))
        {
            Debug.LogError("未能解析 RoboList 当前机器人名称，请检查下拉列表配置。");
            return;
        }

        string resolvedRobot = ResolveRobotNameForWham(selectedRobot);
        if (string.IsNullOrWhiteSpace(resolvedRobot))
        {
            Debug.LogError($"当前机器人不受 WHAM 支持: {selectedRobot}");
            return;
        }

        resolvedOutputRootPath = ResolveOutputRootAbsolutePath(resolvedWorkingDir);
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            Debug.LogError("OUTPUT_ROOT 无效，无法启动 Bash 命令。");
            return;
        }

        Directory.CreateDirectory(resolvedOutputRootPath);

        // Resolve CSV path now so TryClearCsv and the monitor coroutine share the same value.
        string csvRelative = ResolveCsvRelativePath();
        resolvedCsvPath = ResolvePathFromBaseDirectory(csvRelative, resolvedOutputRootPath);

        // Clear stale CSV before launching so the monitor never sees data from a prior run.
        if (clearCsvOnStart)
        {
            TryClearCsv("OnStart");
        }

        string resolvedScriptPath = ResolveScriptPath(resolvedWorkingDir);
        string commandArguments = BuildCommandArguments(executablePrefixArgs, resolvedScriptPath);

        try
        {
            // On Linux we wrap the command with `setsid` so that bash and all
            // children (WHAM Python, GMR Python, MuJoCo/OpenCV windows) share a
            // single new process group. StopBashProcess() then kills the entire
            // group with one signal instead of only the bash shell.
            string launchExecutable = executable;
            string launchArguments  = commandArguments;

            if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
            {
                launchArguments  = $"{executable} {commandArguments}";
                launchExecutable = "setsid";
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = launchExecutable,
                Arguments = launchArguments,
                WorkingDirectory = resolvedWorkingDir,
                UseShellExecute = false,
                RedirectStandardOutput = logBashOutput,
                RedirectStandardError = logBashOutput,
                CreateNoWindow = true
            };

            startInfo.EnvironmentVariables["OUTPUT_ROOT"] = resolvedOutputRootPath;
            startInfo.EnvironmentVariables["ROBOT"] = resolvedRobot;
            startInfo.EnvironmentVariables["RECORD_GMRVIDEO"] = recordGmrVideo ? "1" : "0";
            startInfo.EnvironmentVariables["RECORD_WHAMVIDEO"] = recordWhamVideo ? "1" : "0";
            string resolvedVideoPath = ResolveVideoPath(resolvedWorkingDir);
            if (!string.IsNullOrWhiteSpace(resolvedVideoPath))
            {
                startInfo.EnvironmentVariables["VIDEO"] = resolvedVideoPath;
            }

            // WHAM performance env vars (read by handle_wham_gmr.py via os.environ).
            startInfo.EnvironmentVariables["WHAM_USE_AMP"] = whamUseAmp ? "1" : "0";
            startInfo.EnvironmentVariables["WHAM_DETECT_INTERVAL"] = whamDetectInterval.ToString();
            startInfo.EnvironmentVariables["WHAM_INFER_INTERVAL"] = whamInferInterval.ToString();
            startInfo.EnvironmentVariables["WHAM_STREAM_SEQ_LEN"] = whamStreamSeqLen.ToString();
            startInfo.EnvironmentVariables["WHAM_INPUT_SCALE"] = whamInputScale.ToString("F3", CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["GMR_TORCH_DEVICE"] = string.IsNullOrWhiteSpace(gmrTorchDevice) ? "cpu" : gmrTorchDevice.Trim();
            startInfo.EnvironmentVariables["TRACK"] = track ? "1" : "0";
            pythonProcess = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            pythonProcess.Exited += (_, __) => Debug.Log($"Bash 进程退出，ExitCode={pythonProcess.ExitCode}");

            if (logBashOutput)
            {
                pythonProcess.OutputDataReceived += (_, args) =>
                {
                    if (!string.IsNullOrWhiteSpace(args.Data))
                    {
                        Debug.Log($"[Pipeline] {args.Data}");
                    }
                };

                pythonProcess.ErrorDataReceived += (_, args) =>
                {
                    if (!string.IsNullOrWhiteSpace(args.Data))
                    {
                        Debug.Log($"[Pipeline-ERR] {args.Data}");
                    }
                };
            }

            pythonProcess.Start();

            if (logBashOutput)
            {
                pythonProcess.BeginOutputReadLine();
                pythonProcess.BeginErrorReadLine();
            }

            Debug.Log($"已启动 Bash 命令: {executable} {commandArguments}");
            Debug.Log("运行参数: ROBOT=" + resolvedRobot + " (raw=" + selectedRobot + "), OUTPUT_ROOT=" + resolvedOutputRootPath + ", TRACK=" + (track ? "1" : "0"));
        }
        catch (System.Exception e)
        {
            Debug.LogError($"启动 Bash 命令失败: {e.Message}");
        }
    }

    private void StopBashProcess()
    {
        if (pythonProcess == null)
        {
            return;
        }

        try
        {
            if (!pythonProcess.HasExited)
            {
                if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
                {
                    // Kill the entire process group so WHAM, GMR, and their GUI
                    // windows (MuJoCo / OpenCV) all receive SIGKILL.
                    // The process group ID equals the PID of the setsid-launched
                    // bash process, so `kill -9 -<pid>` targets every member.
                    KillProcessGroupLinux(pythonProcess.Id);
                }
                else
                {
                    // Windows: kill entire process tree so Python children don't
                    // survive as orphans (process.Kill() only kills the top-level
                    // powershell.exe, not python.exe / MuJoCo windows underneath).
                    KillProcessTreeWindows(pythonProcess.Id);
                }

                pythonProcess.WaitForExit(2000);
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"停止 Bash 进程时出现异常: {e.Message}");
        }
        finally
        {
            pythonProcess.Dispose();
            pythonProcess = null;
        }

        // Belt-and-suspenders: kill any surviving Python workers by matching
        // the script names that the pipeline always launches.
        if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
        {
            KillOrphansByName("demo_stream_mt");
            KillOrphansByName("smplx_to_robot_stream");
        }
        else
        {
            // Windows: the pipeline now runs handle_wham_gmr.py directly.
            KillOrphansByName("handle_wham_gmr");
        }
    }

    /// <summary>
    /// Kills the entire process tree rooted at <paramref name="pid"/> on Windows
    /// via <c>taskkill /T /F /PID</c>.  The /T flag ensures child processes
    /// (python.exe, MuJoCo windows, etc.) are also terminated.
    /// </summary>
    private static void KillProcessTreeWindows(int pid)
    {
        try
        {
            var killInfo = new ProcessStartInfo
            {
                FileName               = "taskkill",
                Arguments              = $"/T /F /PID {pid}",
                UseShellExecute        = false,
                RedirectStandardOutput = false,
                RedirectStandardError  = false,
                CreateNoWindow         = true
            };

            using (Process killer = Process.Start(killInfo))
            {
                killer?.WaitForExit(5000);
            }

            Debug.Log($"[StopBash] taskkill /T /F /PID {pid} executed");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[StopBash] taskkill failed (pid={pid}): {e.Message}");
        }
    }

    /// <summary>
    /// Sends SIGKILL to the entire Linux process group whose PGID equals
    /// <paramref name="leaderPid"/>. Because we launched bash with <c>setsid</c>,
    /// its PID is also the PGID, so every child inherits that group.
    /// </summary>
    private static void KillProcessGroupLinux(int leaderPid)
    {
        try
        {
            // `kill -9 -<pgid>` — the negative sign means "process group".
            var killInfo = new ProcessStartInfo
            {
                FileName               = "kill",
                Arguments              = $"-9 -{leaderPid}",
                UseShellExecute        = false,
                RedirectStandardOutput = false,
                RedirectStandardError  = false,
                CreateNoWindow         = true
            };

            using (Process killer = Process.Start(killInfo))
            {
                killer?.WaitForExit(3000);
            }

            Debug.Log($"[StopBash] 已发送 SIGKILL 至进程组 -{leaderPid}");
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[StopBash] kill 进程组失败 (pgid={leaderPid}): {e.Message}");
        }
    }

    /// <summary>
    /// Kills any remaining processes whose command line contains
    /// <paramref name="scriptNameFragment"/>.
    /// On Linux uses <c>pkill -9 -f</c>; on Windows uses <c>wmic</c> to
    /// terminate python.exe instances matching the fragment.
    /// </summary>
    private static void KillOrphansByName(string scriptNameFragment)
    {
        try
        {
            if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
            {
                var pkillInfo = new ProcessStartInfo
                {
                    FileName               = "pkill",
                    Arguments              = $"-9 -f \"{scriptNameFragment}\"",
                    UseShellExecute        = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError  = false,
                    CreateNoWindow         = true
                };

                using (Process pkill = Process.Start(pkillInfo))
                {
                    pkill?.WaitForExit(3000);
                }

                Debug.Log($"[StopBash] pkill -9 -f \"{scriptNameFragment}\" 已执行");
            }
            else
            {
                // Windows: use wmic to find and kill python processes whose
                // command line contains the script fragment.
                string filter = $"name='python.exe' and commandline like '%{scriptNameFragment}%'";
                var wmicInfo = new ProcessStartInfo
                {
                    FileName               = "wmic",
                    Arguments              = $"process where \"{filter}\" call terminate",
                    UseShellExecute        = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError  = false,
                    CreateNoWindow         = true
                };

                using (Process wmic = Process.Start(wmicInfo))
                {
                    wmic?.WaitForExit(5000);
                }

                Debug.Log($"[StopBash] wmic terminate python.exe *{scriptNameFragment}* executed");
            }
        }
        catch (System.Exception e)
        {
            Debug.Log($"[StopBash] orphan cleanup '{scriptNameFragment}': {e.Message}");
        }
    }

    private void StartCsvMonitor()
    {
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            string baseDir = ResolveWorkingDirectoryPath();
            resolvedOutputRootPath = ResolveOutputRootAbsolutePath(baseDir);
        }

        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            Debug.LogError("OUTPUT_ROOT 无效，无法开始监听实时 CSV。");
            return;
        }

        string csvName = ResolveCsvRelativePath();
        resolvedCsvPath = ResolvePathFromBaseDirectory(csvName, resolvedOutputRootPath);
        if (string.IsNullOrWhiteSpace(resolvedCsvPath))
        {
            Debug.LogError("实时 CSV 路径为空，无法开始监听。");
            return;
        }

        StopCsvMonitor();

        lastCsvLength = -1;
        lastCsvWriteTimeUtc = System.DateTime.MinValue;
        replayBootstrapped = false;
        csvMissingLogged = false;
        monitorCoroutine = StartCoroutine(MonitorCsvCoroutine());

        Debug.Log($"开始监听实时 CSV: {resolvedCsvPath}");
    }

    private void StopCsvMonitor()
    {
        if (monitorCoroutine != null)
        {
            StopCoroutine(monitorCoroutine);
            monitorCoroutine = null;
        }
    }

    private IEnumerator MonitorCsvCoroutine()
    {
        while (true)
        {
            if (!File.Exists(resolvedCsvPath))
            {
                if (!csvMissingLogged)
                {
                    Debug.LogWarning($"CSV 文件不存在，等待生成: {resolvedCsvPath}");
                    csvMissingLogged = true;
                }

                yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
                continue;
            }

            csvMissingLogged = false;

            long currentLength;
            System.DateTime currentWriteTimeUtc;
            bool readFailed = false;
            try
            {
                FileInfo info = new FileInfo(resolvedCsvPath);
                currentLength = info.Length;
                currentWriteTimeUtc = info.LastWriteTimeUtc;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"读取 CSV 文件信息失败: {e.Message}");
                currentLength = lastCsvLength;
                currentWriteTimeUtc = lastCsvWriteTimeUtc;
                readFailed = true;
            }

            if (readFailed)
            {
                yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
                continue;
            }

            bool isChanged = currentLength != lastCsvLength || currentWriteTimeUtc != lastCsvWriteTimeUtc;

            if (isChanged)
            {
                lastCsvLength = currentLength;
                lastCsvWriteTimeUtc = currentWriteTimeUtc;
                ApplyRealtimeCsvToAgent();
            }

            yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
        }
    }

    private void ApplyRealtimeCsvToAgent()
    {
        IMimicAgent agent = ResolveActiveAgent();
        if (agent == null)
        {
            Debug.LogWarning("未找到匹配当前机器人的 IMimicAgent，无法应用实时 CSV。\n" +
                             "请确认场景里至少有一个机器人组件实现 IMimicAgent，且其 RobotKey 与 RoboList 选择一致。");
            return;
        }

        bool loaded = agent.LoadReplayCsvFromPath(resolvedCsvPath, keepReplayProgressOnCsvUpdate);
        if (!loaded)
        {
            return;
        }

        agent.UseExternalReplayData = true;
        agent.ReplayMode = true;

        // Make sure the other registered robots are not still consuming an
        // older live stream — only the active one should be in external-replay
        // mode at a time.
        if (MimicAgentRegistry.Instance != null)
        {
            MimicAgentRegistry.Instance.SetActiveTarget(agent);
        }

        if (!replayBootstrapped && restartEpisodeOnFirstCsv)
        {
            agent.RequestEndEpisode();
        }

        replayBootstrapped = true;
    }

    /// <summary>
    /// Resolve the agent that should receive live retargeting data. Resolution
    /// order: inspector-assigned <c>targetAgentBehaviour</c>, then the agent
    /// registered with a RobotKey matching the current RoboList selection,
    /// then the first registered agent as a final fallback.
    /// </summary>
    private IMimicAgent ResolveActiveAgent()
    {
        // 1) Inspector-pinned target wins (preserves the legacy single-robot
        //    scene setup where the agent is dragged directly into the field).
        if (targetAgentBehaviour is IMimicAgent pinned && pinned.AgentGameObject != null)
        {
            targetAgent = pinned;
            return targetAgent;
        }

        // 2) Match the registry by the robot key currently selected in the UI.
        string selectedRaw = ResolveSelectedRobotName();
        string selectedKey = ResolveRobotNameForWham(selectedRaw);
        if (string.IsNullOrWhiteSpace(selectedKey)) selectedKey = (defaultRobotName ?? string.Empty).Trim();

        if (!string.IsNullOrWhiteSpace(selectedKey) && MimicAgentRegistry.Instance != null)
        {
            IMimicAgent byKey = MimicAgentRegistry.Instance.FindByKey(selectedKey);
            if (byKey != null && byKey.AgentGameObject != null)
            {
                if (lastResolvedRobotKey != selectedKey)
                {
                    Debug.Log($"[StartInput] 实时重定向目标已切换为机器人: {selectedKey} ({byKey.AgentGameObject.name})");
                    lastResolvedRobotKey = selectedKey;
                }
                targetAgent = byKey;
                return targetAgent;
            }
        }

        // 3) Last-resort: first registered alive agent.
        if (MimicAgentRegistry.Instance != null)
        {
            IMimicAgent first = MimicAgentRegistry.Instance.GetFirstAlive();
            if (first != null)
            {
                if (lastResolvedRobotKey != "<fallback>")
                {
                    Debug.LogWarning($"[StartInput] 未找到 RobotKey='{selectedKey}' 对应的机器人，" +
                                     $"回退到第一个已注册机器人: {first.RobotKey}");
                    lastResolvedRobotKey = "<fallback>";
                }
                targetAgent = first;
                return targetAgent;
            }
        }

        targetAgent = null;
        return null;
    }

    // Built-in roots tried (in order) when bashWorkingDirectory is a relative
    // path. Lets the script keep working regardless of whether the user keeps
    // the Imitation folder directly under Assets/ or nested under Assets/Gewu/.
    private static readonly string[] RelativeWorkingDirRoots =
    {
        "Gewu/Imitation",
        "Imitation",
        "",               // last-ditch: resolve against Assets/ directly
    };

    private string ResolveWorkingDirectoryPath()
    {
        if (string.IsNullOrWhiteSpace(bashWorkingDirectory))
        {
            Debug.LogError("请在 Inspector 中设置 bashWorkingDirectory（相对路径会从 Assets/Gewu/Imitation/ 或 Assets/Imitation/ 解析）。");
            return string.Empty;
        }

        string expanded = ExpandPathPrefix(bashWorkingDirectory);
        string normalized = NormalizePathSeparators(expanded);

        // Absolute path: just normalize and use.
        if (Path.IsPathRooted(normalized))
        {
            try { return Path.GetFullPath(normalized); }
            catch (System.Exception e)
            {
                Debug.LogError($"解析 bashWorkingDirectory 失败: {e.Message}");
                return string.Empty;
            }
        }

        // Relative path: try each built-in root in order, returning the first
        // one that actually exists on disk. If none exist, fall back to the
        // first candidate so the caller's "Directory.Exists" check produces
        // a helpful error message listing the path we ended up at.
        List<string> tried = new List<string>();
        string firstCandidate = string.Empty;
        foreach (string root in RelativeWorkingDirRoots)
        {
            string baseDir = string.IsNullOrEmpty(root)
                ? Application.dataPath
                : Path.Combine(Application.dataPath, root);

            string candidate;
            try { candidate = Path.GetFullPath(Path.Combine(baseDir, normalized)); }
            catch { continue; }

            tried.Add(candidate);
            if (string.IsNullOrEmpty(firstCandidate)) firstCandidate = candidate;
            if (Directory.Exists(candidate)) return candidate;
        }

        Debug.LogWarning("[StartInput] bashWorkingDirectory 在每个内置根目录下都找不到。已尝试:\n  " +
                         string.Join("\n  ", tried));
        return firstCandidate;
    }

    private string ResolveOutputRootAbsolutePath(string baseDir)
    {
        return ResolvePathFromBaseDirectory(outputRoot, baseDir);
    }

    private string ResolveScriptPath(string baseDir)
    {
        return ResolvePathFromBaseDirectory(bashScriptPath, baseDir);
    }

    private string ResolveVideoPath(string baseDir)
    {
        string trimmedVideoPath = (videoPath ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(trimmedVideoPath))
        {
            return string.Empty;
        }

        if (trimmedVideoPath == "0")
        {
            return "0";
        }

        return ResolvePathFromBaseDirectory(trimmedVideoPath, baseDir);
    }

    private string ResolveCsvRelativePath()
    {
        string csvName = string.IsNullOrWhiteSpace(outputCsvFileName) ? "csv/live_motion.csv" : outputCsvFileName.Trim();
        if (string.Equals(csvName, "live_motion.csv", System.StringComparison.OrdinalIgnoreCase))
        {
            return "csv/live_motion.csv";
        }

        return csvName;
    }

    private string ResolveRobotNameForWham(string rawRobotName)
    {
        string resolved = TryResolveRobotKeyQuiet(rawRobotName);
        if (string.IsNullOrWhiteSpace(resolved))
        {
            Debug.LogError($"不支持的机器人名称: {rawRobotName}。支持列表: {string.Join(", ", SupportedRobotNames)}");
        }
        return resolved;
    }

    /// <summary>
    /// Resolve a dropdown label / arbitrary string into a WHAM-supported
    /// robot key, without logging an error on miss. Used by listeners that
    /// fire on every dropdown click so unknown labels don't spam the console.
    /// </summary>
    private static string TryResolveRobotKeyQuiet(string rawRobotName)
    {
        if (string.IsNullOrWhiteSpace(rawRobotName)) return string.Empty;

        string trimmed = rawRobotName.Trim();
        if (SupportedRobotNames.Contains(trimmed)) return trimmed;

        if (RobotAliases.TryGetValue(trimmed, out string aliasTarget) && SupportedRobotNames.Contains(aliasTarget))
            return aliasTarget;

        return string.Empty;
    }

    private string ResolvePathFromBaseDirectory(string configuredPath, string baseDir)
    {
        if (string.IsNullOrWhiteSpace(configuredPath))
        {
            return string.Empty;
        }

        string expanded = ExpandPathPrefix(configuredPath);
        string normalized = NormalizePathSeparators(expanded);

        try
        {
            if (Path.IsPathRooted(normalized))
            {
                return Path.GetFullPath(normalized);
            }

            if (string.IsNullOrWhiteSpace(baseDir))
            {
                return string.Empty;
            }

            return Path.GetFullPath(Path.Combine(baseDir, normalized));
        }
        catch (System.Exception e)
        {
            Debug.LogError($"解析路径失败: {configuredPath} ({e.Message})");
            return string.Empty;
        }
    }

    private static string ExpandPathPrefix(string rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            return string.Empty;
        }

        string trimmed = rawPath.Trim();
        string homeDirectory = System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile);

        if (trimmed == "~")
        {
            return homeDirectory;
        }

        if (trimmed.StartsWith("~/", System.StringComparison.Ordinal) ||
            trimmed.StartsWith("~\\", System.StringComparison.Ordinal))
        {
            return Path.Combine(homeDirectory, trimmed.Substring(2));
        }

        return System.Environment.ExpandEnvironmentVariables(trimmed);
    }

    private static string NormalizePathSeparators(string rawPath)
    {
        return (rawPath ?? string.Empty)
            .Replace('/', Path.DirectorySeparatorChar)
            .Replace('\\', Path.DirectorySeparatorChar);
    }

    private void ParseExecutable(string rawExecutable, out string executable, out string prefixArgs)
    {
        executable = (rawExecutable ?? string.Empty).Trim();
        prefixArgs = string.Empty;

        if (string.IsNullOrWhiteSpace(executable))
        {
            return;
        }

        // 路径本身存在时，不拆分（兼容包含空格的绝对路径）。
        if (File.Exists(executable))
        {
            return;
        }

        if (executable.StartsWith("\"") && executable.Contains("\" "))
        {
            int quoteEnd = executable.IndexOf("\" ", System.StringComparison.Ordinal);
            if (quoteEnd > 0)
            {
                string quotedPath = executable.Substring(1, quoteEnd - 1);
                string remain = executable.Substring(quoteEnd + 2).Trim();
                if (!string.IsNullOrWhiteSpace(quotedPath))
                {
                    executable = quotedPath;
                    prefixArgs = remain;
                    return;
                }
            }
        }

        int firstSpace = executable.IndexOf(' ');
        if (firstSpace > 0)
        {
            prefixArgs = executable.Substring(firstSpace + 1).Trim();
            executable = executable.Substring(0, firstSpace).Trim();
        }
    }

    private string BuildCommandArguments(string executablePrefixArgs, string resolvedScriptPath)
    {
        string extraArgs = (bashArguments ?? string.Empty).Trim();
        string scriptArg = string.IsNullOrWhiteSpace(resolvedScriptPath) ? string.Empty : QuoteArgument(resolvedScriptPath);

        // Build: [prefixArgs] [scriptPath] [extraArgs]
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(executablePrefixArgs))
            parts.Add(executablePrefixArgs);
        if (!string.IsNullOrWhiteSpace(scriptArg))
            parts.Add(scriptArg);
        if (!string.IsNullOrWhiteSpace(extraArgs))
            parts.Add(extraArgs);

        return string.Join(" ", parts);
    }

    private static string QuoteArgument(string argument)
    {
        if (string.IsNullOrWhiteSpace(argument))
        {
            return string.Empty;
        }

        string escaped = argument.Replace("\"", "\\\"");
        return $"\"{escaped}\"";
    }

    private string ResolveSelectedRobotName()
    {
        ResolveRoboListReferences();

        if (roboListFileBrowser != null)
        {
            string selectedFolderPath = roboListFileBrowser.GetSelectedFolderPath();
            if (!string.IsNullOrWhiteSpace(selectedFolderPath))
            {
                return Path.GetFileName(selectedFolderPath).Trim();
            }

            string selectedCsvName = roboListFileBrowser.GetSelectedCsvName();
            if (!string.IsNullOrWhiteSpace(selectedCsvName))
            {
                return selectedCsvName.Trim();
            }
        }

        if (roboListDropdown != null && roboListDropdown.options != null && roboListDropdown.options.Count > 0)
        {
            string selectedText = roboListDropdown.options[roboListDropdown.value].text.Trim();
            if (!IsPlaceholderOption(selectedText))
            {
                return selectedText;
            }
        }

        return string.Empty;
    }

    private void ResolveRoboListReferences()
    {
        if (roboListFileBrowser != null && roboListDropdown != null)
        {
            return;
        }

        GameObject roboListObject = GameObject.Find(roboListObjectName);
        if (roboListObject == null)
        {
            return;
        }

        if (roboListFileBrowser == null)
        {
            roboListFileBrowser = roboListObject.GetComponent<FileBrowser>();
        }

        if (roboListDropdown == null)
        {
            roboListDropdown = roboListObject.GetComponent<TMP_Dropdown>();
        }
    }

    private bool IsPlaceholderOption(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return true;
        }

        if (text.StartsWith("(", System.StringComparison.Ordinal))
        {
            return true;
        }

        string lowered = text.ToLowerInvariant();
        return lowered.Contains("can't find") || lowered.Contains("cannot find");
    }
}
