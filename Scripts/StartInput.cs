using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Threading;
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
    [SerializeField] private bool enableTcpStreaming = true;
    [Tooltip("When TCP streaming is enabled, do not also open/write WHAM/GMR preview videos. TCP frames still render offscreen for Unity.")]
    [SerializeField] private bool disablePreviewVideoWhenTcpStreaming = true;
    [Tooltip("Input video path passed to run.ps1. Defaults to an examples video instead of webcam input.")]
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

    [Header("CsvList")]
    [SerializeField] private TMP_Dropdown csvListDropdown;
    [SerializeField] private FileBrowser csvListFileBrowser;
    [SerializeField] private string csvListObjectName = "CsvList";
    [SerializeField] private bool filterCsvListByRobot = true;

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
    [Tooltip("Wait for this many 30fps source rows before starting live Unity playback.")]
    [SerializeField] private int realtimeWarmupSourceRows = 6;
    [Tooltip("Maximum realtime CSV batches applied on the Unity main thread per frame.")]
    [SerializeField] private int maxRealtimeCsvBatchesPerFrame = 4;
    [Tooltip("Maximum queued pipeline log lines emitted to the Unity console per frame.")]
    [SerializeField] private int maxUnityLogMessagesPerFrame = 80;

    [Header("Debug")]
    [SerializeField] private bool verboseDebugLogging = true;
    [SerializeField] private bool logCsvChangeEvents = true;
    [SerializeField] private bool logReferenceResolution = false;
    [SerializeField] private bool writeDebugLogFile = true;
    [SerializeField] private string debugLogFileName = "unity_startinput.log";
    [Tooltip("Warn if the Python pipeline has started but live_motion.csv remains empty for this many seconds.")]
    [SerializeField] private float csvNoDataWarningSeconds = 10f;

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
    private System.DateTime csvMonitorStartUtc = System.DateTime.MinValue;
    private System.DateTime lastCsvNoDataWarningUtc = System.DateTime.MinValue;
    private bool firstNonEmptyCsvLogged;
    private bool initialRobotSelectionApplied;
    private long csvReadOffset;
    private string csvPendingText = string.Empty;
    private int realtimeSourceRowsRead;
    private readonly List<float[]> pendingRealtimeRows = new List<float[]>();
    private IRealtimeCsvMimicAgent activeRealtimeCsvAgent;
    private IMimicAgent activeRealtimeMimicAgent;
    private Thread csvWorkerThread;
    private volatile bool csvWorkerRunning;
    private volatile bool isStoppingPipeline;
    private readonly object processStateLock = new object();
    private readonly ConcurrentQueue<List<float[]>> realtimeCsvRowsQueue = new ConcurrentQueue<List<float[]>>();
    private readonly ConcurrentQueue<string> realtimeCsvResetQueue = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<QueuedUnityLog> unityLogQueue = new ConcurrentQueue<QueuedUnityLog>();
    private readonly ConcurrentQueue<string> debugLogQueue = new ConcurrentQueue<string>();
    private readonly object debugLogWriterLock = new object();
    private readonly AutoResetEvent debugLogSignal = new AutoResetEvent(false);
    private Thread debugLogWriterThread;
    private volatile bool debugLogWriterRunning;
    private string resolvedDebugLogPath = string.Empty;

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
        "x02lite",
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
        { "X02", "x02lite" },
        { "X02Lite", "x02lite" },
        { "OpenLoong", "openloong" },
        { "T1",   "booster_t1" },
    };

    private enum QueuedUnityLogType
    {
        Log,
        Warning,
        Error,
    }

    private struct QueuedUnityLog
    {
        public readonly string Message;
        public readonly QueuedUnityLogType Type;

        public QueuedUnityLog(string message, QueuedUnityLogType type)
        {
            Message = message;
            Type = type;
        }
    }

    public string BashWorkingDirectory
    {
        get => bashWorkingDirectory;
        set => bashWorkingDirectory = value ?? string.Empty;
    }

    private void LogVerbose(string message)
    {
        if (!verboseDebugLogging) return;
        string line = "[StartInput/Debug] " + message;
        Debug.Log(line);
        AppendDebugLogLine(line);
    }

    private void QueueUnityLog(string message, QueuedUnityLogType type = QueuedUnityLogType.Log)
    {
        unityLogQueue.Enqueue(new QueuedUnityLog(message, type));
        AppendDebugLogLine(message);
    }

    private void FlushUnityLogQueue()
    {
        int maxLogs = Mathf.Max(1, maxUnityLogMessagesPerFrame);
        for (int i = 0; i < maxLogs && unityLogQueue.TryDequeue(out QueuedUnityLog item); i++)
        {
            switch (item.Type)
            {
                case QueuedUnityLogType.Warning:
                    Debug.LogWarning(item.Message);
                    break;
                case QueuedUnityLogType.Error:
                    Debug.LogError(item.Message);
                    break;
                default:
                    Debug.Log(item.Message);
                    break;
            }
        }
    }

    private void AppendDebugLogLine(string message)
    {
        if (!writeDebugLogFile)
        {
            return;
        }

        try
        {
            if (string.IsNullOrWhiteSpace(resolvedDebugLogPath))
            {
                resolvedDebugLogPath = ResolveDebugLogPath();
                if (string.IsNullOrWhiteSpace(resolvedDebugLogPath))
                {
                    return;
                }
            }

            EnsureDebugLogWriter();
            debugLogQueue.Enqueue($"{System.DateTime.Now:O} {message}");
            debugLogSignal.Set();
        }
        catch
        {
            // Logging must never break Start/Stop.
        }
    }

    private void EnsureDebugLogWriter()
    {
        if (!writeDebugLogFile || string.IsNullOrWhiteSpace(resolvedDebugLogPath))
        {
            return;
        }

        lock (debugLogWriterLock)
        {
            if (debugLogWriterThread != null && debugLogWriterThread.IsAlive)
            {
                return;
            }

            debugLogWriterRunning = true;
            debugLogWriterThread = new Thread(DebugLogWriterLoop)
            {
                Name = "StartInput-DebugLogWriter",
                IsBackground = true
            };
            debugLogWriterThread.Start();
        }
    }

    private void DebugLogWriterLoop()
    {
        var batch = new List<string>(128);
        while (debugLogWriterRunning || !debugLogQueue.IsEmpty)
        {
            batch.Clear();
            while (batch.Count < 128 && debugLogQueue.TryDequeue(out string line))
            {
                batch.Add(line);
            }

            if (batch.Count > 0)
            {
                try
                {
                    string path = resolvedDebugLogPath;
                    if (!string.IsNullOrWhiteSpace(path))
                    {
                        string dir = Path.GetDirectoryName(path);
                        if (!string.IsNullOrWhiteSpace(dir))
                        {
                            Directory.CreateDirectory(dir);
                        }

                        File.AppendAllLines(path, batch, Encoding.UTF8);
                    }
                }
                catch
                {
                    // Logging must never break Start/Stop.
                }

                continue;
            }

            debugLogSignal.WaitOne(250);
        }
    }

    private void StopDebugLogWriter()
    {
        debugLogWriterRunning = false;
        debugLogSignal.Set();
        Thread writer = debugLogWriterThread;
        if (writer != null && writer.IsAlive)
        {
            try { writer.Join(500); } catch { }
        }
        debugLogWriterThread = null;
    }

    private void ResetDebugLogFile()
    {
        if (!writeDebugLogFile)
        {
            return;
        }

        try
        {
            string path = ResolveDebugLogPath();
            if (string.IsNullOrWhiteSpace(path))
            {
                return;
            }

            resolvedDebugLogPath = path;
            string dir = Path.GetDirectoryName(path);
            if (!string.IsNullOrWhiteSpace(dir))
            {
                Directory.CreateDirectory(dir);
            }

            File.WriteAllText(path, $"{System.DateTime.Now:O} [StartInput] debug log reset{System.Environment.NewLine}", Encoding.UTF8);
            EnsureDebugLogWriter();
        }
        catch
        {
            // Logging must never break Start/Stop.
        }
    }

    private string ResolveDebugLogPath()
    {
        string logName = string.IsNullOrWhiteSpace(debugLogFileName)
            ? "unity_startinput.log"
            : debugLogFileName.Trim();

        if (Path.IsPathRooted(logName))
        {
            return Path.GetFullPath(logName);
        }

        string outputRootPath = resolvedOutputRootPath;
        if (string.IsNullOrWhiteSpace(outputRootPath))
        {
            string workingDir = ResolveWorkingDirectoryPath();
            outputRootPath = ResolveOutputRootAbsolutePath(workingDir);
        }

        if (string.IsNullOrWhiteSpace(outputRootPath))
        {
            return string.Empty;
        }

        return ResolvePathFromBaseDirectory(logName, outputRootPath);
    }

    [ContextMenu("Dump StartInput Debug State")]
    private void DumpDebugState()
    {
        ResolveRoboListReferences();
        ResolveCsvListReferences();

        string selectedRaw = ResolveSelectedRobotName();
        string selectedKey = TryResolveRobotKeyQuiet(selectedRaw);
        string workingDir = ResolveWorkingDirectoryPath();
        string outputRootAbs = ResolveOutputRootAbsolutePath(workingDir);
        string csvAbs = ResolveCsvAbsolutePath();
        string resolvedVideo = ResolveVideoPath(workingDir);

        IMimicAgent resolvedAgent = null;
        if (!string.IsNullOrWhiteSpace(selectedKey) && MimicAgentRegistry.Instance != null)
        {
            resolvedAgent = MimicAgentRegistry.Instance.FindByKey(selectedKey);
        }

        var sb = new StringBuilder();
        sb.AppendLine("[StartInput/Debug] State Snapshot");
        sb.AppendLine($"  GameObject: {gameObject.name}");
        sb.AppendLine($"  enabled: {enabled}, activeInHierarchy: {gameObject.activeInHierarchy}");
        sb.AppendLine($"  selectedRaw: '{selectedRaw}'");
        sb.AppendLine($"  selectedKey: '{selectedKey}'");
        sb.AppendLine($"  defaultRobotName: '{defaultRobotName}'");
        sb.AppendLine($"  runBashOnClick: {runBashOnClick}, monitorCsvOnClick: {monitorCsvOnClick}, tcp: {enableTcpStreaming}");
        sb.AppendLine($"  bashExecutable: {bashExecutable}");
        sb.AppendLine($"  bashScriptPath: {bashScriptPath}");
        sb.AppendLine($"  bashWorkingDirectory: {bashWorkingDirectory}");
        sb.AppendLine($"  resolvedWorkingDirectory: {workingDir}");
        sb.AppendLine($"  outputRoot: {outputRoot}");
        sb.AppendLine($"  resolvedOutputRootPath: {resolvedOutputRootPath}");
        sb.AppendLine($"  resolvedOutputRootAbsolute: {outputRootAbs}");
        sb.AppendLine($"  outputCsvFileName: {outputCsvFileName}");
        sb.AppendLine($"  resolvedCsvPath: {resolvedCsvPath}");
        sb.AppendLine($"  resolvedCsvAbsolute: {csvAbs}");
        sb.AppendLine($"  videoPath: {videoPath}");
        sb.AppendLine($"  resolvedVideoPath: {resolvedVideo}");
        sb.AppendLine($"  monitorRunning: {monitorCoroutine != null}");
        sb.AppendLine($"  csvExists: {!string.IsNullOrWhiteSpace(csvAbs) && File.Exists(csvAbs)}");
        sb.AppendLine($"  csvNoDataWarningSeconds: {csvNoDataWarningSeconds}");
        sb.AppendLine($"  replayBootstrapped: {replayBootstrapped}");
        sb.AppendLine($"  lastCsvLength: {lastCsvLength}");
        sb.AppendLine($"  lastCsvWriteTimeUtc: {lastCsvWriteTimeUtc:O}");
        sb.AppendLine($"  startButtonBound: {startButton != null}, runtimeListenerAdded: {addedRuntimeListener}");
        bool processAlive = false;
        string processIdText = "<null>";
        if (pythonProcess != null)
        {
            try
            {
                processAlive = !pythonProcess.HasExited;
                processIdText = pythonProcess.Id.ToString();
            }
            catch (System.Exception e)
            {
                processIdText = "<unavailable: " + e.Message + ">";
            }
        }
        sb.AppendLine($"  processAlive: {processAlive}");
        sb.AppendLine($"  processId: {processIdText}");
        sb.AppendLine($"  resolvedAgent: {(resolvedAgent != null ? resolvedAgent.RobotKey + " / " + resolvedAgent.AgentGameObject?.name : "<null>")}");
        sb.AppendLine($"  targetAgentField: {(targetAgent != null ? targetAgent.RobotKey + " / " + targetAgent.AgentGameObject?.name : "<null>")}");
        sb.AppendLine($"  registryAvailable: {MimicAgentRegistry.Instance != null}");

        if (roboListDropdown != null && roboListDropdown.options != null)
        {
            sb.AppendLine($"  roboList options: {roboListDropdown.options.Count}, value: {roboListDropdown.value}");
        }

        if (csvListDropdown != null && csvListDropdown.options != null)
        {
            sb.AppendLine($"  csvList options: {csvListDropdown.options.Count}, value: {csvListDropdown.value}");
        }

        if (csvListFileBrowser != null)
        {
            sb.AppendLine($"  csvList selectedPath: {csvListFileBrowser.GetSelectedCsvPath()}");
        }

        Debug.Log(sb.ToString());
    }

    void Awake()
    {
        EnsureStartButtonListener();
        LogVerbose($"Awake: runBashOnClick={runBashOnClick}, monitorCsvOnClick={monitorCsvOnClick}, defaultRobotName='{defaultRobotName}', tcp={enableTcpStreaming}");

        // Attach onValueChanged on the RoboList dropdown so switching the
        // selection mid-session retargets to the newly chosen robot.
        ResolveRoboListReferences();
        if (roboListDropdown != null && (switchActiveRobotOnDropdownChange || filterCsvListByRobot))
        {
            roboListDropdown.onValueChanged.RemoveListener(OnRoboListChanged);
            roboListDropdown.onValueChanged.AddListener(OnRoboListChanged);
        }

        RefreshCsvListForSelectedRobot();
    }

    private IEnumerator Start()
    {
        LogVerbose("Start coroutine entered; waiting one frame before applying initial robot selection.");
        yield return null;
        ApplyInitialRobotSelectionState();
    }

    void OnEnable()
    {
        EnsureStartButtonListener();
        LogVerbose($"OnEnable: initialRobotSelectionApplied={initialRobotSelectionApplied}, activeInHierarchy={gameObject.activeInHierarchy}");

        if (!initialRobotSelectionApplied && isActiveAndEnabled)
        {
            StartCoroutine(ApplyInitialRobotSelectionStateNextFrame());
        }
    }

    void Update()
    {
        FlushUnityLogQueue();
        ConsumeRealtimeCsvQueues();
    }

    private void ConsumeRealtimeCsvQueues()
    {
        bool resetSeen = false;
        while (realtimeCsvResetQueue.TryDequeue(out string reason))
        {
            resetSeen = true;
            ResetRealtimeCsvReader(reason);
        }

        if (resetSeen)
        {
            while (realtimeCsvRowsQueue.TryDequeue(out _)) { }
        }

        int maxBatches = Mathf.Max(1, maxRealtimeCsvBatchesPerFrame);
        for (int i = 0; i < maxBatches && realtimeCsvRowsQueue.TryDequeue(out List<float[]> rows); i++)
        {
            ApplyRealtimeCsvRows(rows);
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
        StopBashProcessBlocking(clearCsvOnExit ? ResolveCsvAbsolutePath() : string.Empty);

        StopDebugLogWriter();
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
        if (roboListDropdown == null || roboListDropdown.options == null) return;
        if (newIndex < 0 || newIndex >= roboListDropdown.options.Count) return;

        string newLabel = roboListDropdown.options[newIndex].text;
        string newKey   = TryResolveRobotKeyQuiet(newLabel);
        RefreshCsvListForSelectedRobot(newKey, newLabel);

        if (!switchActiveRobotOnDropdownChange) return;
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

    private IEnumerator ApplyInitialRobotSelectionStateNextFrame()
    {
        yield return null;
        ApplyInitialRobotSelectionState();
    }

    private void ApplyInitialRobotSelectionState()
    {
        if (initialRobotSelectionApplied)
        {
            return;
        }

        ResolveRoboListReferences();
        ResolveCsvListReferences();

        if (roboListFileBrowser != null)
        {
            roboListFileBrowser.PopulateDropdown();
        }

        string selectedLabel = ResolveSelectedRobotName();
        string selectedKey = TryResolveRobotKeyQuiet(selectedLabel);

        if ((string.IsNullOrWhiteSpace(selectedLabel) || string.IsNullOrWhiteSpace(selectedKey)) &&
            TrySelectRoboListOption(defaultRobotName))
        {
            selectedLabel = ResolveSelectedRobotName();
            selectedKey = TryResolveRobotKeyQuiet(selectedLabel);
        }

        if (string.IsNullOrWhiteSpace(selectedLabel))
        {
            selectedLabel = defaultRobotName.Trim();
        }

        if (string.IsNullOrWhiteSpace(selectedKey))
        {
            selectedKey = TryResolveRobotKeyQuiet(defaultRobotName);
        }

        RefreshCsvListForSelectedRobot(selectedKey, selectedLabel);

        if (!string.IsNullOrWhiteSpace(selectedLabel) || !string.IsNullOrWhiteSpace(selectedKey))
        {
            ApplyRobotVisibility(selectedLabel, selectedKey);
            Debug.Log($"[StartInput] Initial robot selection applied: label='{selectedLabel}', key='{selectedKey}'.");
        }
        else
        {
            LogVerbose("ApplyInitialRobotSelectionState: no label/key resolved.");
        }

        initialRobotSelectionApplied = true;
    }

    private bool TrySelectRoboListOption(string robotKeyOrLabel)
    {
        if (roboListDropdown == null || roboListDropdown.options == null || roboListDropdown.options.Count == 0)
        {
            return false;
        }

        string desired = (robotKeyOrLabel ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(desired))
        {
            return false;
        }

        string desiredKey = TryResolveRobotKeyQuiet(desired);
        for (int i = 0; i < roboListDropdown.options.Count; i++)
        {
            string optionText = roboListDropdown.options[i].text?.Trim() ?? string.Empty;
            if (string.Equals(optionText, desired, System.StringComparison.OrdinalIgnoreCase))
            {
                roboListDropdown.SetValueWithoutNotify(i);
                roboListDropdown.RefreshShownValue();
                LogVerbose($"TrySelectRoboListOption: matched desired='{desired}' with option='{optionText}' at index={i}");
                return true;
            }

            string optionKey = TryResolveRobotKeyQuiet(optionText);
            if (!string.IsNullOrWhiteSpace(desiredKey) &&
                string.Equals(optionKey, desiredKey, System.StringComparison.OrdinalIgnoreCase))
            {
                roboListDropdown.SetValueWithoutNotify(i);
                roboListDropdown.RefreshShownValue();
                LogVerbose($"TrySelectRoboListOption: matched desiredKey='{desiredKey}' with option='{optionText}' at index={i}");
                return true;
            }
        }

        LogVerbose($"TrySelectRoboListOption: no match for '{desired}' (desiredKey='{desiredKey}').");
        return false;
    }

    // Called by Unity when the application is quitting (covers editor Stop, build exit, and
    // OS-level termination signals that Unity intercepts).
    void OnApplicationQuit()
    {
        StopCsvMonitor();
        StopBashProcessBlocking(clearCsvOnExit ? ResolveCsvAbsolutePath() : string.Empty);
        StopDebugLogWriter();
    }

    public void OnStartButtonClicked()
    {
        if (isStoppingPipeline)
        {
            Debug.LogWarning("[StartInput] Start ignored because the previous retargeting pipeline is still stopping.");
            return;
        }

        ApplyInitialRobotSelectionState();
        string selectedRaw = ResolveSelectedRobotName();
        string selectedKey = TryResolveRobotKeyQuiet(selectedRaw);
        Debug.LogWarning("[StartInput] Start button clicked.");
        AppendDebugLogLine("[StartInput] Start button clicked.");
        LogVerbose($"Start click: selectedRaw='{selectedRaw}', selectedKey='{selectedKey}', runBashOnClick={runBashOnClick}, monitorCsvOnClick={monitorCsvOnClick}");

        bool launchOk = true;
        if (runBashOnClick)
        {
            launchOk = StartOrRestartBashProcess();
        }

        bool monitorOk = true;
        if (monitorCsvOnClick && (!runBashOnClick || launchOk))
        {
            monitorOk = StartCsvMonitor();
        }
        else if (monitorCsvOnClick && !launchOk)
        {
            monitorOk = false;
        }

        if (launchOk && monitorOk && monitorCsvOnClick)
        {
            SetRealtimeDropdownsInteractable(false);
        }
        else
        {
            SetRealtimeDropdownsInteractable(true);
            if (launchOk && runBashOnClick && !monitorOk)
            {
                StopBashProcessAsync(null);
            }
        }
    }

    private void EnsureStartButtonListener()
    {
        if (startButton == null)
        {
            startButton = GetComponent<Button>();
        }

        if (startButton == null || HasPersistentStartHandler(startButton))
        {
            return;
        }

        startButton.onClick.RemoveListener(OnStartButtonClicked);
        startButton.onClick.AddListener(OnStartButtonClicked);
        addedRuntimeListener = true;
        LogVerbose($"EnsureStartButtonListener: listener bound on '{gameObject.name}'.");
    }

    public void StopStartPipeline()
    {
        StopCsvMonitor();
        EndActiveRealtimeCsv();
        SetRealtimeDropdownsInteractable(true);

        string csvToClear = clearCsvOnExit ? ResolveCsvAbsolutePath() : string.Empty;
        bool stopStarted = StopBashProcessAsync(csvToClear);
        if (!stopStarted && clearCsvOnExit)
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

    private bool StartOrRestartBashProcess()
    {
        if (pythonProcess != null && !pythonProcess.HasExited)
        {
            LogVerbose($"StartOrRestartBashProcess: existing process alive pid={pythonProcess.Id}, restartBashIfRunning={restartBashIfRunning}");
            if (!restartBashIfRunning)
            {
                Debug.Log("Bash 进程已在运行，跳过重复启动。");
                return true;

            }

            StopBashProcessBlocking(null);
        }

        string resolvedWorkingDir = ResolveWorkingDirectoryPath();
        if (string.IsNullOrWhiteSpace(resolvedWorkingDir) || !Directory.Exists(resolvedWorkingDir))
        {
            Debug.LogError($"Bash 工作目录无效: {resolvedWorkingDir}");
            return false;

        }

        string executable = (bashExecutable ?? string.Empty).Trim();
        string executablePrefixArgs = string.Empty;
        ParseExecutable(executable, out executable, out executablePrefixArgs);

        if (string.IsNullOrWhiteSpace(executable))
        {
            Debug.LogError("bashExecutable 为空，无法启动 Bash 命令。");
            return false;

        }

        string selectedRobot = ResolveSelectedRobotName();
        if (string.IsNullOrWhiteSpace(selectedRobot))
        {
            selectedRobot = defaultRobotName.Trim();
        }

        if (string.IsNullOrWhiteSpace(selectedRobot))
        {
            Debug.LogError("未能解析 RoboList 当前机器人名称，请检查下拉列表配置。");
            return false;

        }

        string resolvedRobot = ResolveRobotNameForWham(selectedRobot);
        if (string.IsNullOrWhiteSpace(resolvedRobot))
        {
            Debug.LogError($"当前机器人不受 WHAM 支持: {selectedRobot}");
            return false;

        }

        resolvedOutputRootPath = ResolveOutputRootAbsolutePath(resolvedWorkingDir);
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            Debug.LogError("OUTPUT_ROOT 无效，无法启动 Bash 命令。");
            return false;

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
        string resolvedVideoPath = ResolveVideoPath(resolvedWorkingDir);
        string commandArguments = BuildCommandArguments(executablePrefixArgs, resolvedScriptPath);
        ResetDebugLogFile();
        bool effectiveRecordGmrVideo = recordGmrVideo && !(enableTcpStreaming && disablePreviewVideoWhenTcpStreaming);
        bool effectiveRecordWhamVideo = recordWhamVideo && !(enableTcpStreaming && disablePreviewVideoWhenTcpStreaming);
        string requestedGmrTorchDevice = string.IsNullOrWhiteSpace(gmrTorchDevice) ? "cpu" : gmrTorchDevice.Trim();
        string launchGmrTorchDevice = "cpu";
        if (!string.Equals(requestedGmrTorchDevice, "cpu", System.StringComparison.OrdinalIgnoreCase))
        {
            Debug.LogWarning(
                $"[StartInput] GMR_TORCH_DEVICE requested '{requestedGmrTorchDevice}', forcing 'cpu' for GMR postprocessing " +
                "to avoid CUDA NVRTC architecture errors. WHAM still uses its configured CUDA device.");
            AppendDebugLogLine($"[StartInput] GMR_TORCH_DEVICE forced cpu (requested {requestedGmrTorchDevice}).");
        }

        LogVerbose(
            $"Resolved launch config: selectedRobot='{selectedRobot}', resolvedRobot='{resolvedRobot}', " +
            $"workingDir='{resolvedWorkingDir}', script='{resolvedScriptPath}', outputRoot='{resolvedOutputRootPath}', " +
            $"csv='{resolvedCsvPath}', video='{resolvedVideoPath}', tcp={enableTcpStreaming}, track={track}, " +
            $"recordGmr={effectiveRecordGmrVideo} (raw={recordGmrVideo}), recordWham={effectiveRecordWhamVideo} (raw={recordWhamVideo}), " +
            $"gmrTorchDevice='{launchGmrTorchDevice}' (requested='{requestedGmrTorchDevice}')");

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
            startInfo.EnvironmentVariables["RECORD_GMRVIDEO"] = effectiveRecordGmrVideo ? "1" : "0";
            startInfo.EnvironmentVariables["RECORD_WHAMVIDEO"] = effectiveRecordWhamVideo ? "1" : "0";
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
            startInfo.EnvironmentVariables["GMR_TORCH_DEVICE"] = launchGmrTorchDevice;
            startInfo.EnvironmentVariables["TRACK"] = track ? "1" : "0";
            startInfo.EnvironmentVariables["TCP"] = enableTcpStreaming ? "1" : "0";
            Process process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };
            lock (processStateLock)
            {
                pythonProcess = process;
            }

            process.Exited += (sender, __) =>
            {
                var exitedProcess = sender as Process;
                string exitCode = "<unknown>";
                try
                {
                    if (exitedProcess != null)
                    {
                        exitCode = exitedProcess.ExitCode.ToString(CultureInfo.InvariantCulture);
                    }
                }
                catch
                {
                    // ExitCode can throw if the process object is already disposed.
                }

                QueueUnityLog($"Bash process exited, ExitCode={exitCode}");
            };

            if (logBashOutput)
            {
                process.OutputDataReceived += (_, args) =>
                {
                    if (!string.IsNullOrWhiteSpace(args.Data))
                    {
                        string line = $"[Pipeline] {args.Data}";
                        QueueUnityLog(line);
                    }
                };

                process.ErrorDataReceived += (_, args) =>
                {
                    if (!string.IsNullOrWhiteSpace(args.Data))
                    {
                        string line = $"[Pipeline-ERR] {args.Data}";
                        QueueUnityLog(line, QueuedUnityLogType.Warning);
                    }
                };
            }

            process.Start();
            LogVerbose($"Python process started successfully. pid={process.Id}");

            if (logBashOutput)
            {
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            }

            Debug.Log($"已启动 Bash 命令: {executable} {commandArguments}");
            string pipelineEnvLine =
                "[StartInput] Pipeline env: " +
                "ROBOT=" + resolvedRobot +
                " (raw=" + selectedRobot + ")" +
                ", VIDEO=" + (string.IsNullOrWhiteSpace(resolvedVideoPath) ? "<run.ps1 default>" : resolvedVideoPath) +
                ", OUTPUT_ROOT=" + resolvedOutputRootPath +
                ", CSV=" + resolvedCsvPath +
                ", TCP=" + (enableTcpStreaming ? "1" : "0") +
                ", TRACK=" + (track ? "1" : "0") +
                ", RECORD_WHAMVIDEO=" + (effectiveRecordWhamVideo ? "1" : "0") +
                ", RECORD_GMRVIDEO=" + (effectiveRecordGmrVideo ? "1" : "0");
            Debug.LogWarning(pipelineEnvLine);
            AppendDebugLogLine(pipelineEnvLine);
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"启动 Bash 命令失败: {e.Message}");
            return false;
        }
    }

    private Process DetachPythonProcess()
    {
        lock (processStateLock)
        {
            Process process = pythonProcess;
            pythonProcess = null;
            return process;
        }
    }

    private bool StopBashProcessAsync(string csvToClearAfterStop = null)
    {
        Process processToStop = DetachPythonProcess();
        if (processToStop == null)
        {
            return false;
        }

        isStoppingPipeline = true;
        QueueUnityLog($"[StopBash] Background stop started for pid={SafeProcessId(processToStop)}.");

        var stopThread = new Thread(() =>
        {
            try
            {
                StopBashProcessCore(processToStop);
                TryClearCsvPath(csvToClearAfterStop, "StopStartPipeline/background");
            }
            finally
            {
                isStoppingPipeline = false;
                QueueUnityLog("[StopBash] Background stop completed.");
            }
        })
        {
            Name = "StartInput-StopPipeline",
            IsBackground = true
        };
        stopThread.Start();
        return true;
    }

    private void StopBashProcessBlocking(string csvToClearAfterStop = null)
    {
        Process processToStop = DetachPythonProcess();
        if (processToStop == null)
        {
            TryClearCsvPath(csvToClearAfterStop, "StopBashProcessBlocking");
            return;
        }

        isStoppingPipeline = true;
        try
        {
            StopBashProcessCore(processToStop);
            TryClearCsvPath(csvToClearAfterStop, "StopBashProcessBlocking");
        }
        finally
        {
            isStoppingPipeline = false;
        }
    }

    private void StopBashProcessCore(Process processToStop)
    {
        if (processToStop == null)
        {
            return;
        }

        try
        {
            if (!processToStop.HasExited)
            {
                int pid = processToStop.Id;
                if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
                {
                    KillProcessGroupLinuxQueued(pid);
                }
                else
                {
                    KillProcessTreeWindowsQueued(pid);
                }

                processToStop.WaitForExit(2000);
            }
        }
        catch (System.Exception e)
        {
            QueueUnityLog($"[StopBash] Stop process failed: {e.Message}", QueuedUnityLogType.Warning);
        }
        finally
        {
            try { processToStop.Dispose(); } catch { }
        }

        if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
        {
            KillOrphansByNameQueued("demo_stream_mt");
            KillOrphansByNameQueued("smplx_to_robot_stream");
        }
        else
        {
            KillOrphansByNameQueued("handle_wham_gmr");
        }
    }

    private static int SafeProcessId(Process process)
    {
        try { return process != null ? process.Id : -1; }
        catch { return -1; }
    }

    private void TryClearCsvPath(string csvPath, string caller)
    {
        if (string.IsNullOrWhiteSpace(csvPath) || !File.Exists(csvPath))
        {
            return;
        }

        try
        {
            File.Delete(csvPath);
            QueueUnityLog($"[{caller}] live_motion.csv cleared: {csvPath}");
        }
        catch (System.Exception e)
        {
            QueueUnityLog($"[{caller}] Failed to clear live_motion.csv ({csvPath}): {e.Message}", QueuedUnityLogType.Warning);
        }
    }

    private void KillProcessTreeWindowsQueued(int pid)
    {
        try
        {
            var killInfo = new ProcessStartInfo
            {
                FileName = "taskkill",
                Arguments = $"/T /F /PID {pid}",
                UseShellExecute = false,
                RedirectStandardOutput = false,
                RedirectStandardError = false,
                CreateNoWindow = true
            };

            using (Process killer = Process.Start(killInfo))
            {
                killer?.WaitForExit(5000);
            }

            QueueUnityLog($"[StopBash] taskkill /T /F /PID {pid} executed");
        }
        catch (System.Exception e)
        {
            QueueUnityLog($"[StopBash] taskkill failed (pid={pid}): {e.Message}", QueuedUnityLogType.Warning);
        }
    }

    private void KillProcessGroupLinuxQueued(int leaderPid)
    {
        try
        {
            var killInfo = new ProcessStartInfo
            {
                FileName = "kill",
                Arguments = $"-9 -{leaderPid}",
                UseShellExecute = false,
                RedirectStandardOutput = false,
                RedirectStandardError = false,
                CreateNoWindow = true
            };

            using (Process killer = Process.Start(killInfo))
            {
                killer?.WaitForExit(3000);
            }

            QueueUnityLog($"[StopBash] kill -9 -{leaderPid} executed");
        }
        catch (System.Exception e)
        {
            QueueUnityLog($"[StopBash] kill process group failed (pgid={leaderPid}): {e.Message}", QueuedUnityLogType.Warning);
        }
    }

    private void KillOrphansByNameQueued(string scriptNameFragment)
    {
        try
        {
            if (System.Environment.OSVersion.Platform == System.PlatformID.Unix)
            {
                var pkillInfo = new ProcessStartInfo
                {
                    FileName = "pkill",
                    Arguments = $"-9 -f \"{scriptNameFragment}\"",
                    UseShellExecute = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false,
                    CreateNoWindow = true
                };

                using (Process pkill = Process.Start(pkillInfo))
                {
                    pkill?.WaitForExit(3000);
                }

                QueueUnityLog($"[StopBash] pkill -9 -f \"{scriptNameFragment}\" executed");
            }
            else
            {
                string fragment = (scriptNameFragment ?? string.Empty).Replace("'", "''");
                string psCommand =
                    "$fragment = '" + fragment + "'; " +
                    "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | " +
                    "Where-Object { $_.CommandLine -like ('*' + $fragment + '*') } | " +
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }";
                string encodedCommand = System.Convert.ToBase64String(Encoding.Unicode.GetBytes(psCommand));

                var powershellInfo = new ProcessStartInfo
                {
                    FileName = "powershell",
                    Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand " + encodedCommand,
                    UseShellExecute = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError = false,
                    CreateNoWindow = true
                };

                using (Process powershell = Process.Start(powershellInfo))
                {
                    powershell?.WaitForExit(5000);
                }

                QueueUnityLog($"[StopBash] PowerShell cleanup for python *{scriptNameFragment}* executed");
            }
        }
        catch (System.Exception e)
        {
            QueueUnityLog($"[StopBash] orphan cleanup '{scriptNameFragment}': {e.Message}", QueuedUnityLogType.Warning);
        }
    }

    private void StopBashProcessLegacy()
    {
        if (pythonProcess == null)
        {
            return;
        }

        LogVerbose("StopBashProcess invoked.");

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
    /// On Linux uses <c>pkill -9 -f</c>; on Windows uses PowerShell/CIM to
    /// terminate python.exe/pythonw.exe instances matching the fragment.
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
                // WMIC is not installed on current Windows builds by default.
                // Use PowerShell/CIM so stale WHAM/GMR workers do not keep the
                // webcam or TCP port occupied after Stop.
                string fragment = (scriptNameFragment ?? string.Empty).Replace("'", "''");
                string psCommand =
                    "$fragment = '" + fragment + "'; " +
                    "Get-CimInstance Win32_Process -Filter \"Name = 'python.exe' OR Name = 'pythonw.exe'\" | " +
                    "Where-Object { $_.CommandLine -like ('*' + $fragment + '*') } | " +
                    "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }";
                string encodedCommand = System.Convert.ToBase64String(Encoding.Unicode.GetBytes(psCommand));

                var powershellInfo = new ProcessStartInfo
                {
                    FileName               = "powershell",
                    Arguments              = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand " + encodedCommand,
                    UseShellExecute        = false,
                    RedirectStandardOutput = false,
                    RedirectStandardError  = false,
                    CreateNoWindow         = true
                };

                using (Process powershell = Process.Start(powershellInfo))
                {
                    powershell?.WaitForExit(5000);
                }

                Debug.Log($"[StopBash] PowerShell cleanup for python *{scriptNameFragment}* executed");
            }
        }
        catch (System.Exception e)
        {
            Debug.Log($"[StopBash] orphan cleanup '{scriptNameFragment}': {e.Message}");
        }
    }

    private bool StartCsvMonitor()
    {
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            string baseDir = ResolveWorkingDirectoryPath();
            resolvedOutputRootPath = ResolveOutputRootAbsolutePath(baseDir);
        }

        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            Debug.LogError("OUTPUT_ROOT 无效，无法开始监听实时 CSV。");
            return false;
        }

        string csvName = ResolveCsvRelativePath();
        resolvedCsvPath = ResolvePathFromBaseDirectory(csvName, resolvedOutputRootPath);
        if (string.IsNullOrWhiteSpace(resolvedCsvPath))
        {
            Debug.LogError("实时 CSV 路径为空，无法开始监听。");
            return false;
        }

        StopCsvMonitor();
        EndActiveRealtimeCsv();

        activeRealtimeMimicAgent = ResolveActiveAgent();
        activeRealtimeCsvAgent = activeRealtimeMimicAgent as IRealtimeCsvMimicAgent;
        if (activeRealtimeMimicAgent == null || activeRealtimeCsvAgent == null)
        {
            Debug.LogError($"[StartInput] Selected robot '{ResolveSelectedRobotName()}' does not provide realtime CSV ingestion.");
            activeRealtimeMimicAgent = null;
            activeRealtimeCsvAgent = null;
            return false;
        }

        lastCsvLength = -1;
        lastCsvWriteTimeUtc = System.DateTime.MinValue;
        replayBootstrapped = false;
        csvMissingLogged = false;
        csvMonitorStartUtc = System.DateTime.UtcNow;
        lastCsvNoDataWarningUtc = System.DateTime.MinValue;
        firstNonEmptyCsvLogged = false;
        csvReadOffset = 0;
        csvPendingText = string.Empty;
        realtimeSourceRowsRead = 0;
        pendingRealtimeRows.Clear();
        DrainRealtimeCsvQueues();
        if (writeDebugLogFile && string.IsNullOrWhiteSpace(resolvedDebugLogPath))
        {
            resolvedDebugLogPath = ResolveDebugLogPath();
            EnsureDebugLogWriter();
        }

        StartCsvWorker(resolvedCsvPath, activeRealtimeCsvAgent.ExpectedCsvColumns, activeRealtimeMimicAgent.RobotKey);
        LogVerbose(
            $"CSV worker started: pollInterval={csvPollInterval}, path='{resolvedCsvPath}', realtimeWarmupSourceRows={realtimeWarmupSourceRows}, " +
            $"expectedColumns={activeRealtimeCsvAgent.ExpectedCsvColumns}, robot='{activeRealtimeMimicAgent.RobotKey}'");

        Debug.Log($"开始监听实时 CSV: {resolvedCsvPath}");
        return true;
    }


    private void StopCsvMonitor()
    {
        csvWorkerRunning = false;
        Thread worker = csvWorkerThread;
        if (worker != null && worker.IsAlive)
        {
            try { worker.Join(250); } catch { }
        }
        csvWorkerThread = null;

        if (monitorCoroutine != null)
        {
            LogVerbose("CSV monitor stopped.");
            StopCoroutine(monitorCoroutine);
            monitorCoroutine = null;
        }

        DrainRealtimeCsvQueues();
    }

    private void DrainRealtimeCsvQueues()
    {
        while (realtimeCsvRowsQueue.TryDequeue(out List<float[]> _)) { }
        while (realtimeCsvResetQueue.TryDequeue(out string _)) { }
    }

    private void StartCsvWorker(string csvPath, int expectedColumns, string robotKey)
    {
        csvWorkerRunning = true;
        int sleepMs = System.Math.Max(20, (int)System.Math.Round(csvPollInterval * 1000f));
        float emptyWarningSeconds = System.Math.Max(1f, csvNoDataWarningSeconds);
        bool logChanges = logCsvChangeEvents;

        csvWorkerThread = new Thread(() => CsvWorkerLoop(csvPath, expectedColumns, robotKey, sleepMs, logChanges, emptyWarningSeconds))
        {
            Name = "StartInput-CsvWorker",
            IsBackground = true
        };
        csvWorkerThread.Start();
    }

    private void CsvWorkerLoop(string csvPath, int expectedColumns, string robotKey, int sleepMs, bool logChanges, float emptyWarningSeconds)
    {
        bool missingLogged = false;
        bool firstDataLogged = false;
        string lastReadError = string.Empty;
        long readOffset = 0L;
        string pendingText = string.Empty;
        long previousLength = -1L;
        System.DateTime previousWriteTimeUtc = System.DateTime.MinValue;
        System.DateTime monitorStartUtc = System.DateTime.UtcNow;
        System.DateTime lastEmptyWarningUtc = System.DateTime.MinValue;

        QueueUnityLog(
            $"[StartInput] Realtime CSV worker active: robot='{robotKey}', expectedColumns={expectedColumns}, csv='{csvPath}', pollMs={sleepMs}");

        while (csvWorkerRunning)
        {
            try
            {
                if (!File.Exists(csvPath))
                {
                    if (!missingLogged)
                    {
                        QueueUnityLog($"CSV file does not exist yet, waiting: {csvPath}", QueuedUnityLogType.Warning);
                        missingLogged = true;
                    }

                    Thread.Sleep(sleepMs);
                    continue;
                }

                missingLogged = false;
                FileInfo info = new FileInfo(csvPath);
                long currentLength = info.Length;
                System.DateTime currentWriteTimeUtc = info.LastWriteTimeUtc;

                if (currentLength < readOffset)
                {
                    readOffset = 0L;
                    pendingText = string.Empty;
                    firstDataLogged = false;
                    monitorStartUtc = System.DateTime.UtcNow;
                    realtimeCsvResetQueue.Enqueue("CSV truncated or recreated");
                }

                if (currentLength <= 0)
                {
                    if (currentLength != previousLength || currentWriteTimeUtc != previousWriteTimeUtc)
                    {
                        if (logChanges)
                        {
                            QueueUnityLog($"[StartInput] CSV exists but is empty: path='{csvPath}', writeTimeUtc={currentWriteTimeUtc:O}");
                        }

                        previousLength = currentLength;
                        previousWriteTimeUtc = currentWriteTimeUtc;
                    }

                    System.DateTime nowUtc = System.DateTime.UtcNow;
                    if ((nowUtc - monitorStartUtc).TotalSeconds >= emptyWarningSeconds &&
                        (lastEmptyWarningUtc == System.DateTime.MinValue || (nowUtc - lastEmptyWarningUtc).TotalSeconds >= emptyWarningSeconds))
                    {
                        bool processAlive = false;
                        int pid = -1;
                        Process process = pythonProcess;
                        try
                        {
                            processAlive = process != null && !process.HasExited;
                            pid = process != null ? process.Id : -1;
                        }
                        catch { }

                        QueueUnityLog(
                            $"[StartInput] live_motion.csv is still empty. elapsed={(nowUtc - monitorStartUtc).TotalSeconds:F1}s, " +
                            $"processAlive={processAlive}, pid={pid}, csv='{csvPath}'. Check [Pipeline]/[Pipeline-ERR] logs.",
                            QueuedUnityLogType.Warning);
                        lastEmptyWarningUtc = nowUtc;
                    }

                    Thread.Sleep(sleepMs);
                    continue;
                }

                if (currentLength > readOffset)
                {
                    List<float[]> rows = ReadNewRealtimeCsvRowsFromPath(csvPath, currentLength, expectedColumns, ref readOffset, ref pendingText);
                    if (rows.Count > 0)
                    {
                        if (!firstDataLogged)
                        {
                            firstDataLogged = true;
                            QueueUnityLog($"[StartInput] Realtime CSV has data: {csvPath} ({currentLength} bytes)");
                        }

                        if (logChanges)
                        {
                            QueueUnityLog($"[StartInput] CSV appended: rows={rows.Count}, bytes={currentLength}, offset={readOffset}, path='{csvPath}'");
                        }

                        realtimeCsvRowsQueue.Enqueue(rows);
                    }
                }

                previousLength = currentLength;
                previousWriteTimeUtc = currentWriteTimeUtc;
                lastReadError = string.Empty;
            }
            catch (System.Exception e)
            {
                string message = e.Message ?? string.Empty;
                if (!string.Equals(message, lastReadError, System.StringComparison.Ordinal))
                {
                    QueueUnityLog($"[StartInput] CSV worker read failed: {message}", QueuedUnityLogType.Warning);
                    lastReadError = message;
                }
            }

            Thread.Sleep(sleepMs);
        }
    }

    private List<float[]> ReadNewRealtimeCsvRowsFromPath(string csvPath, long currentLength, int expectedColumns, ref long readOffset, ref string pendingText)
    {
        var rows = new List<float[]>();
        if (expectedColumns <= 0 || currentLength <= readOffset)
        {
            return rows;
        }

        long bytesToReadLong = currentLength - readOffset;
        if (bytesToReadLong <= 0 || bytesToReadLong > int.MaxValue)
        {
            return rows;
        }

        byte[] buffer = new byte[(int)bytesToReadLong];
        int bytesRead;
        using (FileStream fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        {
            fs.Seek(readOffset, SeekOrigin.Begin);
            bytesRead = fs.Read(buffer, 0, buffer.Length);
            readOffset += bytesRead;
        }

        if (bytesRead <= 0)
        {
            return rows;
        }

        string text = pendingText + Encoding.UTF8.GetString(buffer, 0, bytesRead);
        text = text.Replace("\r\n", "\n").Replace('\r', '\n');
        bool hasTrailingNewline = text.EndsWith("\n", System.StringComparison.Ordinal);
        string[] lines = text.Split('\n');
        int completeLineCount = hasTrailingNewline ? lines.Length : System.Math.Max(0, lines.Length - 1);
        pendingText = hasTrailingNewline || lines.Length == 0 ? string.Empty : lines[lines.Length - 1];

        for (int i = 0; i < completeLineCount; i++)
        {
            if (TryParseRealtimeCsvLine(lines[i], expectedColumns, out float[] row))
            {
                rows.Add(row);
            }
        }

        return rows;
    }

    private IEnumerator MonitorCsvCoroutine()
    {
        while (true)
        {
            if (!File.Exists(resolvedCsvPath))
            {
                if (!csvMissingLogged)
                {
                    Debug.LogWarning($"CSV file does not exist yet, waiting: {resolvedCsvPath}");
                    csvMissingLogged = true;
                }

                yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
                continue;
            }

            csvMissingLogged = false;

            long currentLength = 0L;
            System.DateTime currentWriteTimeUtc = System.DateTime.MinValue;
            bool csvReadFailed = false;
            try
            {
                FileInfo info = new FileInfo(resolvedCsvPath);
                currentLength = info.Length;
                currentWriteTimeUtc = info.LastWriteTimeUtc;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"Failed to read CSV file info: {e.Message}");
                csvReadFailed = true;
            }

            if (csvReadFailed)
            {
                yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
                continue;
            }

            if (currentLength < csvReadOffset)
            {
                ResetRealtimeCsvReader("CSV truncated or recreated");
            }

            if (currentLength <= 0)
            {
                if (currentLength != lastCsvLength || currentWriteTimeUtc != lastCsvWriteTimeUtc)
                {
                    if (logCsvChangeEvents)
                    {
                        LogVerbose($"CSV exists but is empty: path='{resolvedCsvPath}', writeTimeUtc={currentWriteTimeUtc:O}");
                    }

                    lastCsvLength = currentLength;
                    lastCsvWriteTimeUtc = currentWriteTimeUtc;
                }

                MaybeWarnCsvStillEmpty();
                yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
                continue;
            }

            if (currentLength > csvReadOffset)
            {
                List<float[]> rows = ReadNewRealtimeCsvRows(currentLength);
                if (rows.Count > 0)
                {
                    if (!firstNonEmptyCsvLogged)
                    {
                        firstNonEmptyCsvLogged = true;
                        Debug.Log($"[StartInput] Realtime CSV has data: {resolvedCsvPath} ({currentLength} bytes)");
                    }

                    if (logCsvChangeEvents)
                    {
                        LogVerbose($"CSV appended: rows={rows.Count}, bytes={currentLength}, offset={csvReadOffset}, path='{resolvedCsvPath}'");
                    }

                    ApplyRealtimeCsvRows(rows);
                }
            }

            lastCsvLength = currentLength;
            lastCsvWriteTimeUtc = currentWriteTimeUtc;
            yield return new WaitForSeconds(Mathf.Max(0.02f, csvPollInterval));
        }
    }

    private List<float[]> ReadNewRealtimeCsvRows(long currentLength)
    {
        var rows = new List<float[]>();
        if (activeRealtimeCsvAgent == null || currentLength <= csvReadOffset)
        {
            return rows;
        }

        long bytesToReadLong = currentLength - csvReadOffset;
        if (bytesToReadLong <= 0 || bytesToReadLong > int.MaxValue)
        {
            return rows;
        }

        byte[] buffer = new byte[(int)bytesToReadLong];
        int bytesRead;
        using (FileStream fs = new FileStream(resolvedCsvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        {
            fs.Seek(csvReadOffset, SeekOrigin.Begin);
            bytesRead = fs.Read(buffer, 0, buffer.Length);
            csvReadOffset += bytesRead;
        }

        if (bytesRead <= 0)
        {
            return rows;
        }

        string text = csvPendingText + Encoding.UTF8.GetString(buffer, 0, bytesRead);
        text = text.Replace("\r\n", "\n").Replace('\r', '\n');
        bool hasTrailingNewline = text.EndsWith("\n", System.StringComparison.Ordinal);
        string[] lines = text.Split('\n');
        int completeLineCount = hasTrailingNewline ? lines.Length : Mathf.Max(0, lines.Length - 1);
        csvPendingText = hasTrailingNewline || lines.Length == 0 ? string.Empty : lines[lines.Length - 1];

        for (int i = 0; i < completeLineCount; i++)
        {
            if (TryParseRealtimeCsvLine(lines[i], activeRealtimeCsvAgent.ExpectedCsvColumns, out float[] row))
            {
                rows.Add(row);
            }
        }

        return rows;
    }

    private bool TryParseRealtimeCsvLine(string line, int expectedColumns, out float[] row)
    {
        row = null;
        if (string.IsNullOrWhiteSpace(line) || expectedColumns <= 0)
        {
            return false;
        }

        string[] tokens = line.Split(',');
        if (tokens.Length < expectedColumns)
        {
            return false;
        }

        row = new float[expectedColumns];
        for (int i = 0; i < expectedColumns; i++)
        {
            string token = tokens[i].Trim();
            if (!float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) &&
                !float.TryParse(token, out value))
            {
                row = null;
                return false;
            }

            row[i] = value;
        }

        return true;
    }

    private void ApplyRealtimeCsvRows(List<float[]> rows)
    {
        if (rows == null || rows.Count == 0)
        {
            return;
        }

        if (activeRealtimeMimicAgent == null || activeRealtimeCsvAgent == null)
        {
            Debug.LogWarning("[StartInput] Realtime CSV rows arrived but no realtime-capable agent is active.");
            return;
        }

        realtimeSourceRowsRead += rows.Count;

        if (!replayBootstrapped)
        {
            pendingRealtimeRows.AddRange(rows);
            int warmupRows = Mathf.Max(1, realtimeWarmupSourceRows);
            if (pendingRealtimeRows.Count < warmupRows)
            {
                return;
            }

            activeRealtimeCsvAgent.BeginRealtimeCsv();
            activeRealtimeMimicAgent.UseExternalReplayData = true;
            activeRealtimeMimicAgent.ReplayMode = true;
            if (!activeRealtimeCsvAgent.AppendRealtimeCsvRows(pendingRealtimeRows))
            {
                Debug.LogWarning("[StartInput] Initial realtime CSV append produced no frames.");
                return;
            }

            pendingRealtimeRows.Clear();
            if (MimicAgentRegistry.Instance != null)
            {
                MimicAgentRegistry.Instance.SetActiveTarget(activeRealtimeMimicAgent);
            }

            if (restartEpisodeOnFirstCsv)
            {
                activeRealtimeMimicAgent.RequestEndEpisode();
            }

            replayBootstrapped = true;
            Debug.Log(
                $"[StartInput] Realtime CSV playback started for '{activeRealtimeMimicAgent.RobotKey}' after {realtimeSourceRowsRead} source rows: {resolvedCsvPath}. " +
                $"expectedColumns={activeRealtimeCsvAgent.ExpectedCsvColumns}, warmupRows={Mathf.Max(1, realtimeWarmupSourceRows)}, " +
                $"selectedRobot='{ResolveSelectedRobotName()}', bufferedBatches={realtimeCsvRowsQueue.Count}, rootMapping=csv[0..2]->Unity(-y,z,x), quat=csv[3..6]->Unity(-y,z,x,-w)");
            return;
        }

        activeRealtimeCsvAgent.AppendRealtimeCsvRows(rows);
    }

    private void ResetRealtimeCsvReader(string reason)
    {
        LogVerbose($"Reset realtime CSV reader: {reason}");
        csvReadOffset = 0;
        csvPendingText = string.Empty;
        realtimeSourceRowsRead = 0;
        replayBootstrapped = false;
        firstNonEmptyCsvLogged = false;
        pendingRealtimeRows.Clear();
        activeRealtimeCsvAgent?.BeginRealtimeCsv();
    }

    private void EndActiveRealtimeCsv()
    {
        activeRealtimeCsvAgent?.EndRealtimeCsv();
        activeRealtimeCsvAgent = null;
        activeRealtimeMimicAgent = null;
        pendingRealtimeRows.Clear();
        csvPendingText = string.Empty;
        csvReadOffset = 0;
        realtimeSourceRowsRead = 0;
        replayBootstrapped = false;
    }

    private void MaybeWarnCsvStillEmpty()
    {
        float threshold = Mathf.Max(1f, csvNoDataWarningSeconds);
        System.DateTime now = System.DateTime.UtcNow;
        if (csvMonitorStartUtc == System.DateTime.MinValue ||
            (now - csvMonitorStartUtc).TotalSeconds < threshold)
        {
            return;
        }

        if (lastCsvNoDataWarningUtc != System.DateTime.MinValue &&
            (now - lastCsvNoDataWarningUtc).TotalSeconds < threshold)
        {
            return;
        }

        lastCsvNoDataWarningUtc = now;

        bool processAlive = false;
        int processId = -1;
        if (pythonProcess != null)
        {
            try
            {
                processAlive = !pythonProcess.HasExited;
                processId = pythonProcess.Id;
            }
            catch
            {
                processAlive = false;
            }
        }

        string warning =
            "[StartInput] live_motion.csv is still empty. " +
            $"elapsed={(now - csvMonitorStartUtc).TotalSeconds:F1}s, " +
            $"processAlive={processAlive}, pid={(processId >= 0 ? processId.ToString() : "<none>")}, " +
            $"csv='{resolvedCsvPath}'. " +
            "Check [Pipeline]/[Pipeline-ERR] logs for [Input], [Detect], [WHAM], [GMR], and [TCP] messages.";
        Debug.LogWarning(warning);
        AppendDebugLogLine(warning);
    }

    /// <summary>
    /// Resolve the agent that should receive live retargeting data. RoboList is
    /// authoritative; a selected robot key must match the registered agent key.
    /// </summary>
    private IMimicAgent ResolveActiveAgent()
    {
        string selectedRaw = ResolveSelectedRobotName();
        string selectedKey = !string.IsNullOrWhiteSpace(selectedRaw)
            ? ResolveRobotNameForWham(selectedRaw)
            : string.Empty;
        if (string.IsNullOrWhiteSpace(selectedKey))
        {
            selectedKey = ResolveRobotNameForWham(defaultRobotName);
        }

        LogVerbose($"ResolveActiveAgent: selectedRaw='{selectedRaw}', selectedKey='{selectedKey}', defaultRobotName='{defaultRobotName}'");

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

            if (targetAgentBehaviour is IMimicAgent pinned &&
                pinned.AgentGameObject != null &&
                string.Equals(pinned.RobotKey, selectedKey, System.StringComparison.OrdinalIgnoreCase))
            {
                targetAgent = pinned;
                return targetAgent;
            }

            if (lastResolvedRobotKey != selectedKey)
            {
                Debug.LogError($"[StartInput] No IMimicAgent registered for selected robot '{selectedKey}'. Live CSV will not be routed to another robot.");
                lastResolvedRobotKey = selectedKey;
            }

            targetAgent = null;
            return null;
        }

        if (!string.IsNullOrWhiteSpace(selectedRaw) || !string.IsNullOrWhiteSpace(defaultRobotName))
        {
            Debug.LogError($"[StartInput] Unable to resolve a registered IMimicAgent for selected robot '{selectedRaw}' / default '{defaultRobotName}'.");
            targetAgent = null;
            return null;
        }

        if (targetAgentBehaviour is IMimicAgent legacyPinned && legacyPinned.AgentGameObject != null)
        {
            targetAgent = legacyPinned;
            return targetAgent;
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
        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrWhiteSpace(projectRoot))
        {
            try
            {
                string projectCandidate = Path.GetFullPath(Path.Combine(projectRoot, normalized));
                tried.Add(projectCandidate);
                firstCandidate = projectCandidate;
                if (Directory.Exists(projectCandidate)) return projectCandidate;
            }
            catch
            {
                // Continue with the historical Assets-relative roots below.
            }
        }

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

    private void SetRealtimeDropdownsInteractable(bool interactable)
    {
        ResolveRoboListReferences();
        ResolveCsvListReferences();

        if (roboListDropdown != null)
        {
            roboListDropdown.interactable = interactable;
        }

        if (csvListDropdown != null)
        {
            csvListDropdown.interactable = interactable;
        }

        LogVerbose($"Realtime dropdowns interactable={interactable}");
    }

    private void RefreshCsvListForSelectedRobot()
    {
        if (!filterCsvListByRobot)
        {
            return;
        }

        string selectedRobot = ResolveSelectedRobotName();
        string resolvedRobotKey = ResolveRobotNameForWham(selectedRobot);
        RefreshCsvListForSelectedRobot(resolvedRobotKey, selectedRobot);
    }

    private void RefreshCsvListForSelectedRobot(string resolvedRobotKey, string selectedLabel)
    {
        if (!filterCsvListByRobot)
        {
            return;
        }

        ResolveCsvListReferences();
        if (csvListFileBrowser == null)
        {
            return;
        }

        string filterKey = !string.IsNullOrWhiteSpace(resolvedRobotKey)
            ? resolvedRobotKey
            : (selectedLabel ?? string.Empty).Trim();
        csvListFileBrowser.SetCsvRobotFilter(filterKey);
        LogVerbose($"RefreshCsvListForSelectedRobot: label='{selectedLabel}', resolvedKey='{resolvedRobotKey}', filterKey='{filterKey}'");

        if (csvListDropdown == null)
        {
            csvListDropdown = csvListFileBrowser.GetComponent<TMP_Dropdown>();
        }

        if (csvListDropdown != null &&
            csvListDropdown.options != null &&
            csvListDropdown.options.Count > 0 &&
            IsPlaceholderOption(csvListDropdown.options[csvListDropdown.value].text))
        {
            Debug.LogWarning($"[StartInput] No compatible CSV files for robot '{filterKey}'.");
        }
    }

    private void ResolveCsvListReferences()
    {
        if (csvListFileBrowser != null && csvListDropdown != null)
        {
            return;
        }

        GameObject csvListObject = GameObject.Find(csvListObjectName);
        if (csvListObject == null)
        {
            if (logReferenceResolution) LogVerbose($"ResolveCsvListReferences: GameObject '{csvListObjectName}' not found.");
            return;
        }

        if (csvListFileBrowser == null)
        {
            csvListFileBrowser = csvListObject.GetComponent<FileBrowser>();
        }

        if (csvListDropdown == null)
        {
            csvListDropdown = csvListObject.GetComponent<TMP_Dropdown>();
        }

        if (logReferenceResolution)
        {
            LogVerbose($"ResolveCsvListReferences: object='{csvListObject.name}', hasFileBrowser={csvListFileBrowser != null}, hasDropdown={csvListDropdown != null}");
        }
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
            if (logReferenceResolution) LogVerbose($"ResolveRoboListReferences: GameObject '{roboListObjectName}' not found.");
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

        if (logReferenceResolution)
        {
            LogVerbose($"ResolveRoboListReferences: object='{roboListObject.name}', hasFileBrowser={roboListFileBrowser != null}, hasDropdown={roboListDropdown != null}");
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
        return lowered.Contains("can't find") ||
               lowered.Contains("cannot find") ||
               lowered.Contains("no compatible");
    }
}
