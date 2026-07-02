using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
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
    [Tooltip("Resolve relative OUTPUT_ROOT under the Unity project's Library folder so live CSV/log/pkl writes do not trigger AssetDatabase imports. Absolute outputRoot paths are unchanged.")]
    [SerializeField] private bool keepRuntimeOutputOutsideAssets = true;
    [SerializeField] private string outputCsvFileName = "csv/live_motion.csv";
    [SerializeField] private string defaultRobotName = "unitree_g1";
    [SerializeField] private bool recordGmrVideo = false;
    [SerializeField] private bool recordWhamVideo = false;
    [SerializeField] private bool track = true;  // Alias for --camera_follow
    [SerializeField] private bool enableTcpStreaming = true;
    [Tooltip("When TCP streaming is enabled, do not also open/write WHAM/GMR preview videos. TCP frames still render offscreen for Unity.")]
    [SerializeField] private bool disablePreviewVideoWhenTcpStreaming = true;
    [Tooltip("Input video path passed to run.ps1. Defaults to an examples video instead of webcam input.")]
    [SerializeField] private string videoPath = "examples/IMG_9732.mov";

    [Header("GMR Viewer Camera")]
    [SerializeField] private bool gmrCameraFollow = true;
    [SerializeField] private float gmrCameraLookatHeightOffset = 0.50f;
    [SerializeField] private float gmrCameraElevation = -28.0f;
    [SerializeField] private float gmrCameraDistanceScale = 1.70f;
    [SerializeField] private string gmrCameraAzimuth = "";
    [SerializeField] private int gmrTcpRenderWidth = 960;
    [SerializeField] private int gmrTcpRenderHeight = 720;
    [Tooltip("Limit GMR TCP preview rendering FPS. CSV generation still runs every frame.")]
    [SerializeField] private float gmrPreviewFps = 3.0f;

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
    [Tooltip("Maximum IK iterations per GMR frame. Lower values improve live FPS with a small accuracy tradeoff.")]
    [SerializeField] private int gmrMaxIter = 5;
    [Tooltip("Flush live_motion.csv every N rows. Higher values reduce disk stalls; Unity keeps a playback buffer for latency tolerance.")]
    [SerializeField] private int gmrCsvFlushInterval = 5;
    [Tooltip("Flush WHAM tail stream every N records when tail streaming is enabled.")]
    [SerializeField] private int whamTailFlushInterval = 10;
    [Tooltip("Pipeline log heartbeat interval in source frames.")]
    [SerializeField] private int pipelineHeartbeatFrames = 60;

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

    [Header("Realtime UI Controls")]
    [SerializeField] private Button replayButton;
    [SerializeField] private string replayButtonObjectName = "Replay";
    [SerializeField] private Button stopButton;
    [SerializeField] private string stopButtonObjectName = "Stop";

    [Tooltip("When the RoboList dropdown selection changes at runtime, stop the running " +
             "WHAM/GMR pipeline (the CSV format is robot-specific so it must be restarted) " +
             "and switch the live retargeting target to the newly selected robot. The user " +
             "then presses Start again to relaunch the pipeline with the new ROBOT env var.")]
    [SerializeField] private bool switchActiveRobotOnDropdownChange = true;

    [Tooltip("If true, also auto-press Start after the robot switch so the user doesn't have " +
             "to. Off by default because relaunching the Python pipeline is heavy and you " +
             "usually want to confirm before paying that cost.")]
    [SerializeField] private bool autoStartOnRobotSwitch = false;

    [Tooltip("When ON, only the selected robot remains visible/collidable. Inactive robots stay active " +
             "as GameObjects but their Renderers and Colliders are disabled to avoid articulation rebuilds.")]
    [SerializeField] private bool hideInactiveRobotsOnSwitch = true;
    [Tooltip("Add lightweight BoxColliders at runtime, but only for the currently selected robot.")]
    [SerializeField] private bool addRuntimeCollidersForSelectedRobot = true;
    [Tooltip("World-space X spacing between selected robots during multi-robot replay/live display.")]
    [SerializeField] private float multiRobotDisplaySpacingMeters = 1.8f;
    [SerializeField] private Vector3 fallbackRuntimeColliderSize = new Vector3(0.08f, 0.08f, 0.08f);
    [Tooltip("Runtime-added robot colliders should be trigger/query-only. This prevents ArticulationBody self-contact from destabilizing joints.")]
    [SerializeField] private bool runtimeRobotCollidersAreTriggers = true;

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
    [Tooltip("Keep Unity playback this many seconds behind the latest CSV row to avoid catching the writer and stuttering on tail frames.")]
    [SerializeField] private float realtimePlaybackBufferSeconds = 0.2f;
    [Tooltip("Fallback live CSV producer frame rate. StartInput estimates the real rate from appended rows when enabled below.")]
    [SerializeField] private float defaultRealtimeCsvFps = ReplayCsvUtility.SourceFps;
    [Tooltip("Estimate WHAM/GMR CSV production fps from newly appended rows and align Unity live playback to it.")]
    [SerializeField] private bool estimateRealtimeCsvProducerFps = true;
    [Tooltip("Sliding window in seconds for estimating realtime CSV producer fps. The first backlog batch is ignored.")]
    [SerializeField] private float realtimeCsvFpsWindowSeconds = 2f;
    [Tooltip("Maximum realtime CSV batches applied on the Unity main thread per frame.")]
    [SerializeField] private int maxRealtimeCsvBatchesPerFrame = 4;
    [Tooltip("Maximum queued pipeline log lines emitted to the Unity console per frame.")]
    [SerializeField] private int maxUnityLogMessagesPerFrame = 2;
    [Tooltip("Maximum pending Unity console log lines. Extra normal logs are dropped; warnings/errors are retained.")]
    [SerializeField] private int maxPendingUnityLogMessages = 200;

    [Header("Realtime CSV Safety")]
    [SerializeField] private bool enableRealtimeCsvSafetyGate = true;
    [Tooltip("Reject one live CSV row if root translation jumps more than this many meters from the previous accepted row.")]
    [SerializeField] private float realtimeMaxRootJumpMeters = 1.0f;
    [Tooltip("Reject one live CSV row if root rotation jumps more than this many degrees from the previous accepted row.")]
    [SerializeField] private float realtimeMaxRootRotationJumpDegrees = 120f;
    [Tooltip("Reject one live CSV row if any joint jumps more than this many radians from the previous accepted row.")]
    [SerializeField] private float realtimeMaxJointJumpRadians = 3.5f;
    [Tooltip("Reject one live CSV row if any root position component exceeds this absolute meter value.")]
    [SerializeField] private float realtimeMaxAbsRootPositionMeters = 50f;
    [Tooltip("Reject one live CSV row if any joint value exceeds this absolute radian value.")]
    [SerializeField] private float realtimeMaxAbsJointRadians = 20f;
    [Tooltip("Abort live CSV ingestion after this many consecutive rejected rows to avoid feeding corrupt tail data into articulation bodies.")]
    [SerializeField] private int realtimeMaxConsecutiveRejectedRows = 5;

    [Header("Debug")]
    [SerializeField] private bool verboseDebugLogging = false;
    [SerializeField] private bool logCsvChangeEvents = false;
    [Tooltip("If enabled together with logCsvChangeEvents, high-frequency CSV append summaries are emitted to the Unity Console.")]
    [SerializeField] private bool emitRealtimeCsvAppendLogsToConsole = false;
    [Tooltip("If enabled, every Python stdout/stderr line is emitted to the Unity Console. Normal pipeline output is always written to the debug log file.")]
    [SerializeField] private bool emitPipelineOutputToUnityConsole = false;
    [Tooltip("Minimum seconds between high-frequency CSV append logs. Startup/empty-file warnings are not throttled by this.")]
    [SerializeField] private float csvChangeLogIntervalSeconds = 1f;
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
    private float[] lastAcceptedRealtimeRow;
    private int consecutiveRejectedRealtimeRows;
    private bool realtimeSafetyWarningLogged;
    private readonly List<float[]> pendingRealtimeRows = new List<float[]>();
    private IRealtimeCsvMimicAgent activeRealtimeCsvAgent;
    private IMimicAgent activeRealtimeMimicAgent;
    private Thread csvWorkerThread;
    private readonly List<Thread> csvWorkerThreads = new List<Thread>();
    private volatile bool csvWorkerRunning;
    private readonly HashSet<string> selectedRobotKeys = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
    private bool hasExplicitRobotSelection;
    private readonly Dictionary<string, FileBrowser> robotCsvBrowsers = new Dictionary<string, FileBrowser>(System.StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, TMP_Dropdown> robotCsvDropdowns = new Dictionary<string, TMP_Dropdown>(System.StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<string, RealtimeRobotState> realtimeRobotStates = new Dictionary<string, RealtimeRobotState>(System.StringComparer.OrdinalIgnoreCase);
    private volatile bool isStoppingPipeline;
    private readonly object processStateLock = new object();
    private readonly ConcurrentQueue<RealtimeCsvBatch> realtimeCsvRowsQueue = new ConcurrentQueue<RealtimeCsvBatch>();
    private readonly ConcurrentQueue<string> realtimeCsvResetQueue = new ConcurrentQueue<string>();
    private readonly ConcurrentQueue<QueuedUnityLog> unityLogQueue = new ConcurrentQueue<QueuedUnityLog>();
    private readonly ConcurrentQueue<string> debugLogQueue = new ConcurrentQueue<string>();
    private readonly object debugLogWriterLock = new object();
    private readonly AutoResetEvent debugLogSignal = new AutoResetEvent(false);
    private Thread debugLogWriterThread;
    private volatile bool debugLogWriterRunning;
    private string resolvedDebugLogPath = string.Empty;
    public GameObject CurrentSelectedRobotRoot { get; private set; }
    public string CurrentSelectedRobotKey { get; private set; } = string.Empty;
    public bool IsPipelineStopping => isStoppingPipeline;
    public bool IsPipelineRunning
    {
        get
        {
            Process process = pythonProcess;
            try
            {
                return process != null && !process.HasExited;
            }
            catch
            {
                return false;
            }
        }
    }

    public bool IsRealtimeControlsLocked => PipelineStartupActive || (IsPipelineRunning && !isStoppingPipeline);
    public bool PipelineStartupActive { get; private set; }
    public float PipelineProgress01 { get; private set; }
    public string PipelineStatusText { get; private set; } = string.Empty;
    public string PipelineActivityText { get; private set; } = string.Empty;

    public static string NormalizeRobotKeyForUi(string rawRobotName)
    {
        return TryResolveRobotKeyQuiet(rawRobotName);
    }

    public IReadOnlyList<string> GetSelectedRobotKeys()
    {
        return BuildSelectedRobotKeyList();
    }

    public IReadOnlyList<GameObject> GetSelectedRobotRoots()
    {
        var roots = new List<GameObject>();
        List<string> keys = BuildSelectedRobotKeyList();
        for (int i = 0; i < keys.Count; i++)
        {
            GameObject root = ResolveRobotRootByKey(keys[i]);
            if (root != null && !roots.Contains(root))
            {
                roots.Add(root);
            }
        }

        if (roots.Count == 0 && !hasExplicitRobotSelection && CurrentSelectedRobotRoot != null)
        {
            roots.Add(CurrentSelectedRobotRoot);
        }

        return roots;
    }

    public bool IsRobotSelected(string robotKeyOrLabel)
    {
        string key = TryResolveRobotKeyQuiet(robotKeyOrLabel);
        if (string.IsNullOrWhiteSpace(key))
        {
            return false;
        }

        List<string> keys = BuildSelectedRobotKeyList();
        return keys.Contains(key);
    }

    public void SetRobotSelected(string robotKeyOrLabel, bool selected)
    {
        string key = TryResolveRobotKeyQuiet(robotKeyOrLabel);
        if (string.IsNullOrWhiteSpace(key))
        {
            Debug.LogWarning($"[StartInput] Ignoring unknown robot selection '{robotKeyOrLabel}'.");
            return;
        }

        if (selected)
        {
            hasExplicitRobotSelection = true;
            selectedRobotKeys.Add(key);
        }
        else
        {
            hasExplicitRobotSelection = true;
            selectedRobotKeys.Remove(key);
        }

        if (selectedRobotKeys.Count == 0)
        {
            Debug.LogWarning("[StartInput] No robots selected. Start/replay will be disabled until at least one robot is selected.");
        }

        RefreshRegisteredCsvBrowsers();
        ApplyRobotVisibility(string.Empty, key);
        SelectedRobotCameraFollow.NotifyRobotSelectionChanged(key, selected);
    }

    public void RegisterRobotCsvBrowser(string robotKeyOrLabel, FileBrowser browser, TMP_Dropdown dropdown)
    {
        string key = TryResolveRobotKeyQuiet(robotKeyOrLabel);
        if (string.IsNullOrWhiteSpace(key))
        {
            return;
        }

        if (browser != null)
        {
            robotCsvBrowsers[key] = browser;
            browser.SetCsvRobotFilter(key);
        }

        if (dropdown != null)
        {
            robotCsvDropdowns[key] = dropdown;
        }
    }

    public bool TryGetSelectedCsvForRobot(string robotKeyOrLabel, out string csvPath, out string csvName)
    {
        csvPath = string.Empty;
        csvName = string.Empty;
        string key = TryResolveRobotKeyQuiet(robotKeyOrLabel);
        if (string.IsNullOrWhiteSpace(key))
        {
            return false;
        }

        if (robotCsvBrowsers.TryGetValue(key, out FileBrowser browser) && browser != null)
        {
            csvPath = browser.GetSelectedCsvPath();
            csvName = browser.GetSelectedCsvName();
            if (!string.IsNullOrWhiteSpace(csvPath) && File.Exists(csvPath))
            {
                return true;
            }
        }

        if (string.Equals(key, TryResolveRobotKeyQuiet(ResolveSelectedRobotName()), System.StringComparison.OrdinalIgnoreCase))
        {
            ResolveCsvListReferences();
            if (csvListFileBrowser != null)
            {
                csvPath = csvListFileBrowser.GetSelectedCsvPath();
                csvName = csvListFileBrowser.GetSelectedCsvName();
                if (!string.IsNullOrWhiteSpace(csvPath) && File.Exists(csvPath))
                {
                    return true;
                }
            }
        }

        return false;
    }
    public bool HasPipelineError { get; private set; }
    public string LastPipelineError { get; private set; } = string.Empty;

    public void ClearPipelineError()
    {
        HasPipelineError = false;
        LastPipelineError = string.Empty;
    }

    public void SetGmrTcpRenderSizeFromHud(int width, int height)
    {
        gmrTcpRenderWidth = Mathf.Max(320, width);
        gmrTcpRenderHeight = Mathf.Max(240, height);
    }

    public bool TryGetRuntimeParameterValue(string parameterName, out object value)
    {
        value = null;
        FieldInfo field = FindRuntimeParameterField(parameterName);
        if (field == null)
        {
            return false;
        }

        value = field.GetValue(this);
        return true;
    }

    public bool TrySetRuntimeParameter(string parameterName, string rawValue, out string error)
    {
        error = string.Empty;
        FieldInfo field = FindRuntimeParameterField(parameterName);
        if (field == null)
        {
            error = $"Unknown StartInput parameter '{parameterName}'.";
            return false;
        }

        if (IsRealtimeControlsLocked)
        {
            error = $"Cannot edit '{parameterName}' while the retargeting pipeline is running.";
            return false;
        }

        if (!TryConvertRuntimeParameterValue(field.FieldType, rawValue, out object converted, out error))
        {
            error = $"Invalid value for '{parameterName}': {error}";
            return false;
        }

        field.SetValue(this, converted);
        return true;
    }

    public bool TrySetRuntimeParameter(string parameterName, bool value, out string error)
    {
        return TrySetRuntimeParameter(parameterName, value ? "true" : "false", out error);
    }

    private static FieldInfo FindRuntimeParameterField(string parameterName)
    {
        if (string.IsNullOrWhiteSpace(parameterName))
        {
            return null;
        }

        return typeof(StartInput).GetField(
            parameterName,
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
    }

    private static bool TryConvertRuntimeParameterValue(System.Type type, string rawValue, out object value, out string error)
    {
        value = null;
        error = string.Empty;
        string text = rawValue ?? string.Empty;

        if (type == typeof(string))
        {
            value = text;
            return true;
        }

        if (type == typeof(bool))
        {
            if (TryParseRuntimeBool(text, out bool boolValue))
            {
                value = boolValue;
                return true;
            }

            error = "expected true/false, 1/0, on/off, or yes/no.";
            return false;
        }

        if (type == typeof(int))
        {
            if (int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int intValue))
            {
                value = intValue;
                return true;
            }

            error = "expected an integer.";
            return false;
        }

        if (type == typeof(float))
        {
            if (float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float floatValue))
            {
                value = floatValue;
                return true;
            }

            error = "expected a floating point number.";
            return false;
        }

        error = $"unsupported field type {type.Name}.";
        return false;
    }

    private static bool TryParseRuntimeBool(string rawValue, out bool value)
    {
        value = false;
        string text = (rawValue ?? string.Empty).Trim();
        if (bool.TryParse(text, out value))
        {
            return true;
        }

        if (string.Equals(text, "1", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(text, "yes", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(text, "y", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(text, "on", System.StringComparison.OrdinalIgnoreCase))
        {
            value = true;
            return true;
        }

        if (string.Equals(text, "0", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(text, "no", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(text, "n", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(text, "off", System.StringComparison.OrdinalIgnoreCase))
        {
            value = false;
            return true;
        }

        return false;
    }

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

    private static readonly string[] MultiRobotDisplayOrder =
    {
        "unitree_g1",
        "unitree_h1",
        "x02lite",
        "openloong",
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

    private struct RealtimeCsvBatch
    {
        public readonly string RobotKey;
        public readonly List<float[]> Rows;
        public readonly float ProducerFps;

        public RealtimeCsvBatch(string robotKey, List<float[]> rows, float producerFps)
        {
            RobotKey = robotKey ?? string.Empty;
            Rows = rows;
            ProducerFps = producerFps;
        }
    }

    private sealed class RealtimeRobotState
    {
        public string RobotKey;
        public string CsvPath;
        public IMimicAgent MimicAgent;
        public IRealtimeCsvMimicAgent CsvAgent;
        public bool ReplayBootstrapped;
        public int SourceRowsRead;
        public float[] LastAcceptedRow;
        public int ConsecutiveRejectedRows;
        public bool SafetyWarningLogged;
        public readonly List<float[]> PendingRows = new List<float[]>();
    }

    private struct RealtimeFpsSample
    {
        public readonly System.DateTime TimeUtc;
        public readonly int RowCount;

        public RealtimeFpsSample(System.DateTime timeUtc, int rowCount)
        {
            TimeUtc = timeUtc;
            RowCount = rowCount;
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
        if (message == null)
        {
            message = string.Empty;
        }
        string line = "[StartInput/Debug] " + message;
        AppendDebugLogLine(line);

        bool highFrequencyCsvLine =
            message.StartsWith("CSV appended", System.StringComparison.OrdinalIgnoreCase) ||
            message.StartsWith("CSV exists but is empty", System.StringComparison.OrdinalIgnoreCase);
        if (highFrequencyCsvLine && !emitRealtimeCsvAppendLogsToConsole)
        {
            return;
        }

        Debug.Log(line);
    }

    private void QueueUnityLog(string message, QueuedUnityLogType type = QueuedUnityLogType.Log, bool appendDebug = true)
    {
        if (appendDebug)
        {
            AppendDebugLogLine(message);
        }

        UpdatePipelineStartupFromUnityLog(message, type);

        int pendingLimit = System.Math.Max(1, maxPendingUnityLogMessages);
        if (type == QueuedUnityLogType.Log && unityLogQueue.Count >= pendingLimit)
        {
            return;
        }

        unityLogQueue.Enqueue(new QueuedUnityLog(message, type));
    }

    private void HandlePipelineLogLine(string data, bool isErrorStream)
    {
        if (string.IsNullOrWhiteSpace(data))
        {
            return;
        }

        QueuedUnityLogType logType = ClassifyPipelineLogLine(data);
        string prefix = logType == QueuedUnityLogType.Error
            ? "[Pipeline-ERR]"
            : logType == QueuedUnityLogType.Warning
                ? "[Pipeline-WARN]"
                : "[Pipeline]";
        string line = $"{prefix} {data}";

        AppendDebugLogLine(isErrorStream && logType == QueuedUnityLogType.Log
            ? $"[Pipeline/stderr] {data}"
            : line);
        UpdatePipelineStartupFromPipelineLog(data, logType);

        if (!emitPipelineOutputToUnityConsole && !ShouldPromotePipelineLogLine(data))
        {
            return;
        }

        QueueUnityLog(line, logType, appendDebug: false);
    }

    private static QueuedUnityLogType ClassifyPipelineLogLine(string data)
    {
        if (string.IsNullOrWhiteSpace(data))
        {
            return QueuedUnityLogType.Log;
        }

        string lower = data.ToLowerInvariant();

        // Loguru writes INFO to stderr by default. Respect the explicit log
        // level in the message instead of treating stderr as a warning.
        if (ContainsLoguruLevel(lower, "debug") || ContainsLoguruLevel(lower, "info"))
        {
            return QueuedUnityLogType.Log;
        }

        if (ContainsLoguruLevel(lower, "warning") || ContainsLoguruLevel(lower, "warn"))
        {
            return QueuedUnityLogType.Warning;
        }

        if (ContainsLoguruLevel(lower, "error") ||
            ContainsLoguruLevel(lower, "critical") ||
            lower.Contains("traceback") ||
            lower.Contains("unhandled exception") ||
            lower.Contains("fatal") ||
            lower.Contains("aborted") ||
            lower.Contains("failed"))
        {
            return QueuedUnityLogType.Error;
        }

        if (lower.Contains("warning") || lower.Contains("warn"))
        {
            return QueuedUnityLogType.Warning;
        }

        if (lower.Contains("error") || lower.Contains("exception"))
        {
            return QueuedUnityLogType.Error;
        }

        return QueuedUnityLogType.Log;
    }

    private static bool ContainsLoguruLevel(string lower, string level)
    {
        return lower.Contains("| " + level + " ");
    }

    private static bool ShouldPromotePipelineLogLine(string data)
    {
        if (string.IsNullOrWhiteSpace(data))
        {
            return false;
        }

        string lower = data.ToLowerInvariant();
        return lower.Contains("traceback") ||
               lower.Contains("exception") ||
               lower.Contains("error") ||
               lower.Contains("fatal") ||
               lower.Contains("failed") ||
               lower.Contains("aborted") ||
               lower.Contains("loading models") ||
               lower.Contains("first frame read") ||
               lower.Contains("pipeline started") ||
               lower.Contains("csv output opened") ||
               lower.Contains("initialized robot=") ||
               ContainsFirstCsvRowMarker(lower) ||
               lower.Contains("connected to unity") ||
               lower.Contains("sent first");
    }

    private static bool ContainsFirstCsvRowMarker(string lower)
    {
        const string marker = "wrote csv row=1";
        int index = lower.IndexOf(marker, System.StringComparison.Ordinal);
        if (index < 0)
        {
            return false;
        }

        int nextIndex = index + marker.Length;
        return nextIndex >= lower.Length || !char.IsDigit(lower[nextIndex]);
    }

    private void BeginPipelineStartupStatus(string activity)
    {
        ClearPipelineError();
        PipelineStartupActive = true;
        PipelineStatusText = "Preparing retargeting";
        PipelineActivityText = activity;
        PipelineProgress01 = 0.03f;
    }

    private void SetPipelineStartupError(string message)
    {
        HasPipelineError = true;
        LastPipelineError = string.IsNullOrWhiteSpace(message) ? "Retargeting pipeline failed." : message;
        PipelineStatusText = "Retargeting failed";
        PipelineActivityText = LastPipelineError;
        PipelineStartupActive = false;
        PipelineProgress01 = 0f;
    }

    private void FinishPipelineStartup(string activity)
    {
        PipelineStatusText = "Retargeting running";
        PipelineActivityText = string.IsNullOrWhiteSpace(activity) ? "Realtime playback started." : activity;
        PipelineProgress01 = 1f;
        PipelineStartupActive = false;
    }

    private void AdvancePipelineStartup(string status, string activity, float minimumProgress, float logStep = 0.012f)
    {
        if (!PipelineStartupActive)
        {
            return;
        }

        if (!string.IsNullOrWhiteSpace(status))
        {
            PipelineStatusText = status;
        }

        if (!string.IsNullOrWhiteSpace(activity))
        {
            PipelineActivityText = activity;
        }

        float stepped = Mathf.Min(0.94f, PipelineProgress01 + Mathf.Max(0f, logStep));
        PipelineProgress01 = Mathf.Clamp01(Mathf.Max(PipelineProgress01, minimumProgress, stepped));
    }

    private void UpdatePipelineStartupFromPipelineLog(string data, QueuedUnityLogType logType)
    {
        if (string.IsNullOrWhiteSpace(data))
        {
            return;
        }

        string activity = CompactPipelineLogForHud(data);
        if (logType == QueuedUnityLogType.Error)
        {
            SetPipelineStartupError(activity);
            return;
        }

        string lower = data.ToLowerInvariant();
        if (lower.Contains("loading models"))
        {
            AdvancePipelineStartup("Loading WHAM/GMR models", activity, 0.18f, 0.02f);
        }
        else if (lower.Contains("first frame read"))
        {
            AdvancePipelineStartup("Reading source video", activity, 0.32f, 0.02f);
        }
        else if (lower.Contains("pipeline started"))
        {
            AdvancePipelineStartup("Pipeline process started", activity, 0.42f, 0.02f);
        }
        else if (lower.Contains("csv output opened"))
        {
            AdvancePipelineStartup("Opening live CSV output", activity, 0.54f, 0.02f);
        }
        else if (lower.Contains("initialized robot="))
        {
            AdvancePipelineStartup("Initializing robot retargeter", activity, 0.66f, 0.02f);
        }
        else if (ContainsFirstCsvRowMarker(lower))
        {
            AdvancePipelineStartup("Writing first retargeted frame", activity, 0.78f, 0.02f);
        }
        else if (lower.Contains("connected to unity") || lower.Contains("sent first"))
        {
            AdvancePipelineStartup("Streaming preview frames", activity, 0.72f, 0.015f);
        }
        else
        {
            AdvancePipelineStartup(PipelineStatusText, activity, PipelineProgress01, 0.006f);
        }
    }

    private void UpdatePipelineStartupFromUnityLog(string message, QueuedUnityLogType type)
    {
        if (string.IsNullOrWhiteSpace(message))
        {
            return;
        }

        if (type == QueuedUnityLogType.Error)
        {
            SetPipelineStartupError(message);
            return;
        }

        string lower = message.ToLowerInvariant();
        if (lower.Contains("csv file does not exist yet"))
        {
            AdvancePipelineStartup("Waiting for live CSV", message, 0.44f, 0.01f);
        }
        else if (lower.Contains("live_motion.csv is still empty"))
        {
            AdvancePipelineStartup("Waiting for GMR CSV rows", message, 0.58f, 0.01f);
        }
        else if (lower.Contains("realtime csv has data"))
        {
            AdvancePipelineStartup("Live CSV rows received", message, 0.84f, 0.02f);
        }
        else if (lower.Contains("realtime csv worker active"))
        {
            AdvancePipelineStartup("Starting CSV monitor", message, 0.38f, 0.02f);
        }
        else if (lower.Contains("csv appended"))
        {
            AdvancePipelineStartup("Receiving retargeted CSV rows", message, 0.88f, 0.012f);
        }
    }

    private static string CompactPipelineLogForHud(string data)
    {
        string text = data.Trim();
        int messageIndex = text.IndexOf(" - ", System.StringComparison.Ordinal);
        if (messageIndex >= 0 && messageIndex + 3 < text.Length)
        {
            text = text.Substring(messageIndex + 3).Trim();
        }

        return text.Length <= 180 ? text : text.Substring(0, 177) + "...";
    }

    private void FlushUnityLogQueue()
    {
        int maxLogs = Mathf.Clamp(maxUnityLogMessagesPerFrame, 1, 8);
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
        sb.AppendLine($"  keepRuntimeOutputOutsideAssets: {keepRuntimeOutputOutsideAssets}");
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
        sb.AppendLine($"  realtimePlaybackBufferSeconds: {realtimePlaybackBufferSeconds}");
        sb.AppendLine($"  csvChangeLogIntervalSeconds: {csvChangeLogIntervalSeconds}");
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
        ResetTransientRuntimeStateForFreshPlay();
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

    public void ResetTransientRuntimeStateForFreshPlay()
    {
        selectedRobotKeys.Clear();
        hasExplicitRobotSelection = false;
        realtimeRobotStates.Clear();
        pendingRealtimeRows.Clear();
        DrainRealtimeCsvQueues();
        replayBootstrapped = false;
        realtimeSourceRowsRead = 0;
        CurrentSelectedRobotRoot = null;
        CurrentSelectedRobotKey = string.Empty;
        ResetRealtimeCsvSafetyGate();
        ResetRobotDisplayOffsets();
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

    void LateUpdate()
    {
        if (IsRealtimeControlsLocked)
        {
            SetRuntimeColliderProxiesEnabledForCurrentState();
            return;
        }

        SyncRuntimeColliderProxies();
        SetRuntimeColliderProxiesEnabledForCurrentState();
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
        for (int i = 0; i < maxBatches && realtimeCsvRowsQueue.TryDequeue(out RealtimeCsvBatch batch); i++)
        {
            ApplyRealtimeCsvRows(batch.RobotKey, batch.Rows, batch.ProducerFps);
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

        if (runtimeColliderProxyRoot != null)
        {
            Destroy(runtimeColliderProxyRoot.gameObject);
            runtimeColliderProxyRoot = null;
        }

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
                if (string.Equals(selected.RobotKey, "x02lite", System.StringComparison.OrdinalIgnoreCase))
                {
                    selected.ReplayMode = false;
                }
                else
                {
                    selected.ReplayMode = true;

                    try { selected.ResetToInitialState(); }
                    catch (System.Exception e)
                    {
                        Debug.LogWarning($"[StartInput] {selected.RobotKey} ResetToInitialState threw: {e.Message}");
                    }

                    selected.RequestEndEpisode();
                }
                if (string.Equals(selected.RobotKey, "x02lite", System.StringComparison.OrdinalIgnoreCase))
                {
                    Debug.Log($"[StartInput] Selected '{selected.RobotKey}': keep grounded neutral pose and skip default replay.");
                }
                else
                {
                    Debug.Log($"[StartInput] Selected '{selected.RobotKey}': articulation reset queued via OnEpisodeBegin.");
                }
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
    private readonly Dictionary<GameObject, Collider[]> _cachedRobotColliders =
        new Dictionary<GameObject, Collider[]>();
    private readonly Dictionary<Collider, bool> _originalColliderEnabled =
        new Dictionary<Collider, bool>();
    private readonly Dictionary<GameObject, Collider[]> _generatedRobotColliders =
        new Dictionary<GameObject, Collider[]>();
    private readonly HashSet<Collider> _runtimeRobotColliders =
        new HashSet<Collider>();
    private readonly Dictionary<GameObject, RuntimeRobotColliderProxy[]> _runtimeColliderProxies =
        new Dictionary<GameObject, RuntimeRobotColliderProxy[]>();
    private Transform runtimeColliderProxyRoot;

    private sealed class RuntimeRobotColliderProxy
    {
        public Transform Source;
        public Transform Proxy;
        public Collider Collider;
    }

    private Renderer[] GetOrCacheRenderers(GameObject root)
    {
        if (root == null) return System.Array.Empty<Renderer>();
        if (_cachedRobotRenderers.TryGetValue(root, out var cached) && cached != null) return cached;
        var fresh = root.GetComponentsInChildren<Renderer>(includeInactive: true);
        _cachedRobotRenderers[root] = fresh;
        return fresh;
    }

    private Collider[] GetOrCacheColliders(GameObject root)
    {
        if (root == null) return System.Array.Empty<Collider>();
        if (_cachedRobotColliders.TryGetValue(root, out var cached) && cached != null) return cached;

        var fresh = root.GetComponentsInChildren<Collider>(includeInactive: true);
        if (_generatedRobotColliders.TryGetValue(root, out Collider[] generated) &&
            generated != null &&
            generated.Length > 0)
        {
            var combined = new Collider[fresh.Length + generated.Length];
            System.Array.Copy(fresh, combined, fresh.Length);
            System.Array.Copy(generated, 0, combined, fresh.Length, generated.Length);
            fresh = combined;
        }

        for (int i = 0; i < fresh.Length; i++)
        {
            Collider collider = fresh[i];
            if (collider != null && !_originalColliderEnabled.ContainsKey(collider))
            {
                _originalColliderEnabled[collider] = collider.enabled;
            }
        }

        _cachedRobotColliders[root] = fresh;
        return fresh;
    }

    private static bool EnsureRootActiveForVisibility(GameObject root)
    {
        if (root == null) return false;

        bool changed = false;
        Transform current = root.transform;
        while (current != null)
        {
            GameObject go = current.gameObject;
            if (!go.activeSelf)
            {
                go.SetActive(true);
                changed = true;
            }

            current = current.parent;
        }

        return changed;
    }

    private static int EnsureRendererHierarchyActive(GameObject root, Renderer[] renderers)
    {
        if (root == null || renderers == null) return 0;

        int changed = 0;
        Transform rootTransform = root.transform;
        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer renderer = renderers[i];
            if (renderer == null) continue;

            Transform current = renderer.transform;
            while (current != null)
            {
                GameObject go = current.gameObject;
                if (!go.activeSelf)
                {
                    go.SetActive(true);
                    changed++;
                }

                if (current == rootTransform)
                {
                    break;
                }

                current = current.parent;
            }
        }

        return changed;
    }

    private bool GetOriginalColliderEnabled(Collider collider)
    {
        return collider != null &&
               (!_originalColliderEnabled.TryGetValue(collider, out bool originalEnabled) || originalEnabled);
    }

    private void EnsureRuntimeCollidersForSelectedRobot(GameObject root)
    {
        if (!addRuntimeCollidersForSelectedRobot || root == null) return;
        if (_generatedRobotColliders.ContainsKey(root)) return;

        ArticulationBody[] bodies = root.GetComponentsInChildren<ArticulationBody>(includeInactive: true);
        var generated = new List<Collider>();
        var proxies = new List<RuntimeRobotColliderProxy>();

        for (int i = 0; i < bodies.Length; i++)
        {
            ArticulationBody body = bodies[i];
            if (body == null || BodyHasUsableLocalCollider(body))
            {
                continue;
            }

            BoxCollider box = CreateRuntimeColliderProxy(root, body, out RuntimeRobotColliderProxy proxy);
            if (box == null || proxy == null)
            {
                continue;
            }

            if (TryComputeLocalRendererBounds(body, out Bounds bounds))
            {
                box.center = bounds.center;
                box.size = ClampRuntimeColliderSize(bounds.size);
            }
            else
            {
                box.center = Vector3.zero;
                box.size = ClampRuntimeColliderSize(fallbackRuntimeColliderSize);
            }

            ConfigureRuntimeRobotCollider(box);
            box.enabled = false;
            _originalColliderEnabled[box] = true;
            generated.Add(box);
            proxies.Add(proxy);
        }

        _generatedRobotColliders[root] = generated.ToArray();
        _runtimeColliderProxies[root] = proxies.ToArray();
        if (generated.Count > 0)
        {
            _cachedRobotColliders.Remove(root);
            SyncRuntimeColliderProxies(root);
            Debug.Log($"[StartInput] Added {generated.Count} runtime trigger proxy BoxCollider(s) for selected robot '{root.name}'.");
        }
    }

    private BoxCollider CreateRuntimeColliderProxy(
        GameObject robotRoot,
        ArticulationBody sourceBody,
        out RuntimeRobotColliderProxy proxy)
    {
        proxy = null;
        if (robotRoot == null || sourceBody == null) return null;

        Transform proxyRoot = GetRuntimeColliderProxyRoot();
        if (proxyRoot == null) return null;

        string safeRobotName = robotRoot.name.Replace('/', '_');
        string safeBodyName = sourceBody.name.Replace('/', '_');
        var proxyObject = new GameObject($"{safeRobotName}_{safeBodyName}_QueryCollider");
        proxyObject.hideFlags = HideFlags.DontSave;
        proxyObject.transform.SetParent(proxyRoot, worldPositionStays: false);
        CopyWorldTransform(sourceBody.transform, proxyObject.transform);

        BoxCollider box = proxyObject.AddComponent<BoxCollider>();
        proxy = new RuntimeRobotColliderProxy
        {
            Source = sourceBody.transform,
            Proxy = proxyObject.transform,
            Collider = box,
        };
        return box;
    }

    private Transform GetRuntimeColliderProxyRoot()
    {
        if (runtimeColliderProxyRoot != null) return runtimeColliderProxyRoot;

        var root = new GameObject("__RuntimeRobotQueryColliders");
        root.hideFlags = HideFlags.DontSave;
        root.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        root.transform.localScale = Vector3.one;
        runtimeColliderProxyRoot = root.transform;
        return runtimeColliderProxyRoot;
    }

    private void ConfigureRuntimeRobotCollider(Collider collider)
    {
        if (collider == null) return;
        collider.isTrigger = runtimeRobotCollidersAreTriggers;
        _runtimeRobotColliders.Add(collider);
    }

    private void SyncRuntimeColliderProxies()
    {
        if (_runtimeColliderProxies.Count == 0) return;

        foreach (RuntimeRobotColliderProxy[] proxies in _runtimeColliderProxies.Values)
        {
            SyncRuntimeColliderProxies(proxies);
        }
    }

    private void SyncRuntimeColliderProxies(GameObject root)
    {
        if (root == null) return;
        if (_runtimeColliderProxies.TryGetValue(root, out RuntimeRobotColliderProxy[] proxies))
        {
            SyncRuntimeColliderProxies(proxies);
        }
    }

    private void SetRuntimeColliderProxiesEnabledForCurrentState()
    {
        bool allowForSelected = !IsRealtimeControlsLocked;
        foreach (var kv in _runtimeColliderProxies)
        {
            bool enableForRoot = allowForSelected && kv.Key != null && kv.Key == CurrentSelectedRobotRoot;
            RuntimeRobotColliderProxy[] proxies = kv.Value;
            if (proxies == null) continue;

            for (int i = 0; i < proxies.Length; i++)
            {
                Collider collider = proxies[i]?.Collider;
                if (collider == null) continue;
                collider.enabled = enableForRoot && GetOriginalColliderEnabled(collider);
            }
        }
    }

    private static void SyncRuntimeColliderProxies(RuntimeRobotColliderProxy[] proxies)
    {
        if (proxies == null) return;

        for (int i = 0; i < proxies.Length; i++)
        {
            RuntimeRobotColliderProxy proxy = proxies[i];
            if (proxy == null || proxy.Source == null || proxy.Proxy == null)
            {
                continue;
            }

            CopyWorldTransform(proxy.Source, proxy.Proxy);
        }
    }

    private static void CopyWorldTransform(Transform source, Transform target)
    {
        if (source == null || target == null) return;
        target.SetPositionAndRotation(source.position, source.rotation);
        Vector3 scale = source.lossyScale;
        target.localScale = new Vector3(
            Mathf.Max(Mathf.Abs(scale.x), 0.0001f),
            Mathf.Max(Mathf.Abs(scale.y), 0.0001f),
            Mathf.Max(Mathf.Abs(scale.z), 0.0001f));
    }

    private bool BodyHasUsableLocalCollider(ArticulationBody body)
    {
        if (body == null) return true;
        Collider[] colliders = body.GetComponents<Collider>();
        if (colliders == null || colliders.Length == 0) return false;

        for (int i = 0; i < colliders.Length; i++)
        {
            Collider collider = colliders[i];
            if (collider == null) continue;
            if (collider.enabled || GetOriginalColliderEnabled(collider))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryComputeLocalRendererBounds(ArticulationBody body, out Bounds localBounds)
    {
        localBounds = new Bounds(Vector3.zero, ClampRuntimeColliderSize(fallbackRuntimeColliderSize));
        if (body == null) return false;

        Renderer[] renderers = body.GetComponentsInChildren<Renderer>(includeInactive: true);
        bool hasBounds = false;
        Bounds result = default(Bounds);

        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer renderer = renderers[i];
            if (renderer == null || FindNearestArticulationBody(renderer.transform) != body)
            {
                continue;
            }

            EncapsulateWorldBoundsInLocal(body.transform, renderer.bounds, ref result, ref hasBounds);
        }

        if (!hasBounds) return false;
        localBounds = result;
        return true;
    }

    private static ArticulationBody FindNearestArticulationBody(Transform transform)
    {
        Transform current = transform;
        while (current != null)
        {
            ArticulationBody body = current.GetComponent<ArticulationBody>();
            if (body != null) return body;
            current = current.parent;
        }

        return null;
    }

    private static void EncapsulateWorldBoundsInLocal(
        Transform localFrame,
        Bounds worldBounds,
        ref Bounds localBounds,
        ref bool hasBounds)
    {
        Vector3 center = worldBounds.center;
        Vector3 extents = worldBounds.extents;

        for (int x = -1; x <= 1; x += 2)
        for (int y = -1; y <= 1; y += 2)
        for (int z = -1; z <= 1; z += 2)
        {
            Vector3 worldCorner = center + Vector3.Scale(extents, new Vector3(x, y, z));
            Vector3 localCorner = localFrame.InverseTransformPoint(worldCorner);

            if (!hasBounds)
            {
                localBounds = new Bounds(localCorner, Vector3.zero);
                hasBounds = true;
            }
            else
            {
                localBounds.Encapsulate(localCorner);
            }
        }
    }

    private static Vector3 ClampRuntimeColliderSize(Vector3 size)
    {
        const float minSize = 0.02f;
        const float maxSize = 2.0f;
        return new Vector3(
            Mathf.Clamp(Mathf.Abs(size.x), minSize, maxSize),
            Mathf.Clamp(Mathf.Abs(size.y), minSize, maxSize),
            Mathf.Clamp(Mathf.Abs(size.z), minSize, maxSize));
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
        List<string> selectedKeys = BuildSelectedRobotKeyList();
        HashSet<string> activeKeys = new HashSet<string>(selectedKeys, System.StringComparer.OrdinalIgnoreCase);
        CurrentSelectedRobotKey = selectedKeys.Count > 0 ? selectedKeys[0] : string.Empty;

        if (!hideInactiveRobotsOnSwitch)
        {
            CurrentSelectedRobotRoot = ResolveRobotRootByKey(CurrentSelectedRobotKey);
            return;
        }

        bool useLegacySingleSelection = !hasExplicitRobotSelection;

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

                bool match = false;
                if (useLegacySingleSelection)
                {
                    match =
                        !string.IsNullOrEmpty(entryLabel) && (
                            string.Equals(entryLabel, selectedLabel, System.StringComparison.OrdinalIgnoreCase) ||
                            string.Equals(entryLabel, selectedRobotKey, System.StringComparison.OrdinalIgnoreCase));
                }

                // Or match by the resolved key going through the alias table
                // (so "G1" label still matches a "unitree_g1" entry).
                if (!match && !string.IsNullOrEmpty(entryLabel))
                {
                    string entryKey = TryResolveRobotKeyQuiet(entryLabel);
                    if (!string.IsNullOrEmpty(entryKey) && activeKeys.Contains(entryKey))
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
                bool match = activeKeys.Contains(agent.RobotKey);
                if (!roster.ContainsKey(go) || match)
                    roster[go] = match;
            }
        }

        AddFallbackSceneRobotRoster(roster, selectedLabel, CurrentSelectedRobotKey);

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
        bool allowRuntimeColliderProxies = !IsRealtimeControlsLocked;
        GameObject selectedRoot = null;
        foreach (var kv in roster)
        {
            GameObject root = kv.Key;
            bool isSelected = kv.Value;
            if (root == null) continue;
            if (isSelected)
            {
                selectedRoot = root;
            }

            // One-time SetActive(true) bootstrap for never-shown robots and
            // inactive visual children. We still never SetActive(false) here:
            // hidden robots are renderer-disabled only, so articulation state
            // is not rebuilt on every visibility change.
            if (isSelected)
            {
                bool rootActivated = EnsureRootActiveForVisibility(root);
                Renderer[] freshRenderers = root.GetComponentsInChildren<Renderer>(includeInactive: true);
                int visualNodesActivated = EnsureRendererHierarchyActive(root, freshRenderers);
                if (rootActivated || visualNodesActivated > 0)
                {
                    bootstrapped++;
                }

                // Selection is the only path that can resurrect a previously
                // inactive hierarchy, so rebuild the cached component arrays
                // before enabling renderers/colliders below.
                _cachedRobotRenderers.Remove(root);
                _cachedRobotColliders.Remove(root);
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

            if (isSelected && allowRuntimeColliderProxies)
            {
                EnsureRuntimeCollidersForSelectedRobot(root);
                SyncRuntimeColliderProxies(root);
            }

            Collider[] colliders = GetOrCacheColliders(root);
            for (int i = 0; i < colliders.Length; i++)
            {
                Collider collider = colliders[i];
                if (collider != null)
                {
                    if (_runtimeRobotColliders.Contains(collider))
                    {
                        ConfigureRuntimeRobotCollider(collider);
                        collider.enabled = isSelected && allowRuntimeColliderProxies && GetOriginalColliderEnabled(collider);
                        continue;
                    }
                    collider.enabled = isSelected && GetOriginalColliderEnabled(collider);
                }
            }

            if (agentsByGo.TryGetValue(root, out IMimicAgent visibilityAgent) &&
                visibilityAgent is ISelectableMimicAgent selectableAgent)
            {
                selectableAgent.SetRobotSelectedInScene(isSelected);
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
            Debug.Log($"[StartInput] First-time activated {bootstrapped} robot(s) with SetActive(true) bootstrap.");
        }
        CurrentSelectedRobotRoot = ResolveFirstSelectedRobotRoot(selectedKeys) ?? selectedRoot;
        Debug.Log($"[StartInput] Visibility switch (Renderer+Collider mode): shown={shown}, hidden={hidden}, selectedKeys='{string.Join(",", selectedKeys)}'.");
    }

    private GameObject ResolveFirstSelectedRobotRoot(List<string> selectedKeys)
    {
        if (selectedKeys == null)
        {
            return null;
        }

        for (int i = 0; i < selectedKeys.Count; i++)
        {
            GameObject root = ResolveRobotRootByKey(selectedKeys[i]);
            if (root != null)
            {
                return root;
            }
        }

        return null;
    }

    private void ApplySelectedRobotDisplayOffsets(HashSet<string> activeKeys)
    {
        if (activeKeys == null)
        {
            activeKeys = BuildSelectedRobotKeySet();
        }

        List<string> selectedKeys = BuildSelectedRobotKeyList();
        float spacing = Mathf.Max(0f, multiRobotDisplaySpacingMeters);
        float centerIndex = (selectedKeys.Count - 1) * 0.5f;

        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent is IReplayRootOffsetMimicAgent offsetAgent)
                {
                    offsetAgent.SetReplayRootOffset(Vector3.zero);
                }
            }
        }

        for (int i = 0; i < selectedKeys.Count; i++)
        {
            string key = selectedKeys[i];
            IMimicAgent agent = ResolveAgentByRobotKey(key);
            if (agent is IReplayRootOffsetMimicAgent offsetAgent)
            {
                offsetAgent.SetReplayRootOffset(new Vector3((i - centerIndex) * spacing, 0f, 0f));
            }
        }
    }

    private void ResetRobotDisplayOffsets()
    {
        if (MimicAgentRegistry.Instance == null)
        {
            return;
        }

        foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
        {
            if (agent is IReplayRootOffsetMimicAgent offsetAgent)
            {
                offsetAgent.SetReplayRootOffset(Vector3.zero);
            }
        }
    }

    private GameObject ResolveRobotRootByKey(string robotKeyOrLabel)
    {
        string key = TryResolveRobotKeyQuiet(robotKeyOrLabel);
        return ResolveSelectedRobotRootForCamera(key, key);
    }

    private GameObject ResolveSelectedRobotRootForCamera(string selectedLabel, string selectedRobotKey)
    {
        string key = !string.IsNullOrWhiteSpace(selectedRobotKey)
            ? selectedRobotKey.Trim()
            : TryResolveRobotKeyQuiet(selectedLabel);

        if (sceneRobots != null)
        {
            foreach (RobotSceneEntry entry in sceneRobots)
            {
                if (entry == null || entry.robotRoot == null)
                {
                    continue;
                }

                string entryLabel = (entry.label ?? string.Empty).Trim();
                string entryKey = TryResolveRobotKeyQuiet(entryLabel);
                if ((!string.IsNullOrWhiteSpace(entryLabel) &&
                     string.Equals(entryLabel, selectedLabel, System.StringComparison.OrdinalIgnoreCase)) ||
                    (!string.IsNullOrWhiteSpace(entryLabel) &&
                     string.Equals(entryLabel, key, System.StringComparison.OrdinalIgnoreCase)) ||
                    (!string.IsNullOrWhiteSpace(entryKey) &&
                     string.Equals(entryKey, key, System.StringComparison.OrdinalIgnoreCase)))
                {
                    return entry.robotRoot;
                }
            }
        }

        if (!string.IsNullOrWhiteSpace(key) && MimicAgentRegistry.Instance != null)
        {
            IMimicAgent agent = MimicAgentRegistry.Instance.FindByKey(key);
            if (agent?.AgentGameObject != null)
            {
                return agent.AgentGameObject;
            }
        }

        switch ((key ?? string.Empty).Trim().ToLowerInvariant())
        {
            case "unitree_g1":
                return FindFallbackRobotRoot("G1", "unitree_g1", "g1_29dof_rev_1_0");
            case "unitree_h1":
                return FindFallbackRobotRoot("H1", "h1", "unitree_h1");
            case "x02lite":
                return FindFallbackRobotRoot("X02Lite", "x02lite", "X02");
            case "openloong":
                return FindFallbackRobotRoot("OpenLoong", "openloong");
            default:
                return FindFallbackRobotRoot(selectedLabel, key);
        }
    }

    private void AddFallbackSceneRobotRoster(Dictionary<GameObject, bool> roster, string selectedLabel, string selectedRobotKey)
    {
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "unitree_g1", "G1", "unitree_g1", "g1_29dof_rev_1_0");
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "unitree_h1", "H1", "h1", "unitree_h1");
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "x02lite", "X02Lite", "x02lite", "X02");
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "openloong", "OpenLoong", "openloong");
    }

    private void AddFallbackRobotRoot(Dictionary<GameObject, bool> roster, string selectedLabel, string selectedRobotKey, string robotKey, params string[] sceneNames)
    {
        GameObject root = FindFallbackRobotRoot(sceneNames);
        if (root == null)
        {
            return;
        }

        HashSet<string> activeKeys = BuildSelectedRobotKeySet();
        bool useLegacySingleSelection = !hasExplicitRobotSelection;
        bool match = activeKeys.Contains(robotKey) ||
                     (useLegacySingleSelection && string.Equals(selectedRobotKey, robotKey, System.StringComparison.OrdinalIgnoreCase));
        if (!match && useLegacySingleSelection && !string.IsNullOrWhiteSpace(selectedLabel))
        {
            string labelKey = TryResolveRobotKeyQuiet(selectedLabel);
            match = string.Equals(labelKey, robotKey, System.StringComparison.OrdinalIgnoreCase);
        }

        if (!roster.ContainsKey(root) || match)
        {
            roster[root] = match;
        }
    }

    private static GameObject FindFallbackRobotRoot(params string[] sceneNames)
    {
        for (int pass = 0; pass < 2; pass++)
        {
            bool requireActive = pass == 0;
            foreach (string name in sceneNames)
            {
                if (string.IsNullOrWhiteSpace(name))
                {
                    continue;
                }

                foreach (Transform transform in FindObjectsOfType<Transform>(true))
                {
                    if (transform == null ||
                        !string.Equals(transform.name, name, System.StringComparison.OrdinalIgnoreCase) ||
                        (requireActive && !transform.gameObject.activeInHierarchy))
                    {
                        continue;
                    }

                    if (transform.GetComponentInChildren<ArticulationBody>(true) != null ||
                        transform.GetComponentInChildren<Renderer>(true) != null)
                    {
                        return transform.gameObject;
                    }
                }
            }
        }

        return null;
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

        if (!hasExplicitRobotSelection && !string.IsNullOrWhiteSpace(selectedKey))
        {
            selectedRobotKeys.Clear();
            selectedRobotKeys.Add(selectedKey);
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
        Debug.Log("[StartInput] Start button clicked.");
        AppendDebugLogLine("[StartInput] Start button clicked.");
        BeginPipelineStartupStatus($"Start clicked. robot='{selectedKey}', validating launch settings...");
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
            SetPipelineStartupError(!launchOk
                ? "Failed to launch WHAM/GMR process. Check run.ps1 path and pipeline logs."
                : "Failed to start live CSV monitor.");
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
        PipelineStartupActive = false;
        PipelineStatusText = "Retargeting stopped";
        PipelineActivityText = "Stop requested.";
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

    private void TryClearRealtimeCsvFiles(IReadOnlyList<string> robotKeys, string caller)
    {
        if (robotKeys == null || robotKeys.Count == 0)
        {
            TryClearCsv(caller);
            return;
        }

        foreach (string robotKey in robotKeys)
        {
            TryClearCsvPath(ResolveRobotRealtimeCsvPath(robotKey), caller);
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

    private string ResolveRealtimeCsvRootPath()
    {
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            string baseDir = ResolveWorkingDirectoryPath();
            resolvedOutputRootPath = ResolveOutputRootAbsolutePath(baseDir);
        }

        return string.IsNullOrWhiteSpace(resolvedOutputRootPath)
            ? string.Empty
            : Path.Combine(resolvedOutputRootPath, "csv");
    }

    private string ResolveRobotRealtimeCsvPath(string robotKey)
    {
        string csvRoot = ResolveRealtimeCsvRootPath();
        string safeKey = string.IsNullOrWhiteSpace(robotKey) ? "unitree_g1" : robotKey.Trim();
        return string.IsNullOrWhiteSpace(csvRoot)
            ? string.Empty
            : Path.Combine(csvRoot, safeKey, "live_motion.csv");
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

        List<string> selectedRobotKeysForLaunch = BuildSelectedRobotKeyList();
        string selectedRobot = selectedRobotKeysForLaunch.Count > 0 ? selectedRobotKeysForLaunch[0] : ResolveSelectedRobotName();
        if (string.IsNullOrWhiteSpace(selectedRobot))
        {
            selectedRobot = defaultRobotName.Trim();
        }

        if (string.IsNullOrWhiteSpace(selectedRobot))
        {
            Debug.LogError("未能解析 RoboList 当前机器人名称，请检查下拉列表配置。");
            return false;

        }

        string resolvedRobot = selectedRobotKeysForLaunch.Count > 0 ? selectedRobotKeysForLaunch[0] : ResolveRobotNameForWham(selectedRobot);
        string resolvedRobots = selectedRobotKeysForLaunch.Count > 0 ? string.Join(",", selectedRobotKeysForLaunch) : resolvedRobot;
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

        // Resolve CSV path now so cleanup, Python and Unity CSV readers agree.
        string csvRootPath = ResolveRealtimeCsvRootPath();
        Directory.CreateDirectory(csvRootPath);
        resolvedCsvPath = ResolveRobotRealtimeCsvPath(resolvedRobot);

        // Clear stale CSV before launching so the monitor never sees data from a prior run.
        if (clearCsvOnStart)
        {
            TryClearRealtimeCsvFiles(selectedRobotKeysForLaunch, "OnStart");
        }

        string resolvedScriptPath = ResolveScriptPath(resolvedWorkingDir);
        string resolvedVideoPath = ResolveVideoPath(resolvedWorkingDir);
        string commandArguments = BuildCommandArguments(executablePrefixArgs, resolvedScriptPath);
        ResetDebugLogFile();
        bool streamWhamToUnity = enableTcpStreaming;
        bool streamGmrToUnity = enableTcpStreaming;
        bool effectiveTcpStreaming = streamWhamToUnity || streamGmrToUnity;
        bool effectiveRecordGmrVideo = recordGmrVideo && !(enableTcpStreaming && disablePreviewVideoWhenTcpStreaming);
        bool effectiveRecordWhamVideo = recordWhamVideo && !(enableTcpStreaming && disablePreviewVideoWhenTcpStreaming);
        if (effectiveTcpStreaming && !EnsureUnityStreamReceiverReady(streamWhamToUnity, streamGmrToUnity))
        {
            return false;
        }

        ClearDisabledStreamReceiverTargets(streamWhamToUnity, streamGmrToUnity);
        ClearDisabledRecordingOutputs(resolvedOutputRootPath, effectiveRecordWhamVideo, effectiveRecordGmrVideo);
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
            $"Resolved launch config: selectedRobot='{selectedRobot}', resolvedRobot='{resolvedRobot}', robots='{resolvedRobots}', " +
            $"workingDir='{resolvedWorkingDir}', script='{resolvedScriptPath}', outputRoot='{resolvedOutputRootPath}', " +
            $"csvRoot='{csvRootPath}', csv='{resolvedCsvPath}', video='{resolvedVideoPath}', tcp={effectiveTcpStreaming} (raw={enableTcpStreaming}), " +
            $"tcpWham={streamWhamToUnity}, tcpGmr={streamGmrToUnity}, track={track}, " +
            $"recordGmr={effectiveRecordGmrVideo} (raw={recordGmrVideo}), recordWham={effectiveRecordWhamVideo} (raw={recordWhamVideo}), " +
            $"gmrTorchDevice='{launchGmrTorchDevice}' (requested='{requestedGmrTorchDevice}')");

        if (!effectiveRecordWhamVideo &&
            commandArguments.IndexOf("--record_whamvideo", System.StringComparison.OrdinalIgnoreCase) >= 0)
        {
            SetPipelineStartupError("Record WHAM Video is disabled, but launch arguments still contain --record_whamvideo.");
            return false;
        }

        if (!effectiveRecordGmrVideo &&
            commandArguments.IndexOf("--record_gmrvideo", System.StringComparison.OrdinalIgnoreCase) >= 0)
        {
            SetPipelineStartupError("Record GMR Video is disabled, but launch arguments still contain --record_gmrvideo.");
            return false;
        }

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
            startInfo.EnvironmentVariables["ROBOTS"] = resolvedRobots;
            startInfo.EnvironmentVariables["CSV_ROOT"] = csvRootPath;
            startInfo.EnvironmentVariables["CSV_PATH"] = resolvedCsvPath;
            int launchRobotCount = Mathf.Max(1, selectedRobotKeysForLaunch.Count);
            bool launchLowMemoryMode = launchRobotCount >= 3;
            float launchGmrPreviewFps = launchLowMemoryMode
                ? Mathf.Min(Mathf.Clamp(gmrPreviewFps, 0.5f, 30f), 1.5f)
                : Mathf.Clamp(gmrPreviewFps, 0.5f, 30f);
            int launchWhamRenderQueueSize = launchLowMemoryMode ? 2 : 16;
            int launchPipelineQueueSize = launchLowMemoryMode ? 4 : 10;
            startInfo.EnvironmentVariables["WHAM_LOW_MEMORY_MODE"] = launchLowMemoryMode ? "1" : "0";
            startInfo.EnvironmentVariables["WHAM_PIPELINE_QUEUE_SIZE"] = launchPipelineQueueSize.ToString(CultureInfo.InvariantCulture);
            // Keep this compatible with older PyTorch builds. Some versions reject
            // expandable_segments and abort before WHAM starts.
            startInfo.EnvironmentVariables["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64";
            startInfo.EnvironmentVariables["CUDA_MODULE_LOADING"] = "LAZY";
            // Disable run.ps1's legacy RECORD_VIDEO fallback. The split WHAM/GMR
            // flags below are the only recording source of truth from the HUD.
            startInfo.EnvironmentVariables["RECORD_VIDEO"] = "0";
            startInfo.EnvironmentVariables["RECORD_GMRVIDEO"] = effectiveRecordGmrVideo ? "1" : "0";
            startInfo.EnvironmentVariables["RECORD_WHAMVIDEO"] = effectiveRecordWhamVideo ? "1" : "0";
            if (!effectiveRecordWhamVideo && startInfo.EnvironmentVariables["RECORD_WHAMVIDEO"] != "0")
            {
                SetPipelineStartupError("Invalid launch config: Record WHAM Video is disabled but RECORD_WHAMVIDEO would be enabled.");
                return false;
            }

            if (!effectiveRecordGmrVideo && startInfo.EnvironmentVariables["RECORD_GMRVIDEO"] != "0")
            {
                SetPipelineStartupError("Invalid launch config: Record GMR Video is disabled but RECORD_GMRVIDEO would be enabled.");
                return false;
            }

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
            startInfo.EnvironmentVariables["GMR_MAX_ITER"] = Mathf.Max(1, gmrMaxIter).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["GMR_CSV_FLUSH_INTERVAL"] = Mathf.Max(1, gmrCsvFlushInterval).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["CSV_FLUSH_ROWS"] = Mathf.Max(1, gmrCsvFlushInterval).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["CSV_FLUSH_INTERVAL_MS"] = "100";
            startInfo.EnvironmentVariables["WHAM_TAIL_FLUSH_INTERVAL"] = Mathf.Max(1, whamTailFlushInterval).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["PIPELINE_HEARTBEAT_FRAMES"] = Mathf.Max(1, pipelineHeartbeatFrames).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["CAMERA_FOLLOW"] = (gmrCameraFollow || track) ? "1" : "0";
            startInfo.EnvironmentVariables["CAMERA_LOOKAT_HEIGHT_OFFSET"] = gmrCameraLookatHeightOffset.ToString("F3", CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["CAMERA_ELEVATION"] = gmrCameraElevation.ToString("F3", CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["CAMERA_DISTANCE_SCALE"] = Mathf.Max(0.1f, gmrCameraDistanceScale).ToString("F3", CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["GMR_TCP_RENDER_WIDTH"] = Mathf.Max(320, gmrTcpRenderWidth).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["GMR_TCP_RENDER_HEIGHT"] = Mathf.Max(240, gmrTcpRenderHeight).ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["GMR_PREVIEW_FPS"] = launchGmrPreviewFps.ToString("F2", CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["WHAM_RENDER_QUEUE_SIZE"] = launchWhamRenderQueueSize.ToString(CultureInfo.InvariantCulture);
            startInfo.EnvironmentVariables["GMR_WORKER_QUEUE_SIZE"] = "1";
            if (launchLowMemoryMode)
            {
                AppendDebugLogLine(
                    $"[StartInput] Low-memory realtime mode enabled for {launchRobotCount} robots: " +
                    $"previewFps={launchGmrPreviewFps:F2}, whamRenderQueue={launchWhamRenderQueueSize}, pipelineQueue={launchPipelineQueueSize}.");
            }
            if (!string.IsNullOrWhiteSpace(gmrCameraAzimuth))
            {
                startInfo.EnvironmentVariables["CAMERA_AZIMUTH"] = gmrCameraAzimuth.Trim();
            }
            startInfo.EnvironmentVariables["TRACK"] = track ? "1" : "0";
            startInfo.EnvironmentVariables["TCP"] = effectiveTcpStreaming ? "1" : "0";
            startInfo.EnvironmentVariables["TCP_STREAM_WHAM"] = streamWhamToUnity ? "1" : "0";
            startInfo.EnvironmentVariables["TCP_STREAM_GMR"] = streamGmrToUnity ? "1" : "0";
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
                    HandlePipelineLogLine(args.Data, isErrorStream: false);
                };

                process.ErrorDataReceived += (_, args) =>
                {
                    HandlePipelineLogLine(args.Data, isErrorStream: true);
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
                ", ROBOTS=" + resolvedRobots +
                " (raw=" + selectedRobot + ")" +
                ", VIDEO=" + (string.IsNullOrWhiteSpace(resolvedVideoPath) ? "<run.ps1 default>" : resolvedVideoPath) +
                ", OUTPUT_ROOT=" + resolvedOutputRootPath +
                ", CSV_ROOT=" + csvRootPath +
                ", CSV=" + resolvedCsvPath +
                ", TCP=" + (effectiveTcpStreaming ? "1" : "0") +
                " (raw=" + (enableTcpStreaming ? "1" : "0") + ")" +
                ", TCP_STREAM_WHAM=" + (streamWhamToUnity ? "1" : "0") +
                ", TCP_STREAM_GMR=" + (streamGmrToUnity ? "1" : "0") +
                ", TRACK=" + (track ? "1" : "0") +
                ", RECORD_VIDEO=0" +
                ", RECORD_WHAMVIDEO=" + (effectiveRecordWhamVideo ? "1" : "0") +
                ", RECORD_GMRVIDEO=" + (effectiveRecordGmrVideo ? "1" : "0");
            Debug.Log(pipelineEnvLine);
            AppendDebugLogLine(pipelineEnvLine);
            return true;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"启动 Bash 命令失败: {e.Message}");
            return false;
        }
    }

    private void ClearDisabledStreamReceiverTargets(bool streamWhamToUnity, bool streamGmrToUnity)
    {
        StreamReceiver receiver = FindObjectOfType<StreamReceiver>(true);
        if (receiver == null)
        {
            return;
        }

        if (!streamWhamToUnity && !streamGmrToUnity)
        {
            receiver.ClearAllStreams();
            return;
        }

        if (!streamWhamToUnity)
        {
            receiver.ClearStream(0);
        }

        if (!streamGmrToUnity)
        {
            receiver.ClearStream(1);
        }
    }

    private bool EnsureUnityStreamReceiverReady(bool streamWhamToUnity, bool streamGmrToUnity)
    {
        StreamReceiver receiver = StreamReceiver.EnsureReceiverHost();
        if (receiver == null || !receiver.IsListening)
        {
            SetPipelineStartupError("Unity TCP StreamReceiver is not listening; Python would receive WinError 10061.");
            return false;
        }

        Debug.Log(
            $"[StartInput] Unity TCP StreamReceiver ready on {receiver.BindAddress}:{receiver.Port} " +
            $"(WHAM={(streamWhamToUnity ? "on" : "off")}, GMR={(streamGmrToUnity ? "on" : "off")}).");
        return true;
    }

    private void ClearDisabledRecordingOutputs(string outputRootPath, bool effectiveRecordWhamVideo, bool effectiveRecordGmrVideo)
    {
        if (string.IsNullOrWhiteSpace(outputRootPath))
        {
            return;
        }

        if (!effectiveRecordWhamVideo)
        {
            TryDeleteStaleOutputFile(Path.Combine(outputRootPath, "stream_demo", "output.mp4"));
        }

        if (!effectiveRecordGmrVideo)
        {
            TryDeleteStaleOutputFile(Path.Combine(outputRootPath, "video", "live_stream_robot.mp4"));
        }
    }

    private void TryDeleteStaleOutputFile(string filePath)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(filePath) && File.Exists(filePath))
            {
                File.Delete(filePath);
                AppendDebugLogLine($"[StartInput] Deleted stale disabled recording output: {filePath}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[StartInput] Could not delete stale recording output '{filePath}': {e.Message}");
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

        return StartMultiCsvMonitorForSelectedRobots();
    }
#if false

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
        ResetRealtimeCsvSafetyGate();
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

#endif
    private bool StartMultiCsvMonitorForSelectedRobots()
    {
        List<string> robotKeys = BuildSelectedRobotKeyList();
        if (robotKeys.Count == 0)
        {
            Debug.LogError("[StartInput] No robot is selected, cannot start realtime CSV monitor.");
            return false;
        }

        string csvRootPath = ResolveRealtimeCsvRootPath();
        if (string.IsNullOrWhiteSpace(csvRootPath))
        {
            Debug.LogError("[StartInput] Realtime CSV root path is empty; cannot start monitor.");
            return false;
        }

        StopCsvMonitor();
        EndActiveRealtimeCsv();
        realtimeRobotStates.Clear();

        foreach (string robotKey in robotKeys)
        {
            IMimicAgent mimicAgent = ResolveAgentByRobotKey(robotKey);
            IRealtimeCsvMimicAgent csvAgent = mimicAgent as IRealtimeCsvMimicAgent;
            if (mimicAgent == null || csvAgent == null)
            {
                Debug.LogError($"[StartInput] Selected robot '{robotKey}' does not provide realtime CSV ingestion.");
                realtimeRobotStates.Clear();
                return false;
            }

            string csvPath = ResolveRobotRealtimeCsvPath(robotKey);
            if (string.IsNullOrWhiteSpace(csvPath))
            {
                Debug.LogError($"[StartInput] Could not resolve realtime CSV path for robot '{robotKey}'.");
                realtimeRobotStates.Clear();
                return false;
            }

            realtimeRobotStates[robotKey] = new RealtimeRobotState
            {
                RobotKey = robotKey,
                CsvPath = csvPath,
                MimicAgent = mimicAgent,
                CsvAgent = csvAgent
            };
        }

        RealtimeRobotState firstState = realtimeRobotStates[robotKeys[0]];
        activeRealtimeMimicAgent = firstState.MimicAgent;
        activeRealtimeCsvAgent = firstState.CsvAgent;
        resolvedCsvPath = firstState.CsvPath;

        if (robotKeys.Count > 1)
        {
            ApplySelectedRobotDisplayOffsets(new HashSet<string>(robotKeys, System.StringComparer.OrdinalIgnoreCase));
        }
        else
        {
            ResetRobotDisplayOffsets();
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
        ResetRealtimeCsvSafetyGate();
        pendingRealtimeRows.Clear();
        DrainRealtimeCsvQueues();
        if (writeDebugLogFile && string.IsNullOrWhiteSpace(resolvedDebugLogPath))
        {
            resolvedDebugLogPath = ResolveDebugLogPath();
            EnsureDebugLogWriter();
        }

        foreach (RealtimeRobotState state in realtimeRobotStates.Values)
        {
            StartCsvWorker(state.CsvPath, state.CsvAgent.ExpectedCsvColumns, state.RobotKey);
        }

        LogVerbose(
            $"CSV workers started: robots='{string.Join(",", robotKeys)}', root='{csvRootPath}', pollInterval={csvPollInterval}, realtimeWarmupSourceRows={realtimeWarmupSourceRows}");

        Debug.Log($"[StartInput] Started realtime CSV monitor for robots: {string.Join(", ", robotKeys)}; csvRoot={csvRootPath}");
        return true;
    }

    private void StopCsvMonitor()
    {
        csvWorkerRunning = false;
        for (int i = 0; i < csvWorkerThreads.Count; i++)
        {
            Thread thread = csvWorkerThreads[i];
            if (thread != null && thread.IsAlive)
            {
                try { thread.Join(250); } catch { }
            }
        }
        csvWorkerThreads.Clear();

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
        ResetRobotDisplayOffsets();
    }

    private void DrainRealtimeCsvQueues()
    {
        while (realtimeCsvRowsQueue.TryDequeue(out RealtimeCsvBatch _)) { }
        while (realtimeCsvResetQueue.TryDequeue(out string _)) { }
    }

    private void StartCsvWorker(string csvPath, int expectedColumns, string robotKey)
    {
        csvWorkerRunning = true;
        int sleepMs = System.Math.Max(20, (int)System.Math.Round(csvPollInterval * 1000f));
        float emptyWarningSeconds = System.Math.Max(60f, csvNoDataWarningSeconds);
        bool logChanges = logCsvChangeEvents && emitRealtimeCsvAppendLogsToConsole;
        bool estimateProducerFps = estimateRealtimeCsvProducerFps;
        float configuredFallbackFps = ClampRealtimeCsvFps(defaultRealtimeCsvFps);
        float fallbackFps = estimateProducerFps
            ? ClampRealtimeCsvFps((float)System.Math.Min(configuredFallbackFps, 5f))
            : configuredFallbackFps;
        float fpsWindowSeconds = (float)System.Math.Max(0.25, realtimeCsvFpsWindowSeconds);
        float changeLogIntervalSeconds = (float)System.Math.Max(0.1, csvChangeLogIntervalSeconds);

        csvWorkerThread = new Thread(() => CsvWorkerLoop(csvPath, expectedColumns, robotKey, sleepMs, logChanges, emptyWarningSeconds, fallbackFps, estimateProducerFps, fpsWindowSeconds, changeLogIntervalSeconds))
        {
            Name = "StartInput-CsvWorker",
            IsBackground = true
        };
        csvWorkerThreads.Add(csvWorkerThread);
        csvWorkerThread.Start();
    }

    private void CsvWorkerLoop(string csvPath, int expectedColumns, string robotKey, int sleepMs, bool logChanges, float emptyWarningSeconds, float fallbackFps, bool estimateProducerFps, float fpsWindowSeconds, float changeLogIntervalSeconds)
    {
        bool missingLogged = false;
        bool firstDataLogged = false;
        bool invalidRowLogged = false;
        string lastReadError = string.Empty;
        long readOffset = 0L;
        string pendingText = string.Empty;
        long previousLength = -1L;
        System.DateTime previousWriteTimeUtc = System.DateTime.MinValue;
        System.DateTime monitorStartUtc = System.DateTime.UtcNow;
        System.DateTime lastEmptyWarningUtc = System.DateTime.MinValue;
        System.DateTime lastChangeLogUtc = System.DateTime.MinValue;
        Queue<RealtimeFpsSample> fpsSamples = new Queue<RealtimeFpsSample>();
        float producerFps = fallbackFps;

        QueueUnityLog(
            $"[StartInput] Realtime CSV worker active: robot='{robotKey}', expectedColumns={expectedColumns}, csv='{csvPath}', pollMs={sleepMs}, fallbackFps={fallbackFps:F2}, estimateFps={estimateProducerFps}, fpsWindow={fpsWindowSeconds:F2}s, changeLogInterval={changeLogIntervalSeconds:F2}s");

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
                    invalidRowLogged = false;
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
                    double emptyElapsedSeconds = (nowUtc - monitorStartUtc).TotalSeconds;
                    if (emptyElapsedSeconds >= emptyWarningSeconds &&
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

                        QueuedUnityLogType emptyLogType = (!processAlive || emptyElapsedSeconds >= System.Math.Max(120f, emptyWarningSeconds * 2f))
                            ? QueuedUnityLogType.Warning
                            : QueuedUnityLogType.Log;

                        QueueUnityLog(
                            $"[StartInput] live_motion.csv is still empty. elapsed={emptyElapsedSeconds:F1}s, " +
                            $"processAlive={processAlive}, pid={pid}, csv='{csvPath}'. Check [Pipeline]/[Pipeline-ERR] logs.",
                            emptyLogType);
                        lastEmptyWarningUtc = nowUtc;
                    }

                    Thread.Sleep(sleepMs);
                    continue;
                }

                if (currentLength > readOffset)
                {
                    List<float[]> rows = ReadNewRealtimeCsvRowsFromPath(csvPath, currentLength, expectedColumns, robotKey, ref readOffset, ref pendingText, ref invalidRowLogged);
                    if (rows.Count > 0)
                    {
                        System.DateTime rowsUtc = System.DateTime.UtcNow;
                        if (estimateProducerFps)
                        {
                            producerFps = EstimateCsvProducerFps(fpsSamples, rowsUtc, rows.Count, fpsWindowSeconds, producerFps);
                        }

                        if (!firstDataLogged)
                        {
                            firstDataLogged = true;
                            QueueUnityLog($"[StartInput] Realtime CSV has data: {csvPath} ({currentLength} bytes), producerFps={producerFps:F2}");
                        }

                        if (logChanges &&
                            (lastChangeLogUtc == System.DateTime.MinValue || (rowsUtc - lastChangeLogUtc).TotalSeconds >= changeLogIntervalSeconds))
                        {
                            QueueUnityLog($"[StartInput] CSV appended: rows={rows.Count}, bytes={currentLength}, offset={readOffset}, producerFps={producerFps:F2}, path='{csvPath}'");
                            lastChangeLogUtc = rowsUtc;
                        }

                        realtimeCsvRowsQueue.Enqueue(new RealtimeCsvBatch(robotKey, rows, producerFps));
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

    private List<float[]> ReadNewRealtimeCsvRowsFromPath(
        string csvPath,
        long currentLength,
        int expectedColumns,
        string robotKey,
        ref long readOffset,
        ref string pendingText,
        ref bool invalidRowLogged)
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
            if (TryParseRealtimeCsvLine(lines[i], expectedColumns, out float[] row, out int actualColumns, out string failureReason))
            {
                rows.Add(row);
            }
            else if (!invalidRowLogged && !string.IsNullOrWhiteSpace(lines[i]))
            {
                QueueUnityLog(
                    $"[StartInput] Realtime CSV row rejected for robot='{robotKey}': expectedColumns={expectedColumns}, actualColumns={actualColumns}, reason={failureReason}, csv='{csvPath}'. " +
                    "This usually means the WHAM/GMR --robot output does not match RoboList, or a stale/corrupt live_motion.csv is being read.",
                    QueuedUnityLogType.Warning);
                invalidRowLogged = true;
            }
        }

        return rows;
    }

    private static float EstimateCsvProducerFps(
        Queue<RealtimeFpsSample> samples,
        System.DateTime nowUtc,
        int rowCount,
        float windowSeconds,
        float fallbackFps)
    {
        if (samples == null || rowCount <= 0)
        {
            return fallbackFps;
        }

        samples.Enqueue(new RealtimeFpsSample(nowUtc, rowCount));
        double safeWindowSeconds = System.Math.Max(0.25, windowSeconds);
        while (samples.Count > 0 && (nowUtc - samples.Peek().TimeUtc).TotalSeconds > safeWindowSeconds)
        {
            samples.Dequeue();
        }

        if (samples.Count < 2)
        {
            return fallbackFps;
        }

        bool skippedFirst = false;
        int rowsAfterFirst = 0;
        System.DateTime firstSampleUtc = System.DateTime.MinValue;
        foreach (RealtimeFpsSample sample in samples)
        {
            if (!skippedFirst)
            {
                firstSampleUtc = sample.TimeUtc;
                skippedFirst = true;
                continue;
            }

            rowsAfterFirst += sample.RowCount;
        }

        double seconds = (nowUtc - firstSampleUtc).TotalSeconds;
        if (seconds <= 0.001 || rowsAfterFirst <= 0)
        {
            return fallbackFps;
        }

        return ClampRealtimeCsvFps(rowsAfterFirst / (float)seconds);
    }

    private static float ClampRealtimeCsvFps(float framesPerSecond)
    {
        if (float.IsNaN(framesPerSecond) || float.IsInfinity(framesPerSecond) || framesPerSecond <= 0f)
        {
            return ReplayCsvUtility.SourceFps;
        }

        return (float)System.Math.Max(
            ReplayCsvUtility.MinRealtimeFps,
            System.Math.Min(ReplayCsvUtility.MaxRealtimeFps, framesPerSecond));
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
                    if (logCsvChangeEvents && emitRealtimeCsvAppendLogsToConsole)
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

                    if (logCsvChangeEvents && emitRealtimeCsvAppendLogsToConsole)
                    {
                        LogVerbose($"CSV appended: rows={rows.Count}, bytes={currentLength}, offset={csvReadOffset}, path='{resolvedCsvPath}'");
                    }

                    float fallbackPlaybackFps = estimateRealtimeCsvProducerFps
                        ? ClampRealtimeCsvFps((float)System.Math.Min(ClampRealtimeCsvFps(defaultRealtimeCsvFps), 5f))
                        : ClampRealtimeCsvFps(defaultRealtimeCsvFps);
                    ApplyRealtimeCsvRows(rows, fallbackPlaybackFps);
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
        return TryParseRealtimeCsvLine(line, expectedColumns, out row, out _, out _);
    }

    private bool TryParseRealtimeCsvLine(string line, int expectedColumns, out float[] row, out int actualColumns, out string failureReason)
    {
        row = null;
        actualColumns = 0;
        failureReason = string.Empty;
        if (string.IsNullOrWhiteSpace(line) || expectedColumns <= 0)
        {
            failureReason = string.IsNullOrWhiteSpace(line) ? "empty line" : "invalid expectedColumns";
            return false;
        }

        string[] tokens = line.Split(',');
        actualColumns = tokens.Length;
        if (tokens.Length != expectedColumns)
        {
            failureReason = "column count mismatch";
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
                failureReason = $"invalid float at column {i}";
                return false;
            }

            row[i] = value;
        }

        return true;
    }

    private float ResolveRealtimePlaybackBufferSeconds(string robotKey, float playbackFps)
    {
        float configuredBufferSeconds = Mathf.Max(0f, realtimePlaybackBufferSeconds);
        float safePlaybackFps = Mathf.Max(1f, playbackFps);
        if (string.Equals(robotKey, "x02lite", System.StringComparison.OrdinalIgnoreCase))
        {
            float oneFrameSeconds = 1f / safePlaybackFps;
            return Mathf.Min(configuredBufferSeconds, oneFrameSeconds);
        }

        float twoFrameSeconds = 2f / safePlaybackFps;
        return Mathf.Min(configuredBufferSeconds, twoFrameSeconds);
    }

    private void ApplyRealtimeCsvRows(List<float[]> rows, float producerFps)
    {
        ApplyRealtimeCsvRows(ResolvePrimarySelectedRobotKey(), rows, producerFps);
    }

    private void ApplyRealtimeCsvRows(string robotKey, List<float[]> rows, float producerFps)
    {
        if (rows == null || rows.Count == 0)
        {
            return;
        }

        RealtimeRobotState state = null;
        if (string.IsNullOrWhiteSpace(robotKey) || !realtimeRobotStates.TryGetValue(robotKey, out state))
        {
            if (activeRealtimeMimicAgent != null && activeRealtimeCsvAgent != null)
            {
                state = new RealtimeRobotState
                {
                    RobotKey = activeRealtimeMimicAgent.RobotKey,
                    CsvPath = resolvedCsvPath,
                    MimicAgent = activeRealtimeMimicAgent,
                    CsvAgent = activeRealtimeCsvAgent,
                    ReplayBootstrapped = replayBootstrapped,
                    SourceRowsRead = realtimeSourceRowsRead,
                    LastAcceptedRow = lastAcceptedRealtimeRow,
                    ConsecutiveRejectedRows = consecutiveRejectedRealtimeRows,
                    SafetyWarningLogged = realtimeSafetyWarningLogged
                };
                state.PendingRows.AddRange(pendingRealtimeRows);
            }
        }

        if (state == null || state.MimicAgent == null || state.CsvAgent == null)
        {
            Debug.LogWarning("[StartInput] Realtime CSV rows arrived but no realtime-capable agent is active.");
            return;
        }

        int receivedRowCount = rows.Count;
        rows = FilterSafeRealtimeCsvRows(rows, state);
        if (rows.Count == 0)
        {
            return;
        }

        state.SourceRowsRead += receivedRowCount;
        float playbackFps = ClampRealtimeCsvFps(producerFps);
        float playbackBufferSeconds = ResolveRealtimePlaybackBufferSeconds(state.MimicAgent.RobotKey, playbackFps);
        state.CsvAgent.SetRealtimePlaybackRate(playbackFps, playbackBufferSeconds);

        if (!state.ReplayBootstrapped)
        {
            state.PendingRows.AddRange(rows);
            int warmupRows = Mathf.Max(1, realtimeWarmupSourceRows);
            if (state.PendingRows.Count < warmupRows)
            {
                return;
            }

            if (!state.CsvAgent.BeginRealtimeCsv())
            {
                Debug.LogError($"[StartInput] Realtime CSV bootstrap failed for '{state.MimicAgent.RobotKey}': agent refused BeginRealtimeCsv.");
                AbortRealtimeCsvBootstrap(state);
                return;
            }

            if (!state.CsvAgent.AppendRealtimeCsvRows(state.PendingRows))
            {
                Debug.LogWarning($"[StartInput] Initial realtime CSV append produced no frames for '{state.MimicAgent.RobotKey}'.");
                AbortRealtimeCsvBootstrap(state);
                return;
            }

            state.MimicAgent.UseExternalReplayData = true;
            state.MimicAgent.ReplayMode = true;
            state.PendingRows.Clear();
            if (MimicAgentRegistry.Instance != null && realtimeRobotStates.Count <= 1)
            {
                MimicAgentRegistry.Instance.SetActiveTarget(state.MimicAgent);
            }

            if (restartEpisodeOnFirstCsv)
            {
                state.MimicAgent.RequestEndEpisode();
            }

            state.ReplayBootstrapped = true;
            string playbackStartedMessage =
                $"[StartInput] Realtime CSV playback started for '{state.MimicAgent.RobotKey}' after {state.SourceRowsRead} source rows: {state.CsvPath}. " +
                $"expectedColumns={state.CsvAgent.ExpectedCsvColumns}, warmupRows={Mathf.Max(1, realtimeWarmupSourceRows)}, " +
                $"producerFps={playbackFps:F2}, playbackBuffer={playbackBufferSeconds:F2}s, selectedRobot='{ResolveSelectedRobotName()}', bufferedBatches={realtimeCsvRowsQueue.Count}, " +
                $"rootMapping=csv[0..2]->Unity(-y,z,x), quat=csv[3..6]->Unity(-y,z,x,-w)";
            if (AllRealtimeRobotStatesBootstrapped())
            {
                FinishPipelineStartup(playbackStartedMessage);
            }
            else
            {
                AdvancePipelineStartup(
                    "Waiting for all selected robots",
                    playbackStartedMessage,
                    0.86f,
                    0.01f);
            }
            Debug.Log(playbackStartedMessage);
            return;
        }

        state.CsvAgent.AppendRealtimeCsvRows(rows);
    }

    private bool AllRealtimeRobotStatesBootstrapped()
    {
        if (realtimeRobotStates.Count == 0)
        {
            return false;
        }

        foreach (RealtimeRobotState state in realtimeRobotStates.Values)
        {
            if (state == null || !state.ReplayBootstrapped)
            {
                return false;
            }
        }

        return true;
    }

    private List<float[]> FilterSafeRealtimeCsvRows(List<float[]> rows, RealtimeRobotState state)
    {
        if (!enableRealtimeCsvSafetyGate || rows == null || rows.Count == 0)
        {
            return rows;
        }

        int expectedColumns = state?.CsvAgent != null ? state.CsvAgent.ExpectedCsvColumns : 0;
        var accepted = new List<float[]>(rows.Count);
        for (int i = 0; i < rows.Count; i++)
        {
            float[] row = rows[i];
            if (IsRealtimeCsvRowSafe(row, expectedColumns, state?.LastAcceptedRow, out string reason))
            {
                accepted.Add(row);
                if (state != null)
                {
                    state.LastAcceptedRow = row;
                    state.ConsecutiveRejectedRows = 0;
                    state.SafetyWarningLogged = false;
                }
                continue;
            }

            if (state != null)
            {
                state.ConsecutiveRejectedRows++;
            }
            int rejectedRows = state?.ConsecutiveRejectedRows ?? 1;
            string message =
                $"[StartInput] Realtime CSV safety gate rejected row for '{state?.RobotKey ?? "<none>"}': {reason}. " +
                $"consecutiveRejected={rejectedRows}/{Mathf.Max(1, realtimeMaxConsecutiveRejectedRows)}, csv='{state?.CsvPath ?? resolvedCsvPath}'.";
            if (state == null || !state.SafetyWarningLogged || rejectedRows >= Mathf.Max(1, realtimeMaxConsecutiveRejectedRows))
            {
                Debug.LogWarning(message);
                AppendDebugLogLine(message);
                if (state != null)
                {
                    state.SafetyWarningLogged = true;
                }
            }

            if (rejectedRows >= Mathf.Max(1, realtimeMaxConsecutiveRejectedRows))
            {
                AbortUnsafeRealtimeCsv(state, message);
                break;
            }
        }

        return accepted;
    }

    private bool IsRealtimeCsvRowSafe(float[] row, int expectedColumns, float[] previousAcceptedRow, out string reason)
    {
        reason = string.Empty;
        if (row == null)
        {
            reason = "row is null";
            return false;
        }

        if (expectedColumns > 0 && row.Length != expectedColumns)
        {
            reason = $"column count mismatch ({row.Length} != {expectedColumns})";
            return false;
        }

        if (row.Length < 7)
        {
            reason = $"row has too few columns ({row.Length})";
            return false;
        }

        for (int i = 0; i < row.Length; i++)
        {
            float value = row[i];
            if (float.IsNaN(value) || float.IsInfinity(value))
            {
                reason = $"non-finite value at column {i}";
                return false;
            }
        }

        if (row.Length >= 7)
        {
            float maxAbsRoot = Mathf.Max(1f, realtimeMaxAbsRootPositionMeters);
            for (int i = 0; i < 3; i++)
            {
                if (Mathf.Abs(row[i]) > maxAbsRoot)
                {
                    reason = $"root column {i} absolute value {row[i]:F3}m > {maxAbsRoot:F3}m";
                    return false;
                }
            }

            float quatMagnitude = Mathf.Sqrt(row[3] * row[3] + row[4] * row[4] + row[5] * row[5] + row[6] * row[6]);
            if (quatMagnitude < 0.1f)
            {
                reason = $"invalid root quaternion magnitude {quatMagnitude:F4}";
                return false;
            }
        }

        float maxAbsJoint = Mathf.Max(1f, realtimeMaxAbsJointRadians);
        for (int i = 7; i < row.Length; i++)
        {
            if (Mathf.Abs(row[i]) > maxAbsJoint)
            {
                reason = $"joint column {i} absolute value {row[i]:F3}rad > {maxAbsJoint:F3}rad";
                return false;
            }
        }

        if (previousAcceptedRow == null || previousAcceptedRow.Length != row.Length)
        {
            return true;
        }

        float rootJump = Vector3.Distance(
            new Vector3(previousAcceptedRow[0], previousAcceptedRow[1], previousAcceptedRow[2]),
            new Vector3(row[0], row[1], row[2]));
        if (rootJump > Mathf.Max(0.01f, realtimeMaxRootJumpMeters))
        {
            reason = $"root jump {rootJump:F3}m > {realtimeMaxRootJumpMeters:F3}m";
            return false;
        }

        if (row.Length >= 7)
        {
            Quaternion previous = NormalizeCsvQuaternion(previousAcceptedRow);
            Quaternion current = NormalizeCsvQuaternion(row);
            float angle = Quaternion.Angle(previous, current);
            if (angle > Mathf.Max(1f, realtimeMaxRootRotationJumpDegrees))
            {
                reason = $"root rotation jump {angle:F1}deg > {realtimeMaxRootRotationJumpDegrees:F1}deg";
                return false;
            }
        }

        float jointLimit = Mathf.Max(0.01f, realtimeMaxJointJumpRadians);
        for (int i = 7; i < row.Length; i++)
        {
            float delta = Mathf.Abs(row[i] - previousAcceptedRow[i]);
            if (delta > jointLimit)
            {
                reason = $"joint column {i} jump {delta:F3}rad > {jointLimit:F3}rad";
                return false;
            }
        }

        return true;
    }

    private static Quaternion NormalizeCsvQuaternion(float[] row)
    {
        var q = new Quaternion(row[3], row[4], row[5], row[6]);
        float magnitude = Mathf.Sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        if (magnitude < 0.0001f)
        {
            return Quaternion.identity;
        }

        float inv = 1f / magnitude;
        return new Quaternion(q.x * inv, q.y * inv, q.z * inv, q.w * inv);
    }

    private void ResetRealtimeCsvSafetyGate()
    {
        lastAcceptedRealtimeRow = null;
        consecutiveRejectedRealtimeRows = 0;
        realtimeSafetyWarningLogged = false;
    }

    private void AbortUnsafeRealtimeCsv(RealtimeRobotState state, string message)
    {
        SetPipelineStartupError(message);
        if (state != null)
        {
            state.CsvAgent?.EndRealtimeCsv();
            state.PendingRows.Clear();
            state.ReplayBootstrapped = false;
            state.MimicAgent.UseExternalReplayData = false;
            state.MimicAgent.ReplayMode = false;
        }
    }

    private void AbortRealtimeCsvBootstrap()
    {
        AbortRealtimeCsvBootstrap(null);
    }

    private void AbortRealtimeCsvBootstrap(RealtimeRobotState state)
    {
        if (state != null)
        {
            state.PendingRows.Clear();
            state.ReplayBootstrapped = false;
            if (state.MimicAgent != null)
            {
                state.MimicAgent.UseExternalReplayData = false;
                state.MimicAgent.ReplayMode = false;
            }
            state.CsvAgent?.EndRealtimeCsv();
            return;
        }

        pendingRealtimeRows.Clear();
        replayBootstrapped = false;

        if (activeRealtimeMimicAgent != null)
        {
            activeRealtimeMimicAgent.UseExternalReplayData = false;
            activeRealtimeMimicAgent.ReplayMode = false;
        }
        StopCsvMonitor();
        EndActiveRealtimeCsv();
        SetRealtimeDropdownsInteractable(true);
    }

    private void ResetRealtimeCsvReader(string reason)
    {
        LogVerbose($"Reset realtime CSV reader: {reason}");
        foreach (RealtimeRobotState state in realtimeRobotStates.Values)
        {
            if (state == null)
            {
                continue;
            }

            state.SourceRowsRead = 0;
            state.LastAcceptedRow = null;
            state.ConsecutiveRejectedRows = 0;
            state.SafetyWarningLogged = false;
            state.ReplayBootstrapped = false;
            state.PendingRows.Clear();
            state.CsvAgent?.EndRealtimeCsv();
            if (state.MimicAgent != null)
            {
                state.MimicAgent.UseExternalReplayData = false;
                state.MimicAgent.ReplayMode = false;
            }
        }

        csvReadOffset = 0;
        csvPendingText = string.Empty;
        realtimeSourceRowsRead = 0;
        ResetRealtimeCsvSafetyGate();
        replayBootstrapped = false;
        firstNonEmptyCsvLogged = false;
        pendingRealtimeRows.Clear();
        activeRealtimeCsvAgent?.EndRealtimeCsv();
        if (activeRealtimeMimicAgent != null)
        {
            activeRealtimeMimicAgent.UseExternalReplayData = false;
            activeRealtimeMimicAgent.ReplayMode = false;
        }
    }

    private void EndActiveRealtimeCsv()
    {
        foreach (RealtimeRobotState state in realtimeRobotStates.Values)
        {
            if (state == null)
            {
                continue;
            }

            state.CsvAgent?.EndRealtimeCsv();
            state.PendingRows.Clear();
            state.ReplayBootstrapped = false;
            state.SourceRowsRead = 0;
            state.LastAcceptedRow = null;
            state.ConsecutiveRejectedRows = 0;
            state.SafetyWarningLogged = false;
        }
        realtimeRobotStates.Clear();

        activeRealtimeCsvAgent?.EndRealtimeCsv();
        activeRealtimeCsvAgent = null;
        activeRealtimeMimicAgent = null;
        pendingRealtimeRows.Clear();
        csvPendingText = string.Empty;
        csvReadOffset = 0;
        realtimeSourceRowsRead = 0;
        ResetRealtimeCsvSafetyGate();
        replayBootstrapped = false;
        ResetRobotDisplayOffsets();
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
            $"csv='{resolvedCsvPath}', robots=[{BuildRealtimeCsvStatusSummary()}]. " +
            "Check [Pipeline]/[Pipeline-ERR] logs for [Input], [Detect], [WHAM], [GMR], and [TCP] messages.";
        Debug.LogWarning(warning);
        AppendDebugLogLine(warning);
    }

    private string BuildRealtimeCsvStatusSummary()
    {
        if (realtimeRobotStates.Count == 0)
        {
            return "<none>";
        }

        var parts = new List<string>(realtimeRobotStates.Count);
        foreach (RealtimeRobotState state in realtimeRobotStates.Values)
        {
            if (state == null)
            {
                continue;
            }

            long length = -1;
            bool exists = false;
            try
            {
                if (!string.IsNullOrWhiteSpace(state.CsvPath))
                {
                    var info = new FileInfo(state.CsvPath);
                    exists = info.Exists;
                    length = exists ? info.Length : -1;
                }
            }
            catch
            {
                length = -1;
            }

            parts.Add(
                $"{state.RobotKey}:exists={exists},bytes={length},rows={state.SourceRowsRead},pending={state.PendingRows.Count},boot={state.ReplayBootstrapped}");
        }

        return parts.Count > 0 ? string.Join("; ", parts) : "<none>";
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

    private IMimicAgent ResolveAgentByRobotKey(string robotKey)
    {
        string selectedKey = ResolveRobotNameForWham(robotKey);
        if (string.IsNullOrWhiteSpace(selectedKey))
        {
            return null;
        }

        if (MimicAgentRegistry.Instance != null)
        {
            IMimicAgent byKey = MimicAgentRegistry.Instance.FindByKey(selectedKey);
            if (byKey != null && byKey.AgentGameObject != null)
            {
                return byKey;
            }
        }

        if (targetAgentBehaviour is IMimicAgent pinned &&
            pinned.AgentGameObject != null &&
            string.Equals(pinned.RobotKey, selectedKey, System.StringComparison.OrdinalIgnoreCase))
        {
            return pinned;
        }

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
        if (string.IsNullOrWhiteSpace(outputRoot))
        {
            return string.Empty;
        }

        string expanded = ExpandPathPrefix(outputRoot);
        string normalized = NormalizePathSeparators(expanded);

        try
        {
            if (Path.IsPathRooted(normalized))
            {
                return Path.GetFullPath(normalized);
            }

            if (keepRuntimeOutputOutsideAssets)
            {
                string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
                if (!string.IsNullOrWhiteSpace(projectRoot))
                {
                    return Path.GetFullPath(Path.Combine(projectRoot, "Library", "ImitationRuntime", normalized));
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to resolve OUTPUT_ROOT: {outputRoot} ({e.Message})");
            return string.Empty;
        }

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
        ResolveRealtimeControlButtons();

        if (roboListDropdown != null)
        {
            roboListDropdown.interactable = interactable;
        }

        if (csvListDropdown != null)
        {
            csvListDropdown.interactable = interactable;
        }

        if (startButton != null)
        {
            startButton.interactable = interactable;
        }

        if (replayButton != null)
        {
            replayButton.interactable = interactable;
        }

        if (stopButton != null)
        {
            stopButton.interactable = true;
        }

        LogVerbose(
            $"Realtime controls interactable: dropdowns={interactable}, start={interactable}, replay={interactable}, stop={stopButton != null && stopButton.interactable}");
    }

    private void ResolveRealtimeControlButtons()
    {
        if (startButton == null)
        {
            startButton = GetComponent<Button>();
        }

        if (replayButton == null)
        {
            replayButton = ResolveButtonByObjectName(replayButtonObjectName);
            if (replayButton == null)
            {
                Replay replayComponent = FindObjectOfType<Replay>();
                if (replayComponent != null)
                {
                    replayButton = replayComponent.GetComponent<Button>();
                }
            }
        }

        if (stopButton == null)
        {
            stopButton = ResolveButtonByObjectName(stopButtonObjectName);
            if (stopButton == null)
            {
                Stop stopComponent = FindObjectOfType<Stop>();
                if (stopComponent != null)
                {
                    stopButton = stopComponent.GetComponent<Button>();
                }
            }
        }
    }

    private Button ResolveButtonByObjectName(string objectName)
    {
        if (string.IsNullOrWhiteSpace(objectName))
        {
            return null;
        }

        GameObject buttonObject = GameObject.Find(objectName);
        return buttonObject != null ? buttonObject.GetComponent<Button>() : null;
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

    private List<string> BuildSelectedRobotKeyList()
    {
        var result = new List<string>();
        foreach (string supported in MultiRobotDisplayOrder)
        {
            if (selectedRobotKeys.Contains(supported))
            {
                result.Add(supported);
            }
        }

        foreach (string supported in SupportedRobotNames)
        {
            if (selectedRobotKeys.Contains(supported) && !result.Contains(supported))
            {
                result.Add(supported);
            }
        }

        if (result.Count > 0)
        {
            return result;
        }

        if (hasExplicitRobotSelection)
        {
            return result;
        }

        string selected = ResolveSelectedRobotName();
        string selectedKey = TryResolveRobotKeyQuiet(selected);
        if (string.IsNullOrWhiteSpace(selectedKey))
        {
            selectedKey = TryResolveRobotKeyQuiet(defaultRobotName);
        }

        if (!string.IsNullOrWhiteSpace(selectedKey))
        {
            result.Add(selectedKey);
        }

        return result;
    }

    private HashSet<string> BuildSelectedRobotKeySet()
    {
        return new HashSet<string>(BuildSelectedRobotKeyList(), System.StringComparer.OrdinalIgnoreCase);
    }

    private string ResolvePrimarySelectedRobotKey()
    {
        List<string> keys = BuildSelectedRobotKeyList();
        return keys.Count > 0 ? keys[0] : string.Empty;
    }

    private void RefreshRegisteredCsvBrowsers()
    {
        foreach (var kv in robotCsvBrowsers)
        {
            if (kv.Value != null)
            {
                kv.Value.SetCsvRobotFilter(kv.Key);
            }
        }
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
