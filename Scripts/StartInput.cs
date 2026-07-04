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

public partial class StartInput : MonoBehaviour
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
    private readonly ConcurrentQueue<string> archivedCsvRefreshQueue = new ConcurrentQueue<string>();
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

    private static readonly HashSet<string> SupportedRobotNames =
        new HashSet<string>(RobotCatalog.SupportedPipelineKeys, System.StringComparer.OrdinalIgnoreCase);

    private static readonly string[] MultiRobotDisplayOrder = RobotCatalog.PrimaryDisplayOrder;

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
        public float LastPlaybackFps = ReplayCsvUtility.SourceFps;
        public float[] LastAcceptedRow;
        public int ConsecutiveRejectedRows;
        public bool SafetyWarningLogged;
        public readonly List<float[]> PendingRows = new List<float[]>();
    }

    private sealed class RealtimeCsvArchiveRequest
    {
        public string RobotKey;
        public string SourcePath;
        public string DestinationDirectory;
        public float SourceFps = ReplayCsvUtility.SourceFps;
        public bool RealtimeCameraSource;
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
        ConsumeArchivedCsvRefreshQueue();
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

    private void ConsumeArchivedCsvRefreshQueue()
    {
        bool anyArchived = false;
        while (archivedCsvRefreshQueue.TryDequeue(out _))
        {
            anyArchived = true;
        }

        if (!anyArchived)
        {
            return;
        }

        RefreshRegisteredCsvBrowsers();
        RefreshCsvListForSelectedRobot();
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

    public void StopStartPipeline(bool archiveRealtimeCsv = false)
    {
        PipelineStartupActive = false;
        PipelineStatusText = "Retargeting stopped";
        PipelineActivityText = "Stop requested.";
        List<RealtimeCsvArchiveRequest> archiveRequests = archiveRealtimeCsv
            ? BuildRealtimeCsvArchiveRequests()
            : null;
        StopCsvMonitor();
        EndActiveRealtimeCsv();
        SetRealtimeDropdownsInteractable(true);

        string csvToClear = clearCsvOnExit ? ResolveCsvAbsolutePath() : string.Empty;
        bool stopStarted = StopBashProcessAsync(csvToClear, archiveRequests);
        if (!stopStarted)
        {
            ArchiveRealtimeCsvOutputs(archiveRequests);
            if (clearCsvOnExit)
            {
                TryClearCsv("StopStartPipeline");
            }
        }
    }

}
