using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
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

public partial class StartInput
{
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

}
