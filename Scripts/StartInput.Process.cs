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
                Debug.Log("[StartInput] Bash process is already running; skip duplicate launch.");
                return true;

            }

            StopBashProcessBlocking(null);
        }

        string resolvedWorkingDir = ResolveWorkingDirectoryPath();
        if (string.IsNullOrWhiteSpace(resolvedWorkingDir) || !Directory.Exists(resolvedWorkingDir))
        {
            Debug.LogError($"[StartInput] Invalid bash working directory: {resolvedWorkingDir}");
            return false;

        }

        string executable = (bashExecutable ?? string.Empty).Trim();
        string executablePrefixArgs = string.Empty;
        ParseExecutable(executable, out executable, out executablePrefixArgs);

        if (string.IsNullOrWhiteSpace(executable))
        {
            Debug.LogError("[StartInput] bashExecutable is empty; cannot launch pipeline.");
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
            Debug.LogError("[StartInput] Could not resolve current RoboList robot name; check dropdown configuration.");
            return false;

        }

        string resolvedRobot = selectedRobotKeysForLaunch.Count > 0 ? selectedRobotKeysForLaunch[0] : ResolveRobotNameForWham(selectedRobot);
        string resolvedRobots = selectedRobotKeysForLaunch.Count > 0 ? string.Join(",", selectedRobotKeysForLaunch) : resolvedRobot;
        if (string.IsNullOrWhiteSpace(resolvedRobot))
        {
            Debug.LogError($"[StartInput] Unsupported robot for WHAM/GMR: {selectedRobot}");
            return false;

        }

        resolvedOutputRootPath = ResolveOutputRootAbsolutePath(resolvedWorkingDir);
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            Debug.LogError("[StartInput] OUTPUT_ROOT is invalid; cannot launch pipeline.");
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

            Debug.Log($"[StartInput] Started Bash command: {executable} {commandArguments}");
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
            Debug.LogError($"[StartInput] Failed to start Bash command: {e.Message}");
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

    private bool StopBashProcessAsync(string csvToClearAfterStop = null, List<RealtimeCsvArchiveRequest> archiveRequests = null)
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
                ArchiveRealtimeCsvOutputs(archiveRequests);
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

}
