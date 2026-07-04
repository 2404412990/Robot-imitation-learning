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
    private bool StartCsvMonitor()
    {
        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            string baseDir = ResolveWorkingDirectoryPath();
            resolvedOutputRootPath = ResolveOutputRootAbsolutePath(baseDir);
        }

        if (string.IsNullOrWhiteSpace(resolvedOutputRootPath))
        {
            Debug.LogError("[StartInput] OUTPUT_ROOT is invalid; cannot start realtime CSV monitor.");
            return false;
        }

        return StartMultiCsvMonitorForSelectedRobots();
    }
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
        state.LastPlaybackFps = playbackFps;
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
}
