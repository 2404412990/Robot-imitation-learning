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

    private List<RealtimeCsvArchiveRequest> BuildRealtimeCsvArchiveRequests()
    {
        var requests = new List<RealtimeCsvArchiveRequest>();
        if (realtimeRobotStates.Count > 0)
        {
            foreach (RealtimeRobotState state in realtimeRobotStates.Values)
            {
                AddRealtimeCsvArchiveRequest(requests, state?.RobotKey, state?.CsvPath, state?.LastPlaybackFps ?? ReplayCsvUtility.SourceFps);
            }
        }
        else
        {
            List<string> robotKeys = BuildSelectedRobotKeyList();
            if (robotKeys.Count == 0 && !string.IsNullOrWhiteSpace(resolvedCsvPath))
            {
                robotKeys.Add(ResolvePrimarySelectedRobotKey());
            }

            foreach (string robotKey in robotKeys)
            {
                AddRealtimeCsvArchiveRequest(requests, robotKey, ResolveRobotRealtimeCsvPath(robotKey), ReplayCsvUtility.SourceFps);
            }
        }

        return requests;
    }

    private void AddRealtimeCsvArchiveRequest(List<RealtimeCsvArchiveRequest> requests, string robotKey, string sourcePath, float sourceFps)
    {
        if (requests == null)
        {
            return;
        }

        string normalizedRobotKey = TryResolveRobotKeyQuiet(robotKey);
        if (string.IsNullOrWhiteSpace(normalizedRobotKey))
        {
            normalizedRobotKey = string.IsNullOrWhiteSpace(robotKey) ? ResolvePrimarySelectedRobotKey() : robotKey.Trim();
        }

        if (string.IsNullOrWhiteSpace(normalizedRobotKey) || string.IsNullOrWhiteSpace(sourcePath))
        {
            return;
        }

        if (!TryResolveReplayCsvArchiveDirectory(normalizedRobotKey, out string destinationDirectory))
        {
            QueueUnityLog($"[StartInput] Could not resolve archive dataset folder for robot '{normalizedRobotKey}'.", QueuedUnityLogType.Warning);
            return;
        }

        requests.Add(new RealtimeCsvArchiveRequest
        {
            RobotKey = normalizedRobotKey,
            SourcePath = sourcePath,
            DestinationDirectory = destinationDirectory,
            SourceFps = ResolveArchivedSourceFps(sourceFps),
            RealtimeCameraSource = IsRealtimeCameraSource(),
        });
    }

    private float ResolveArchivedSourceFps(float sourceFps)
    {
        return IsRealtimeCameraSource()
            ? ReplayCsvUtility.ClampRealtimeFps(sourceFps)
            : ReplayCsvUtility.SourceFps;
    }

    private bool IsRealtimeCameraSource()
    {
        string value = (videoPath ?? string.Empty).Trim();
        return string.Equals(value, "0", System.StringComparison.OrdinalIgnoreCase) ||
               string.Equals(value, "webcam", System.StringComparison.OrdinalIgnoreCase) ||
               string.Equals(value, "camera", System.StringComparison.OrdinalIgnoreCase);
    }

    private bool TryResolveReplayCsvArchiveDirectory(string robotKey, out string destinationDirectory)
    {
        destinationDirectory = string.Empty;
        string robotFolder = ResolveRobotDatasetFolder(robotKey);
        if (string.IsNullOrWhiteSpace(robotFolder))
        {
            return false;
        }

        var fallbacks = new List<string>
        {
            "Assets/Gewu/Imitation/dataset",
            "Assets/Imitation/dataset",
        };

        if (ImitationDatasetPaths.TryResolveRobotDatasetPath(
                robotFolder,
                "Assets/Gewu/Imitation/dataset",
                fallbacks,
                out string resolved,
                out _))
        {
            destinationDirectory = resolved;
            return true;
        }

        if (ImitationDatasetPaths.TryResolveDatasetRoot(out string datasetRoot, out _))
        {
            destinationDirectory = Path.Combine(datasetRoot, robotFolder);
            return true;
        }

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrWhiteSpace(projectRoot))
        {
            destinationDirectory = Path.Combine(projectRoot, "Assets", "Imitation", "dataset", robotFolder);
            return true;
        }

        return false;
    }

    private static string ResolveRobotDatasetFolder(string robotKey)
    {
        string normalized = TryResolveRobotKeyQuiet(robotKey);
        if (string.IsNullOrWhiteSpace(normalized))
        {
            normalized = (robotKey ?? string.Empty).Trim();
        }

        return RobotCatalog.TryGetDatasetFolder(normalized, out string folder) ? folder : normalized;
    }

    private void ArchiveRealtimeCsvOutputs(IReadOnlyList<RealtimeCsvArchiveRequest> archiveRequests)
    {
        if (archiveRequests == null || archiveRequests.Count == 0)
        {
            return;
        }

        string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        var archivedRobotKeys = new HashSet<string>(System.StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < archiveRequests.Count; i++)
        {
            RealtimeCsvArchiveRequest request = archiveRequests[i];
            if (request == null ||
                string.IsNullOrWhiteSpace(request.RobotKey) ||
                string.IsNullOrWhiteSpace(request.SourcePath) ||
                string.IsNullOrWhiteSpace(request.DestinationDirectory))
            {
                continue;
            }

            try
            {
                if (!File.Exists(request.SourcePath))
                {
                    QueueUnityLog($"[StartInput] Realtime CSV archive skipped; source missing: {request.SourcePath}", QueuedUnityLogType.Warning);
                    continue;
                }

                var sourceInfo = new FileInfo(request.SourcePath);
                if (sourceInfo.Length <= 0)
                {
                    QueueUnityLog($"[StartInput] Realtime CSV archive skipped; source is empty: {request.SourcePath}", QueuedUnityLogType.Warning);
                    continue;
                }

                Directory.CreateDirectory(request.DestinationDirectory);
                string safeRobotName = SanitizeFileNamePart(request.RobotKey);
                string archivePath = CreateUniqueArchivePath(request.DestinationDirectory, $"{timestamp}_{safeRobotName}.csv");
                float archivedCsvFps = ReplayCsvUtility.SourceFps;
                int sourceRows = 0;
                int archivedRows = 0;
                if (request.RealtimeCameraSource &&
                    Mathf.Abs(request.SourceFps - ReplayCsvUtility.SourceFps) > 0.05f &&
                    TryReadNumericCsvRows(request.SourcePath, out List<float[]> sourceRowsData))
                {
                    sourceRows = sourceRowsData.Count;
                    List<float[]> resampledRows = ReplayCsvUtility.ResampleSourceFpsToTargetFps(
                        sourceRowsData,
                        request.SourceFps,
                        ReplayCsvUtility.SourceFps);
                    WriteNumericCsvRows(archivePath, resampledRows);
                    archivedRows = resampledRows.Count;
                }
                else
                {
                    CopyFileAllowingSharedRead(request.SourcePath, archivePath);
                    sourceRows = CountNumericCsvRows(request.SourcePath);
                    archivedRows = sourceRows;
                    archivedCsvFps = request.RealtimeCameraSource
                        ? ReplayCsvUtility.ClampRealtimeFps(request.SourceFps)
                        : ReplayCsvUtility.SourceFps;
                }

                ReplayCsvUtility.WriteReplayMetadata(
                    archivePath,
                    request.RobotKey,
                    archivedCsvFps,
                    request.SourceFps,
                    request.RealtimeCameraSource,
                    sourceRows,
                    archivedRows);
                archivedRobotKeys.Add(request.RobotKey);
                QueueUnityLog(
                    $"[StartInput] Realtime CSV archived: robot={request.RobotKey}, path={archivePath}, " +
                    $"sourceFps={request.SourceFps:F2}, archivedFps={archivedCsvFps:F2}, sourceRows={sourceRows}, archivedRows={archivedRows}");
            }
            catch (System.Exception e)
            {
                QueueUnityLog(
                    $"[StartInput] Failed to archive realtime CSV for robot '{request.RobotKey}' from '{request.SourcePath}': {e.Message}",
                    QueuedUnityLogType.Warning);
            }
        }

        foreach (string robotKey in archivedRobotKeys)
        {
            archivedCsvRefreshQueue.Enqueue(robotKey);
        }
    }

    private static bool TryReadNumericCsvRows(string csvPath, out List<float[]> rows)
    {
        rows = new List<float[]>();
        int expectedColumns = 0;
        using (FileStream fs = new FileStream(csvPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (StreamReader reader = new StreamReader(fs))
        {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                string[] tokens = line.Split(',');
                if (expectedColumns <= 0)
                {
                    expectedColumns = tokens.Length;
                }

                if (tokens.Length != expectedColumns)
                {
                    continue;
                }

                float[] row = new float[expectedColumns];
                bool ok = true;
                for (int i = 0; i < expectedColumns; i++)
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

                if (ok)
                {
                    rows.Add(row);
                }
            }
        }

        return rows.Count > 0;
    }

    private static int CountNumericCsvRows(string csvPath)
    {
        return TryReadNumericCsvRows(csvPath, out List<float[]> rows) ? rows.Count : 0;
    }

    private static void WriteNumericCsvRows(string csvPath, IReadOnlyList<float[]> rows)
    {
        using (var writer = new StreamWriter(csvPath, append: false, Encoding.UTF8))
        {
            if (rows == null)
            {
                return;
            }

            for (int rowIndex = 0; rowIndex < rows.Count; rowIndex++)
            {
                float[] row = rows[rowIndex];
                if (row == null || row.Length == 0)
                {
                    continue;
                }

                for (int col = 0; col < row.Length; col++)
                {
                    if (col > 0)
                    {
                        writer.Write(',');
                    }

                    writer.Write(row[col].ToString("G9", CultureInfo.InvariantCulture));
                }

                writer.WriteLine();
            }
        }
    }

    private static void CopyFileAllowingSharedRead(string sourcePath, string destinationPath)
    {
        using (FileStream source = new FileStream(sourcePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (FileStream destination = new FileStream(destinationPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
        {
            source.CopyTo(destination);
        }
    }

    private static string CreateUniqueArchivePath(string directory, string fileName)
    {
        string baseName = Path.GetFileNameWithoutExtension(fileName);
        string extension = Path.GetExtension(fileName);
        string candidate = Path.Combine(directory, fileName);
        int suffix = 1;
        while (File.Exists(candidate))
        {
            candidate = Path.Combine(directory, $"{baseName}_{suffix}{extension}");
            suffix++;
        }

        return candidate;
    }

    private static string SanitizeFileNamePart(string raw)
    {
        string value = string.IsNullOrWhiteSpace(raw) ? "robot" : raw.Trim();
        foreach (char invalid in Path.GetInvalidFileNameChars())
        {
            value = value.Replace(invalid, '_');
        }

        return value;
    }

    /// <summary>
    /// Resolves the absolute path of the CSV output file from current inspector settings.
    /// Returns empty string when the path cannot be determined.
    /// </summary>
}
