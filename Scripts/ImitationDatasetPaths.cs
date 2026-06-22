using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public static class ImitationDatasetPaths
{
    public const string StreamingDatasetFolderName = "ImitationDataset";

    public static bool TryResolveDatasetRoot(out string resolved, out string tried)
    {
        return TryResolveDatasetPath(null, null, out resolved, out tried);
    }

    public static bool TryResolveDatasetPath(
        string configuredPath,
        IEnumerable<string> fallbackPaths,
        out string resolved,
        out string tried)
    {
        var candidates = new List<string>();
        AddConfiguredCandidates(candidates, configuredPath);

        if (fallbackPaths != null)
        {
            foreach (string fallback in fallbackPaths)
            {
                AddConfiguredCandidates(candidates, fallback);
            }
        }

        AddRuntimeDatasetCandidates(candidates);

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var triedPaths = new List<string>();
        foreach (string candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            string absolute;
            try
            {
                absolute = Path.GetFullPath(candidate);
            }
            catch
            {
                continue;
            }

            if (!seen.Add(absolute))
            {
                continue;
            }

            triedPaths.Add(absolute);
            if (Directory.Exists(absolute))
            {
                resolved = absolute;
                tried = string.Join(" | ", triedPaths);
                return true;
            }
        }

        resolved = string.Empty;
        tried = string.Join(" | ", triedPaths);
        return false;
    }

    public static bool TryResolveRobotDatasetPath(
        string robotFolder,
        string configuredPath,
        IEnumerable<string> fallbackPaths,
        out string resolved,
        out string tried)
    {
        var candidates = new List<string>();

        AddRobotConfiguredCandidate(candidates, robotFolder, configuredPath);
        if (fallbackPaths != null)
        {
            foreach (string fallback in fallbackPaths)
            {
                AddRobotConfiguredCandidate(candidates, robotFolder, fallback);
            }
        }

        AddRuntimeRobotDatasetCandidates(candidates, robotFolder);

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var triedPaths = new List<string>();
        foreach (string candidate in candidates)
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            string absolute;
            try
            {
                absolute = Path.GetFullPath(candidate);
            }
            catch
            {
                continue;
            }

            if (!seen.Add(absolute))
            {
                continue;
            }

            triedPaths.Add(absolute);
            if (Directory.Exists(absolute))
            {
                resolved = absolute;
                tried = string.Join(" | ", triedPaths);
                return true;
            }
        }

        resolved = string.Empty;
        tried = string.Join(" | ", triedPaths);
        return false;
    }

    private static void AddConfiguredCandidates(List<string> candidates, string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return;
        }

        string normalized = raw.Trim()
            .Replace('/', Path.DirectorySeparatorChar)
            .Replace('\\', Path.DirectorySeparatorChar);

        if (Path.IsPathRooted(normalized))
        {
            candidates.Add(normalized);
            return;
        }

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrWhiteSpace(projectRoot))
        {
            candidates.Add(Path.Combine(projectRoot, normalized));
        }
    }

    private static void AddRobotConfiguredCandidate(List<string> candidates, string robotFolder, string raw)
    {
        if (string.IsNullOrWhiteSpace(raw) || string.IsNullOrWhiteSpace(robotFolder))
        {
            return;
        }

        string normalized = raw.Trim()
            .Replace('/', Path.DirectorySeparatorChar)
            .Replace('\\', Path.DirectorySeparatorChar);

        string absolute;
        if (Path.IsPathRooted(normalized))
        {
            absolute = normalized;
        }
        else
        {
            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
            if (string.IsNullOrWhiteSpace(projectRoot))
            {
                return;
            }

            absolute = Path.Combine(projectRoot, normalized);
        }

        string folderName = Path.GetFileName(absolute.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
        candidates.Add(string.Equals(folderName, robotFolder, StringComparison.OrdinalIgnoreCase)
            ? absolute
            : Path.Combine(absolute, robotFolder));
    }

    private static void AddRuntimeDatasetCandidates(List<string> candidates)
    {
        if (!string.IsNullOrWhiteSpace(Application.streamingAssetsPath))
        {
            candidates.Add(Path.Combine(Application.streamingAssetsPath, StreamingDatasetFolderName));
        }

        string dataParent = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrWhiteSpace(dataParent))
        {
            candidates.Add(Path.Combine(dataParent, "dataset"));
            candidates.Add(Path.Combine(dataParent, StreamingDatasetFolderName));
            candidates.Add(Path.Combine(dataParent, "StreamingAssets", StreamingDatasetFolderName));
        }

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (!string.IsNullOrWhiteSpace(projectRoot))
        {
            candidates.Add(Path.Combine(projectRoot, "Assets", "Imitation", "dataset"));
            candidates.Add(Path.Combine(projectRoot, "Assets", "Gewu", "Imitation", "dataset"));
        }
    }

    private static void AddRuntimeRobotDatasetCandidates(List<string> candidates, string robotFolder)
    {
        if (string.IsNullOrWhiteSpace(robotFolder))
        {
            return;
        }

        var roots = new List<string>();
        AddRuntimeDatasetCandidates(roots);
        foreach (string root in roots)
        {
            candidates.Add(Path.Combine(root, robotFolder));
        }
    }
}
