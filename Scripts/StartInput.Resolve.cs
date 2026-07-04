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
                    Debug.Log($"[StartInput] Realtime target agent resolved: {selectedKey} ({byKey.AgentGameObject.name})");
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
            Debug.LogError("[StartInput] Set bashWorkingDirectory in Inspector. Relative paths resolve from Assets/Gewu/Imitation or Assets/Imitation.");
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
                Debug.LogError($"[StartInput] Failed to resolve bashWorkingDirectory: {e.Message}");
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

        Debug.LogWarning("[StartInput] bashWorkingDirectory was not found under any built-in root. Tried:\n  " +
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
            Debug.LogError($"[StartInput] Unsupported robot name: {rawRobotName}. Supported: {string.Join(", ", SupportedRobotNames)}");
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
        return RobotCatalog.TryNormalizeKey(rawRobotName, out string normalizedKey) ? normalizedKey : string.Empty;
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
            Debug.LogError($"[StartInput] Failed to resolve path: {configuredPath} ({e.Message})");
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

        // If the executable path exists as-is, keep it intact so absolute paths with spaces still work.
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
