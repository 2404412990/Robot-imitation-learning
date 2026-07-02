using System.Collections.Generic;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Gewu.Imitation;

public class Replay : MonoBehaviour
{
    [Header("Optional References")]
    [SerializeField] private TMP_Dropdown csvListDropdown;
    [SerializeField] private FileBrowser csvListFileBrowser;

    [Tooltip("Optional explicit IMimicAgent to drive in replay. RoboList selection has priority; " +
             "if the selected robot has no matching registered agent, replay stops instead of " +
             "falling back to another robot.")]
    [SerializeField] private MonoBehaviour targetAgentBehaviour;
    [SerializeField] private StartInput startInput;
    [Tooltip("Optional RoboList dropdown used to pick which robot the replay applies to.")]
    [SerializeField] private TMP_Dropdown roboListDropdown;
    [SerializeField] private FileBrowser roboListFileBrowser;
    [SerializeField] private string roboListObjectName = "RoboList";

    [Header("Replay Behavior")]
    [SerializeField] private bool stopRealtimeTrackingOnReplay = true;

    [Header("Replay Dataset")]
    [Tooltip("Dataset root containing per-robot replay CSV folders. Replay resolves the active robot to a child folder.")]
    [SerializeField] private string replayDatasetRelativePath = "Assets/Gewu/Imitation/dataset";
    [SerializeField] private string replayDatasetFallbackRelativePath = "Assets/Imitation/dataset";
    [Tooltip("Extra dataset roots tried in order after the two above.")]
    [SerializeField] private List<string> replayDatasetExtraFallbacks = new List<string>
    {
        "Assets/Gewu/Imitation/dataset",
        "Assets/Imitation/dataset",
    };

    private Button replayButton;
    private bool addedRuntimeListener;

    private sealed class ReplayJob
    {
        public IMimicAgent Agent;
        public string RobotKey;
        public string CsvPath;
        public string CsvName;
        public int MotionId;
    }

    private static readonly Dictionary<string, string> RobotAliases =
        new Dictionary<string, string>(System.StringComparer.OrdinalIgnoreCase)
        {
            { "G1", "unitree_g1" },
            { "G1H", "unitree_g1_with_hands" },
            { "H1", "unitree_h1" },
            { "H1_2", "unitree_h1_2" },
            { "X02", "x02lite" },
            { "X02Lite", "x02lite" },
            { "OpenLoong", "openloong" },
        };

    private static readonly Dictionary<string, string> RobotDatasetFolders =
        new Dictionary<string, string>(System.StringComparer.OrdinalIgnoreCase)
        {
            { "unitree_g1", "unitree_g1" },
            { "unitree_g1_with_hands", "unitree_g1" },
            { "unitree_h1", "unitree_h1" },
            { "unitree_h1_2", "unitree_h1" },
            { "x02lite", "x02lite" },
            { "openloong", "openloong" },
        };

    void Awake()
    {
        replayButton = GetComponent<Button>();
        if (replayButton != null && !HasPersistentReplayHandler(replayButton))
        {
            replayButton.onClick.AddListener(OnReplayButtonClicked);
            addedRuntimeListener = true;
        }
    }

    void OnDestroy()
    {
        if (replayButton != null && addedRuntimeListener)
        {
            replayButton.onClick.RemoveListener(OnReplayButtonClicked);
        }
    }

    public void OnReplayButtonClicked()
    {
        ResolveReferences();

        if (TryReplaySelectedRobots())
        {
            return;
        }

        IMimicAgent agent = ResolveActiveAgent();
        if (agent == null)
        {
            Debug.LogError("[Replay] No registered IMimicAgent found. Add a robot agent implementing IMimicAgent to the scene.");
            return;
        }

        string selectedCsvName = GetSelectedCsvName();
        if (string.IsNullOrWhiteSpace(selectedCsvName))
        {
            Debug.LogWarning("[Replay] No CSV option is selected.");
            return;
        }

        string csvAbsolutePath = ResolveCsvAbsolutePath(selectedCsvName, agent.RobotKey);
        int motionId = ResolveMotionIdByName(selectedCsvName, agent.RobotKey);

        if (string.IsNullOrEmpty(csvAbsolutePath))
        {
            Debug.LogWarning($"[Replay] Selected CSV path was not resolved: {selectedCsvName}. Replay will not fall back to MotionId.");
            return;
        }

        if (stopRealtimeTrackingOnReplay && startInput != null)
        {
            startInput.StopStartPipeline();
        }

        // LoadReplayCsvFromPath is the only supported route. MotionId is kept
        // only for display/compat bookkeeping and is resolved inside the active
        // robot's own dataset folder, never from the legacy dataset root.
        bool loadedViaPath;
        try
        {
            loadedViaPath = agent.LoadReplayCsvFromPath(csvAbsolutePath, keepProgress: false);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"[Replay] LoadReplayCsvFromPath threw on {agent.RobotKey}: {e.Message}");
            loadedViaPath = false;
        }

        if (!loadedViaPath)
        {
            Debug.LogWarning($"[Replay] Failed to load selected CSV for robot={agent.RobotKey}: {csvAbsolutePath}. Replay will not fall back to MotionId.");
            return;
        }

        // External CSV is now in the agent's replay buffers. Mark it so
        // OnEpisodeBegin's on-disk reload branch is skipped (it would
        // overwrite the freshly-loaded buffers otherwise).
        agent.UseExternalReplayData = true;
        agent.ReplayMode = true;
        agent.MotionId = motionId >= 0 ? motionId : 0;
        agent.RequestEndEpisode();

        Debug.Log($"[Replay] Started replay from path: robot={agent.RobotKey}, csv={selectedCsvName}, path={csvAbsolutePath}");
    }

    private bool TryReplaySelectedRobots()
    {
        if (startInput == null || MimicAgentRegistry.Instance == null)
        {
            return false;
        }

        IReadOnlyList<string> selectedRobotKeys = startInput.GetSelectedRobotKeys();
        if (selectedRobotKeys == null || selectedRobotKeys.Count == 0)
        {
            return false;
        }

        var replayJobs = new List<ReplayJob>();
        for (int i = 0; i < selectedRobotKeys.Count; i++)
        {
            string robotKey = selectedRobotKeys[i];
            if (!startInput.TryGetSelectedCsvForRobot(robotKey, out string csvPath, out string csvName))
            {
                Debug.LogWarning($"[Replay] No CSV selected for checked robot '{robotKey}'.");
                continue;
            }

            IMimicAgent agent = MimicAgentRegistry.Instance.FindByKey(robotKey);
            if (agent == null || agent.AgentGameObject == null)
            {
                Debug.LogWarning($"[Replay] No registered IMimicAgent for checked robot '{robotKey}'.");
                continue;
            }

            replayJobs.Add(new ReplayJob
            {
                Agent = agent,
                RobotKey = robotKey,
                CsvPath = csvPath,
                CsvName = csvName,
                MotionId = ResolveMotionIdByName(csvName, robotKey)
            });
        }

        if (replayJobs.Count == 0)
        {
            return false;
        }

        if (stopRealtimeTrackingOnReplay)
        {
            startInput.StopStartPipeline();
        }

        int loadedCount = 0;
        foreach (var job in replayJobs)
        {
            bool loadedViaPath;
            try
            {
                loadedViaPath = job.Agent.LoadReplayCsvFromPath(job.CsvPath, keepProgress: false);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[Replay] LoadReplayCsvFromPath threw on {job.RobotKey}: {e.Message}");
                loadedViaPath = false;
            }

            if (!loadedViaPath)
            {
                Debug.LogWarning($"[Replay] Failed to load selected CSV for robot={job.RobotKey}: {job.CsvPath}.");
                continue;
            }

            job.Agent.UseExternalReplayData = true;
            job.Agent.ReplayMode = true;
            job.Agent.MotionId = job.MotionId >= 0 ? job.MotionId : 0;
            job.Agent.RequestEndEpisode();
            loadedCount++;

            Debug.Log($"[Replay] Started replay from path: robot={job.RobotKey}, csv={job.CsvName}, path={job.CsvPath}");
        }

        return loadedCount > 0;
    }

    /// <summary>
    /// Resolve the absolute filesystem path of the dropdown-selected CSV by
    /// scanning the same candidate dataset folders that <see cref="ResolveMotionIdByName"/>
    /// uses. Returns an empty string when no matching file exists.
    /// </summary>
    private string ResolveCsvAbsolutePath(string selectedCsvName, string robotKey)
    {
        if (string.IsNullOrWhiteSpace(selectedCsvName)) return string.Empty;

        if (csvListFileBrowser != null)
        {
            string selectedPath = csvListFileBrowser.GetSelectedCsvPath();
            if (!string.IsNullOrWhiteSpace(selectedPath) && File.Exists(selectedPath))
            {
                return selectedPath;
            }
        }

        string datasetPath = ResolveReplayDatasetPath(robotKey);
        if (string.IsNullOrEmpty(datasetPath)) return string.Empty;

        string selectedNoExt = Path.GetFileNameWithoutExtension(selectedCsvName).Trim();

        try
        {
            string[] allFiles = Directory.GetFiles(datasetPath, "*.csv", SearchOption.AllDirectories);
            foreach (string file in allFiles)
            {
                if (Path.GetExtension(file).ToLower() != ".csv") continue;
                string candidateNoExt = Path.GetFileNameWithoutExtension(file);
                if (string.Equals(candidateNoExt, selectedNoExt, System.StringComparison.OrdinalIgnoreCase))
                {
                    return file;
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("[Replay] Error scanning dataset path for CSV match: " + e.Message);
        }

        return string.Empty;
    }

    private IMimicAgent ResolveActiveAgent()
    {
        string selectedKey = ResolveSelectedRobotKey();

        if (!string.IsNullOrWhiteSpace(selectedKey) && MimicAgentRegistry.Instance != null)
        {
            IMimicAgent byKey = MimicAgentRegistry.Instance.FindByKey(selectedKey);
            if (byKey != null && byKey.AgentGameObject != null) return byKey;

            Debug.LogError($"[Replay] No IMimicAgent registered for selected robot '{selectedKey}'. Replay will not fall back to another robot.");
            return null;
        }

        if (targetAgentBehaviour is IMimicAgent pinned && pinned.AgentGameObject != null)
        {
            return pinned;
        }

        Debug.LogError("[Replay] No robot is selected in RoboList and no targetAgentBehaviour is assigned.");
        return null;
    }

    private string ResolveSelectedRobotKey()
    {
        if (roboListFileBrowser == null && roboListDropdown == null)
        {
            GameObject roboListObject = GameObject.Find(roboListObjectName);
            if (roboListObject != null)
            {
                if (roboListFileBrowser == null) roboListFileBrowser = roboListObject.GetComponent<FileBrowser>();
                if (roboListDropdown == null) roboListDropdown = roboListObject.GetComponent<TMP_Dropdown>();
            }
        }

        if (roboListFileBrowser != null)
        {
            string folder = roboListFileBrowser.GetSelectedFolderPath();
            if (!string.IsNullOrWhiteSpace(folder)) return NormalizeRobotKey(Path.GetFileName(folder).Trim());
            string csv = roboListFileBrowser.GetSelectedCsvName();
            if (!string.IsNullOrWhiteSpace(csv)) return NormalizeRobotKey(csv.Trim());
        }

        if (roboListDropdown != null && roboListDropdown.options != null && roboListDropdown.options.Count > 0)
        {
            string text = roboListDropdown.options[roboListDropdown.value].text;
            if (!string.IsNullOrWhiteSpace(text) && !text.StartsWith("(")) return NormalizeRobotKey(text.Trim());
        }

        return string.Empty;
    }

    private static string NormalizeRobotKey(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return string.Empty;
        }

        string trimmed = raw.Trim();
        return RobotAliases.TryGetValue(trimmed, out string alias) ? alias : trimmed;
    }

    private void ResolveReferences()
    {
        if (csvListFileBrowser == null)
        {
            var csvListObject = GameObject.Find("CsvList");
            if (csvListObject != null)
            {
                csvListFileBrowser = csvListObject.GetComponent<FileBrowser>();
                if (csvListDropdown == null)
                {
                    csvListDropdown = csvListObject.GetComponent<TMP_Dropdown>();
                }
            }
        }

        if (csvListDropdown == null && csvListFileBrowser != null)
        {
            csvListDropdown = csvListFileBrowser.GetComponent<TMP_Dropdown>();
        }

        if (csvListDropdown == null)
        {
            csvListDropdown = FindObjectOfType<TMP_Dropdown>();
        }

        // targetAgentBehaviour is optional; resolution falls back to
        // MimicAgentRegistry when it's left empty (see ResolveActiveAgent).
        if (targetAgentBehaviour != null && !(targetAgentBehaviour is IMimicAgent))
        {
            Debug.LogWarning("[Replay] targetAgentBehaviour does not implement IMimicAgent and will be ignored.");
            targetAgentBehaviour = null;
        }

        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>();
        }
    }

    private string GetSelectedCsvName()
    {
        if (csvListFileBrowser != null)
        {
            string nameFromFileBrowser = csvListFileBrowser.GetSelectedCsvName();
            if (!string.IsNullOrWhiteSpace(nameFromFileBrowser))
            {
                return nameFromFileBrowser;
            }
        }

        if (csvListDropdown == null || csvListDropdown.options == null || csvListDropdown.options.Count == 0)
        {
            return string.Empty;
        }

        string selectedText = csvListDropdown.options[csvListDropdown.value].text;
        return IsMissingFileOption(selectedText) ? string.Empty : selectedText;
    }

    private static bool IsMissingFileOption(string selectedText)
    {
        if (string.IsNullOrWhiteSpace(selectedText))
        {
            return true;
        }

        string normalized = selectedText.Trim();
        return string.Equals(normalized, "(No files found)", System.StringComparison.OrdinalIgnoreCase) ||
               string.Equals(normalized, "(No CSV files found)", System.StringComparison.OrdinalIgnoreCase) ||
               (normalized.StartsWith("(") &&
                normalized.EndsWith(")") &&
                normalized.IndexOf(".csv", System.StringComparison.OrdinalIgnoreCase) < 0);
    }

    private bool HasPersistentReplayHandler(Button button)
    {
        int eventCount = button.onClick.GetPersistentEventCount();
        for (int i = 0; i < eventCount; i++)
        {
            if (button.onClick.GetPersistentTarget(i) == this &&
                button.onClick.GetPersistentMethodName(i) == nameof(OnReplayButtonClicked))
            {
                return true;
            }
        }

        return false;
    }

    private int ResolveMotionIdByName(string selectedCsvName, string robotKey)
    {
        string datasetPath = ResolveReplayDatasetPath(robotKey);
        if (string.IsNullOrEmpty(datasetPath))
        {
            return -1;
        }

        List<string> csvFileNames = GetCsvFileNames(datasetPath);
        string selectedNoExt = Path.GetFileNameWithoutExtension(selectedCsvName).Trim();

        for (int i = 0; i < csvFileNames.Count; i++)
        {
            string candidateNoExt = Path.GetFileNameWithoutExtension(csvFileNames[i]);
            if (string.Equals(candidateNoExt, selectedNoExt, System.StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        return -1;
    }

    private string ResolveReplayDatasetPath(string robotKey)
    {
        // 1) Inspector-configured primary + secondary paths.
        string robotFolder = ResolveRobotDatasetFolder(robotKey);
        if (string.IsNullOrWhiteSpace(robotFolder))
        {
            Debug.LogWarning($"[Replay] Unknown robot key '{robotKey}', cannot resolve a robot-specific replay dataset.");
            return string.Empty;
        }

        var fallbackPaths = new List<string>();
        if (!string.IsNullOrWhiteSpace(replayDatasetFallbackRelativePath))
        {
            fallbackPaths.Add(replayDatasetFallbackRelativePath);
        }

        // 2) Built-in fallback list catches stale serialized typos.
        if (replayDatasetExtraFallbacks != null)
        {
            foreach (string extra in replayDatasetExtraFallbacks)
                fallbackPaths.Add(extra);
        }

        if (ImitationDatasetPaths.TryResolveRobotDatasetPath(robotFolder, replayDatasetRelativePath, fallbackPaths, out string resolved, out string tried))
        {
            return resolved;
        }

        Debug.LogError($"Replay dataset folder not found for robot '{robotKey}'. Tried:\n  {tried.Replace(" | ", "\n  ")}");
        return string.Empty;

    }

    private static string ResolveRobotDatasetFolder(string robotKey)
    {
        string normalized = NormalizeRobotKey(robotKey);
        return RobotDatasetFolders.TryGetValue(normalized, out string folder) ? folder : string.Empty;
    }

    private void AddRobotDatasetCandidate(List<string> candidatePaths, string configuredPath, string robotFolder)
    {
        if (string.IsNullOrWhiteSpace(configuredPath) || string.IsNullOrWhiteSpace(robotFolder))
        {
            return;
        }

        string absoluteRoot = ToAbsoluteProjectPath(configuredPath);
        if (string.IsNullOrWhiteSpace(absoluteRoot))
        {
            return;
        }

        string absolutePath = string.Equals(Path.GetFileName(absoluteRoot), robotFolder, System.StringComparison.OrdinalIgnoreCase)
            ? absoluteRoot
            : Path.Combine(absoluteRoot, robotFolder);

        if (!string.IsNullOrWhiteSpace(absolutePath) && !candidatePaths.Contains(absolutePath))
        {
            candidatePaths.Add(absolutePath);
        }
    }

    private string ToAbsoluteProjectPath(string path)
    {
        string normalized = path.Replace('/', Path.DirectorySeparatorChar).Replace('\\', Path.DirectorySeparatorChar);
        if (Path.IsPathRooted(normalized))
        {
            return normalized;
        }

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        if (string.IsNullOrWhiteSpace(projectRoot))
        {
            return string.Empty;
        }

        return Path.GetFullPath(Path.Combine(projectRoot, normalized));
    }

    private List<string> GetCsvFileNames(string directoryPath)
    {
        List<string> csvFiles = new List<string>();

        try
        {
            string[] allFiles = Directory.GetFiles(directoryPath, "*.csv", SearchOption.AllDirectories);
            foreach (string file in allFiles)
            {
                if (Path.GetExtension(file).ToLower() == ".csv")
                {
                    csvFiles.Add(file);
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError("Error accessing replay csv files: " + e.Message);
        }

        return csvFiles;
    }
}
