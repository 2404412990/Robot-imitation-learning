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

    [Tooltip("Optional explicit IMimicAgent to drive in replay. When left empty, the agent " +
             "matching the currently selected robot key (from the RoboList dropdown wired into " +
             "StartInput) is looked up via MimicAgentRegistry. If that also fails, the first " +
             "registered agent in the scene is used.")]
    [SerializeField] private MonoBehaviour targetAgentBehaviour;
    [SerializeField] private StartInput startInput;
    [Tooltip("Optional RoboList dropdown used to pick which robot the replay applies to. " +
             "When empty, this Replay script falls back to whichever agent ResolveActiveAgent() " +
             "would pick (typically the first registered robot).")]
    [SerializeField] private TMP_Dropdown roboListDropdown;
    [SerializeField] private FileBrowser roboListFileBrowser;
    [SerializeField] private string roboListObjectName = "RoboList";

    [Header("Replay Behavior")]
    [SerializeField] private bool stopRealtimeTrackingOnReplay = true;

    [Header("Replay Dataset")]
    [Tooltip("Primary folder containing the replay CSV files. If missing, the script tries " +
             "each path in replayDatasetExtraFallbacks before giving up.")]
    [SerializeField] private string replayDatasetRelativePath = "Assets/Gewu/Imitation/dataset";
    [SerializeField] private string replayDatasetFallbackRelativePath = "Assets/Imitation/dataset";
    [Tooltip("Extra fallback dataset paths tried in order after the two above. Lets the script " +
             "keep working when scenes carry stale serialized paths.")]
    [SerializeField] private List<string> replayDatasetExtraFallbacks = new List<string>
    {
        "Assets/Gewu/Imitation/dataset",
        "Assets/Imitation/dataset",
        "Assets/Imatation/dataset",
    };

    private Button replayButton;
    private bool addedRuntimeListener;

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

        IMimicAgent agent = ResolveActiveAgent();
        if (agent == null)
        {
            Debug.LogError("未找到任何已注册的 IMimicAgent，无法执行 replay。" +
                           "\n请确认场景里至少挂有一个实现 IMimicAgent 的机器人脚本。");
            return;
        }

        string selectedCsvName = GetSelectedCsvName();
        if (string.IsNullOrWhiteSpace(selectedCsvName))
        {
            Debug.LogWarning("下拉列表没有可用的 csv 选项，无法执行 replay。");
            return;
        }

        // Resolve both the absolute file path and the legacy index. The path
        // is the primary handle — different agents (G1, H1, ...) interpret
        // MotionId against their own dataset layout, so it is only meaningful
        // as a fallback. The absolute path is universal: each agent's
        // LoadReplayCsvFromPath knows how to parse a flat CSV in its own
        // expected column layout.
        string csvAbsolutePath = ResolveCsvAbsolutePath(selectedCsvName);
        int motionId = ResolveMotionIdByName(selectedCsvName);

        if (string.IsNullOrEmpty(csvAbsolutePath) && motionId < 0)
        {
            Debug.LogWarning($"在 replay 数据集中未找到所选 csv: {selectedCsvName}");
            return;
        }

        if (stopRealtimeTrackingOnReplay && startInput != null)
        {
            startInput.StopStartPipeline();
        }

        // Path A — try LoadReplayCsvFromPath first. This is the only route
        // that works correctly when the active agent's on-disk dataset layout
        // differs from the dropdown's source folder. Example:
        //   - Dropdown lists flat CSVs from Assets/Gewu/Imitation/dataset/.
        //   - H1 normally pulls 3-file folders from StreamingAssets/h1_dataset/.
        // Sending the dropdown's MotionId straight at H1 indexes into the
        // wrong list, producing the "fixed action" bug the user hit. Feeding
        // the absolute CSV path lets H1.LoadReplayCsvFromPath parse the flat
        // CSV directly (it consumes the first 26 columns: 3 pos + 4 quat + 19
        // DOFs) so the picked motion drives H1.
        bool loadedViaPath = false;
        if (!string.IsNullOrEmpty(csvAbsolutePath))
        {
            try
            {
                loadedViaPath = agent.LoadReplayCsvFromPath(csvAbsolutePath, keepProgress: false);
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"[Replay] LoadReplayCsvFromPath threw on {agent.RobotKey}: {e.Message}");
                loadedViaPath = false;
            }
        }

        if (loadedViaPath)
        {
            // External CSV is now in the agent's replay buffers. Mark it so
            // OnEpisodeBegin's on-disk reload branch is skipped (it would
            // overwrite the freshly-loaded buffers otherwise).
            agent.UseExternalReplayData = true;
            agent.ReplayMode = true;
            agent.MotionId = motionId >= 0 ? motionId : 0;
            agent.RequestEndEpisode();

            Debug.Log($"开始 replay (path): robot={agent.RobotKey}, csv={selectedCsvName}, path={csvAbsolutePath}");
            return;
        }

        // Path B — legacy fallback: tell the agent the dropdown index and let
        // its OnEpisodeBegin pull from its own dataset. Works for G1 (whose
        // dataset matches the dropdown source) and as a last-resort for any
        // agent that doesn't yet support LoadReplayCsvFromPath.
        agent.UseExternalReplayData = false;
        agent.ReplayMode = true;
        agent.MotionId = motionId >= 0 ? motionId : 0;
        agent.RequestEndEpisode();

        Debug.Log($"开始 replay (fallback motion_id): robot={agent.RobotKey}, csv={selectedCsvName}, motion_id={motionId}");
    }

    /// <summary>
    /// Resolve the absolute filesystem path of the dropdown-selected CSV by
    /// scanning the same candidate dataset folders that <see cref="ResolveMotionIdByName"/>
    /// uses. Returns an empty string when no matching file exists; the caller
    /// then falls back to the legacy MotionId path so old G1-only behaviour
    /// keeps working.
    /// </summary>
    private string ResolveCsvAbsolutePath(string selectedCsvName)
    {
        if (string.IsNullOrWhiteSpace(selectedCsvName)) return string.Empty;

        string datasetPath = ResolveReplayDatasetPath();
        if (string.IsNullOrEmpty(datasetPath)) return string.Empty;

        string selectedNoExt = Path.GetFileNameWithoutExtension(selectedCsvName).Trim();

        try
        {
            string[] allFiles = Directory.GetFiles(datasetPath);
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
        // 1) Inspector override.
        if (targetAgentBehaviour is IMimicAgent pinned && pinned.AgentGameObject != null)
        {
            return pinned;
        }

        // 2) Match the RoboList selection (if a dropdown is wired up).
        string selectedKey = ResolveSelectedRobotKey();
        if (!string.IsNullOrWhiteSpace(selectedKey) && MimicAgentRegistry.Instance != null)
        {
            IMimicAgent byKey = MimicAgentRegistry.Instance.FindByKey(selectedKey);
            if (byKey != null && byKey.AgentGameObject != null) return byKey;
        }

        // 3) Last-resort: first registered alive agent.
        return MimicAgentRegistry.Instance != null
            ? MimicAgentRegistry.Instance.GetFirstAlive()
            : null;
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
            if (!string.IsNullOrWhiteSpace(folder)) return Path.GetFileName(folder).Trim();
            string csv = roboListFileBrowser.GetSelectedCsvName();
            if (!string.IsNullOrWhiteSpace(csv)) return csv.Trim();
        }

        if (roboListDropdown != null && roboListDropdown.options != null && roboListDropdown.options.Count > 0)
        {
            string text = roboListDropdown.options[roboListDropdown.value].text;
            if (!string.IsNullOrWhiteSpace(text) && !text.StartsWith("(")) return text.Trim();
        }

        return string.Empty;
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

        // targetAgentBehaviour is optional — resolution falls back to
        // MimicAgentRegistry when it's left empty (see ResolveActiveAgent).
        if (targetAgentBehaviour != null && !(targetAgentBehaviour is IMimicAgent))
        {
            Debug.LogWarning("[Replay] targetAgentBehaviour 不是 IMimicAgent 实现，已忽略。");
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
        return selectedText == "(未找到文件)" ? string.Empty : selectedText;
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

    private int ResolveMotionIdByName(string selectedCsvName)
    {
        string datasetPath = ResolveReplayDatasetPath();
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

    private string ResolveReplayDatasetPath()
    {
        var candidatePaths = new List<string>();

        // 1) Inspector-configured primary + secondary paths.
        AddDatasetPathCandidate(candidatePaths, replayDatasetRelativePath);
        AddDatasetPathCandidate(candidatePaths, replayDatasetFallbackRelativePath);

        // 2) Built-in fallback list — catches stale serialized typos.
        if (replayDatasetExtraFallbacks != null)
        {
            foreach (string extra in replayDatasetExtraFallbacks)
                AddDatasetPathCandidate(candidatePaths, extra);
        }

        foreach (string candidatePath in candidatePaths)
        {
            if (Directory.Exists(candidatePath))
            {
                return candidatePath;
            }
        }

        Debug.LogError($"replay 数据集目录不存在，已尝试:\n  {string.Join("\n  ", candidatePaths)}");
        return string.Empty;
    }

    private void AddDatasetPathCandidate(List<string> candidatePaths, string configuredPath)
    {
        if (string.IsNullOrWhiteSpace(configuredPath))
        {
            return;
        }

        string absolutePath = ToAbsoluteProjectPath(configuredPath);
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
            string[] allFiles = Directory.GetFiles(directoryPath);
            foreach (string file in allFiles)
            {
                if (Path.GetExtension(file).ToLower() == ".csv")
                {
                    string fileName = Path.GetFileName(file);
                    csvFiles.Add(Path.Combine(directoryPath, fileName));
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
