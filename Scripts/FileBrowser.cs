using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TMPro;
using UnityEngine;

public class FileBrowser : MonoBehaviour
{
    public enum DropdownMode
    {
        CsvFiles,
        Folders,
        StaticList,
    }

    private TMP_Dropdown dropdown;
    private readonly List<string> csvFilePaths = new List<string>();
    private readonly List<string> folderPaths = new List<string>();
    private readonly List<string> folderDisplayNames = new List<string>();

    public DropdownMode dropdownMode = DropdownMode.CsvFiles;

    [Tooltip("Primary folder to scan. Relative paths resolve from the project root. " +
             "If this path doesn't exist, the script falls back to the entries in " +
             "fallbackFolderPaths in order.")]
    public string folderPath = "Assets/Gewu/Imitation/dataset";

    [Tooltip("Fallback folders, tried in order if the primary folderPath doesn't exist. " +
             "Lets the script keep working when scenes carry stale serialized paths.")]
    public List<string> fallbackFolderPaths = new List<string>
    {
        "Assets/Gewu/Imitation/dataset",
        "Assets/Imitation/dataset",
    };

    public string searchPattern = "*.*";
    public string folderSearchPattern = "*";
    public bool includeSubfolders = false;
    public bool foldersOnlyDirectChildren = true;
    public bool showRelativeFolderPath = true;

    [Tooltip("Hand-picked dropdown entries. Required for DropdownMode.StaticList; " +
             "ignored otherwise. Use this for the RoboList dropdown to enumerate the " +
             "WHAM/GMR robot keys (unitree_g1, unitree_h1, ...) without needing matching " +
             "subfolders on disk.")]
    public List<string> staticOptions = new List<string>();

    [Tooltip("When true and the configured paths all resolve to a missing directory, " +
             "leave the dropdown's existing Inspector-authored options alone instead of " +
             "overwriting them with a '(path not found)' placeholder.")]
    public bool preserveManualOptionsOnFailure = true;

    [Header("CSV Compatibility Filter")]
    [SerializeField] private bool filterCsvByRobot;
    [SerializeField] private string csvRobotFilterKey = "";

    private struct CsvColumnCacheEntry
    {
        public long Length;
        public DateTime LastWriteTimeUtc;
        public int ColumnCount;
    }

    private static readonly Dictionary<string, CsvColumnCacheEntry> CsvColumnCache =
        new Dictionary<string, CsvColumnCacheEntry>(StringComparer.OrdinalIgnoreCase);

    // Store only normalized keys here; aliases are handled by NormalizeRobotKey.
    private static readonly Dictionary<string, int> CsvExpectedColumnsByRobot =
        new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase)
        {
            { "unitree_g1", 36 },
            { "unitree_g1_with_hands", 36 },
            { "unitree_h1", 26 },
            { "unitree_h1_2", 26 },
            { "x02lite", 25 },
            { "openloong", 38 },
        };

    void Start()
    {
        dropdown = GetComponent<TMP_Dropdown>();
        if (dropdown == null)
        {
            Debug.LogError("[FileBrowser] No TMP_Dropdown component on this GameObject.");
            return;
        }

        PopulateDropdown();
    }

    public void PopulateDropdown()
    {
        if (dropdown == null)
        {
            dropdown = GetComponent<TMP_Dropdown>();
            if (dropdown == null)
            {
                Debug.LogError("[FileBrowser] No TMP_Dropdown component on this GameObject.");
                return;
            }
        }

        csvFilePaths.Clear();
        folderPaths.Clear();
        folderDisplayNames.Clear();

        if (dropdownMode == DropdownMode.StaticList)
        {
            PopulateStaticOptions();
            return;
        }

        string resolved = ResolveExistingFolderPath();
        if (string.IsNullOrEmpty(resolved))
        {
            string tried = folderPath +
                (fallbackFolderPaths != null && fallbackFolderPaths.Count > 0
                    ? " | " + string.Join(" | ", fallbackFolderPaths)
                    : string.Empty);
            Debug.LogWarning($"[FileBrowser] Folder path not found. Tried: {tried}");

            if (preserveManualOptionsOnFailure && dropdown.options != null && dropdown.options.Count > 0)
            {
                Debug.Log($"[FileBrowser] Keeping {dropdown.options.Count} manually-authored dropdown options on '{name}'.");
                return;
            }

            SetFallbackOption("(path not found)");
            return;
        }

        folderPath = resolved;

        if (dropdownMode == DropdownMode.Folders)
        {
            PopulateFolderOptions();
        }
        else
        {
            PopulateCsvOptions();
        }
    }

    public void SetCsvRobotFilter(string robotKeyOrLabel)
    {
        csvRobotFilterKey = (robotKeyOrLabel ?? string.Empty).Trim();
        filterCsvByRobot = true;
        PopulateDropdown();
    }

    public void ClearCsvRobotFilter()
    {
        csvRobotFilterKey = string.Empty;
        filterCsvByRobot = false;
        PopulateDropdown();
    }

    private string ResolveExistingFolderPath()
    {
        if (TryResolveAbsolute(folderPath, out string abs) && Directory.Exists(abs)) return abs;

        if (fallbackFolderPaths != null)
        {
            foreach (string candidate in fallbackFolderPaths)
            {
                if (TryResolveAbsolute(candidate, out string fb) && Directory.Exists(fb)) return fb;
            }
        }
        return string.Empty;
    }

    private static bool TryResolveAbsolute(string raw, out string absolute)
    {
        absolute = null;
        if (string.IsNullOrWhiteSpace(raw)) return false;

        string normalized = raw.Trim()
            .Replace('/', Path.DirectorySeparatorChar)
            .Replace('\\', Path.DirectorySeparatorChar);

        try
        {
            if (Path.IsPathRooted(normalized))
            {
                absolute = Path.GetFullPath(normalized);
                return true;
            }

            string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
            if (string.IsNullOrWhiteSpace(projectRoot)) return false;

            absolute = Path.GetFullPath(Path.Combine(projectRoot, normalized));
            return true;
        }
        catch
        {
            absolute = null;
            return false;
        }
    }

    private void PopulateStaticOptions()
    {
        if (staticOptions == null || staticOptions.Count == 0)
        {
            if (preserveManualOptionsOnFailure && dropdown.options != null && dropdown.options.Count > 0)
            {
                Debug.Log($"[FileBrowser:{name}] StaticList mode with empty staticOptions - keeping manually-authored Inspector options.");
                return;
            }
            SetFallbackOption("(no static options)");
            return;
        }

        dropdown.ClearOptions();
        dropdown.AddOptions(new List<string>(staticOptions));
        dropdown.RefreshShownValue();
        Debug.Log($"[FileBrowser:{name}] Loaded {staticOptions.Count} static options.");
    }

    private void PopulateCsvOptions()
    {
        string previousSelection = GetCurrentDropdownText();
        SearchOption searchOption = (includeSubfolders || filterCsvByRobot)
            ? SearchOption.AllDirectories
            : SearchOption.TopDirectoryOnly;
        string[] files = Directory.GetFiles(folderPath, searchPattern, searchOption);

        IEnumerable<string> csvFiles = files
            .Where(filePath => string.Equals(Path.GetExtension(filePath), ".csv", StringComparison.OrdinalIgnoreCase));

        if (filterCsvByRobot)
        {
            csvFiles = csvFiles.Where(IsCompatibleWithActiveRobotFilter);
        }

        csvFilePaths.AddRange(csvFiles
            .GroupBy(GetCsvDisplayName, StringComparer.OrdinalIgnoreCase)
            .Select(PickPreferredCsvForDisplayName)
            .OrderBy(filePath => Path.GetFileNameWithoutExtension(filePath), StringComparer.OrdinalIgnoreCase));

        List<string> fileNames = csvFilePaths
            .Select(GetCsvDisplayName)
            .ToList();

        if (fileNames.Count == 0)
        {
            SetFallbackOption(filterCsvByRobot ? "no compatible csv files" : "can't find csv files");
            return;
        }

        dropdown.ClearOptions();
        dropdown.AddOptions(fileNames);
        RestoreSelection(previousSelection, fileNames);
        dropdown.RefreshShownValue();

        Debug.Log($"[FileBrowser:{name}] Loaded {csvFilePaths.Count} csv files" +
                  (filterCsvByRobot ? $" for robot '{csvRobotFilterKey}'." : "."));
    }

    private bool IsCompatibleWithActiveRobotFilter(string filePath)
    {
        if (!TryResolveExpectedColumns(csvRobotFilterKey, out int expectedColumns))
        {
            return false;
        }

        if (!TryReadCsvColumnCount(filePath, out int columnCount))
        {
            return false;
        }

        return columnCount == expectedColumns;
    }

    private string PickPreferredCsvForDisplayName(IGrouping<string, string> group)
    {
        if (group == null)
        {
            return string.Empty;
        }

        return group
            .OrderByDescending(filePath => IsInRobotNamedFolder(filePath, csvRobotFilterKey) ? 1 : 0)
            .ThenBy(filePath => filePath, StringComparer.OrdinalIgnoreCase)
            .FirstOrDefault() ?? string.Empty;
    }

    private static bool IsInRobotNamedFolder(string filePath, string robotKeyOrLabel)
    {
        if (string.IsNullOrWhiteSpace(filePath) || string.IsNullOrWhiteSpace(robotKeyOrLabel))
        {
            return false;
        }

        string normalizedRobot = NormalizeRobotKey(robotKeyOrLabel);
        if (string.IsNullOrWhiteSpace(normalizedRobot))
        {
            return false;
        }

        string directory = Path.GetDirectoryName(filePath) ?? string.Empty;
        char[] separators = { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar };
        foreach (string segment in directory.Split(separators, StringSplitOptions.RemoveEmptyEntries))
        {
            if (string.Equals(NormalizeRobotKey(segment), normalizedRobot, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static string NormalizeRobotKey(string raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return string.Empty;
        }

        string key = raw.Trim();
        if (string.Equals(key, "G1", StringComparison.OrdinalIgnoreCase))
        {
            return "unitree_g1";
        }
        if (string.Equals(key, "G1H", StringComparison.OrdinalIgnoreCase))
        {
            return "unitree_g1_with_hands";
        }
        if (string.Equals(key, "H1", StringComparison.OrdinalIgnoreCase))
        {
            return "unitree_h1";
        }
        if (string.Equals(key, "H1_2", StringComparison.OrdinalIgnoreCase))
        {
            return "unitree_h1_2";
        }
        if (string.Equals(key, "X02", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(key, "X02Lite", StringComparison.OrdinalIgnoreCase))
        {
            return "x02lite";
        }

        return key;
    }

    private static bool TryResolveExpectedColumns(string robotKeyOrLabel, out int expectedColumns)
    {
        expectedColumns = 0;
        if (string.IsNullOrWhiteSpace(robotKeyOrLabel))
        {
            return false;
        }

        string normalizedKey = NormalizeRobotKey(robotKeyOrLabel);
        return CsvExpectedColumnsByRobot.TryGetValue(normalizedKey, out expectedColumns);
    }

    private static bool TryReadCsvColumnCount(string filePath, out int columnCount)
    {
        columnCount = 0;

        try
        {
            FileInfo info = new FileInfo(filePath);
            string key = info.FullName;
            if (CsvColumnCache.TryGetValue(key, out CsvColumnCacheEntry cached) &&
                cached.Length == info.Length &&
                cached.LastWriteTimeUtc == info.LastWriteTimeUtc)
            {
                columnCount = cached.ColumnCount;
                return columnCount > 0;
            }

            int detected = DetectFirstNumericRowColumnCount(info.FullName);
            CsvColumnCache[key] = new CsvColumnCacheEntry
            {
                Length = info.Length,
                LastWriteTimeUtc = info.LastWriteTimeUtc,
                ColumnCount = detected,
            };

            columnCount = detected;
            return columnCount > 0;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[FileBrowser] Failed to inspect CSV '{filePath}': {e.Message}");
            return false;
        }
    }

    private static int DetectFirstNumericRowColumnCount(string filePath)
    {
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (StreamReader reader = new StreamReader(fs))
        {
            string line;
            int inspectedLines = 0;
            while ((line = reader.ReadLine()) != null && inspectedLines < 50)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                inspectedLines++;
                string[] tokens = line.Split(',');
                if (tokens.Length == 0)
                {
                    continue;
                }

                bool numeric = true;
                for (int i = 0; i < tokens.Length; i++)
                {
                    string token = tokens[i].Trim();
                    if (!float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out _) &&
                        !float.TryParse(token, out _))
                    {
                        numeric = false;
                        break;
                    }
                }

                if (numeric)
                {
                    return tokens.Length;
                }
            }
        }

        return 0;
    }

    private void PopulateFolderOptions()
    {
        SearchOption searchOption = foldersOnlyDirectChildren ? SearchOption.TopDirectoryOnly : SearchOption.AllDirectories;
        string[] directories = Directory.GetDirectories(folderPath, folderSearchPattern, searchOption)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        folderPaths.AddRange(directories);

        List<string> folderNames = folderPaths
            .Select(GetFolderDisplayName)
            .ToList();

        folderDisplayNames.AddRange(folderNames);

        if (folderNames.Count == 0)
        {
            SetFallbackOption("can't find folders");
            return;
        }

        dropdown.ClearOptions();
        dropdown.AddOptions(folderNames);
        dropdown.RefreshShownValue();

        Debug.Log($"[FileBrowser:{name}] Loaded {folderPaths.Count} folders.");
    }

    private void SetFallbackOption(string text)
    {
        csvFilePaths.Clear();
        dropdown.ClearOptions();
        dropdown.AddOptions(new List<string> { text });
        dropdown.value = 0;
        dropdown.RefreshShownValue();
    }

    private string GetCurrentDropdownText()
    {
        if (dropdown == null || dropdown.options == null || dropdown.options.Count == 0)
        {
            return string.Empty;
        }

        int index = Mathf.Clamp(dropdown.value, 0, dropdown.options.Count - 1);
        return dropdown.options[index].text;
    }

    private void RestoreSelection(string previousSelection, List<string> fileNames)
    {
        if (string.IsNullOrWhiteSpace(previousSelection))
        {
            dropdown.value = 0;
            return;
        }

        int index = fileNames.FindIndex(fileName =>
            string.Equals(fileName, previousSelection, StringComparison.OrdinalIgnoreCase));
        dropdown.value = index >= 0 ? index : 0;
    }

    private string GetCsvDisplayName(string absolutePath)
    {
        return Path.GetFileNameWithoutExtension(absolutePath);
    }

    private string GetFolderDisplayName(string absolutePath)
    {
        if (!showRelativeFolderPath)
        {
            return new DirectoryInfo(absolutePath).Name;
        }

        try
        {
            string relative = Path.GetRelativePath(folderPath, absolutePath);
            return string.IsNullOrWhiteSpace(relative) ? new DirectoryInfo(absolutePath).Name : relative;
        }
        catch
        {
            return new DirectoryInfo(absolutePath).Name;
        }
    }

    public string GetSelectedCsvName()
    {
        if (dropdown == null)
        {
            dropdown = GetComponent<TMP_Dropdown>();
        }

        if (dropdown == null || dropdownMode != DropdownMode.CsvFiles || csvFilePaths.Count == 0)
        {
            return string.Empty;
        }

        int index = Mathf.Clamp(dropdown.value, 0, csvFilePaths.Count - 1);
        return Path.GetFileNameWithoutExtension(csvFilePaths[index]);
    }

    public string GetSelectedCsvPath()
    {
        if (dropdown == null)
        {
            dropdown = GetComponent<TMP_Dropdown>();
        }

        if (dropdown == null || dropdownMode != DropdownMode.CsvFiles || csvFilePaths.Count == 0)
        {
            return string.Empty;
        }

        int index = Mathf.Clamp(dropdown.value, 0, csvFilePaths.Count - 1);
        return csvFilePaths[index];
    }

    public List<string> GetCsvFilePaths()
    {
        return new List<string>(csvFilePaths);
    }

    public string GetSelectedFolderPath()
    {
        if (dropdown == null)
        {
            dropdown = GetComponent<TMP_Dropdown>();
        }

        if (dropdown == null || dropdownMode != DropdownMode.Folders || folderPaths.Count == 0)
        {
            return string.Empty;
        }

        int index = Mathf.Clamp(dropdown.value, 0, folderPaths.Count - 1);
        return folderPaths[index];
    }

    public List<string> GetFolderPaths()
    {
        return new List<string>(folderPaths);
    }

    public void PopulateDirectSubfoldersOnly()
    {
        dropdownMode = DropdownMode.Folders;
        foldersOnlyDirectChildren = true;
        PopulateDropdown();
    }
}
