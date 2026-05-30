using System;
using System.Collections.Generic;
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
        "Assets/Imatation/dataset",
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

    void Start()
    {
        // 获取挂载在同一物体上的TMP_Dropdown组件
        dropdown = GetComponent<TMP_Dropdown>();

        // 检查组件是否存在
        if (dropdown == null)
        {
            Debug.LogError("未找到TMP_Dropdown组件，请确保脚本挂载在正确的物体上！");
            return;
        }

        // 开始读取文件并填充下拉列表
        PopulateDropdown();
    }

    // 从指定文件夹读取文件并填充到下拉列表中
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

        // StaticList mode: skip filesystem scanning entirely and just push the
        // hand-authored staticOptions into the dropdown. Used by the RoboList
        // dropdown so it can list arbitrary robot keys (unitree_g1, unitree_h1)
        // without needing real subfolders on disk.
        if (dropdownMode == DropdownMode.StaticList)
        {
            PopulateStaticOptions();
            return;
        }

        // Try primary path first, then each fallback in order.
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
                // Keep whatever the user dragged into the dropdown's Options
                // array in the Inspector. Don't wipe it with a placeholder
                // that won't render (Chinese strings break with default TMP
                // font) and that wipes the user's intended options.
                Debug.Log($"[FileBrowser] Keeping {dropdown.options.Count} manually-authored dropdown options on '{name}'.");
                return;
            }

            SetFallbackOption("(path not found)");
            return;
        }

        // Cache the resolved path so subsequent reads (Search/GetCsvFiles…)
        // use the same folder that succeeded the existence check.
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
                Debug.Log($"[FileBrowser:{name}] StaticList mode with empty staticOptions — keeping manually-authored Inspector options.");
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
        SearchOption searchOption = includeSubfolders ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        string[] files = Directory.GetFiles(folderPath, searchPattern, searchOption);

        // 仅保留 csv，按文件名去重，避免 .csv 与 .csv.meta 导致重复显示
        csvFilePaths.AddRange(
            files
                .Where(filePath => string.Equals(Path.GetExtension(filePath), ".csv", StringComparison.OrdinalIgnoreCase))
                .GroupBy(filePath => Path.GetFileNameWithoutExtension(filePath), StringComparer.OrdinalIgnoreCase)
                .Select(group => group.First())
                .OrderBy(filePath => Path.GetFileNameWithoutExtension(filePath), StringComparer.OrdinalIgnoreCase)
        );

        List<string> fileNames = csvFilePaths
            .Select(filePath => Path.GetFileNameWithoutExtension(filePath))
            .ToList();

        if (fileNames.Count == 0)
        {
            SetFallbackOption("can't find csv files");
            return;
        }

        dropdown.ClearOptions();
        dropdown.AddOptions(fileNames);
        dropdown.RefreshShownValue();

        Debug.Log($"已成功将 {csvFilePaths.Count} 个csv文件添加到下拉列表中。");
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

        Debug.Log($"已成功将 {folderPaths.Count} 个文件夹添加到下拉列表中。");
    }

    private void SetFallbackOption(string text)
    {
        dropdown.ClearOptions();
        dropdown.AddOptions(new List<string> { text });
        dropdown.RefreshShownValue();
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