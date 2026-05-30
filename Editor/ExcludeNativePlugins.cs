using UnityEditor;
using UnityEngine;

/// <summary>
/// Prevents Unity from treating third-party .cpp/.h files as native plugins.
/// DPVO, pybind11, and Pangolin headers/sources live under Assets/ but are
/// Python dependencies — they should never be compiled by Unity.
/// </summary>
public class ExcludeNativePlugins : AssetPostprocessor
{
    private static readonly string[] ExcludePaths =
    {
        "Assets/Imitation/Robot-imitation-learning/third-party/DPVO",
        "Assets/Imitation/Robot-imitation-learning/third-party",
    };

    private static bool ShouldExclude(string assetPath)
    {
        foreach (var prefix in ExcludePaths)
            if (assetPath.StartsWith(prefix))
                return true;
        return false;
    }

    private void OnPostprocessAllAssets(
        string[] importedAssets,
        string[] deletedAssets,
        string[] movedAssets,
        string[] movedFromAssetPaths)
    {
        foreach (var path in importedAssets)
        {
            if (!ShouldExclude(path))
                continue;

            var importer = AssetImporter.GetAtPath(path) as PluginImporter;
            if (importer == null)
                continue;

            bool changed = false;

            // Exclude from all platforms
            if (importer.GetCompatibleWithAnyPlatform())
            {
                importer.SetCompatibleWithAnyPlatform(false);
                changed = true;
            }
            if (importer.GetCompatibleWithEditor())
            {
                importer.SetCompatibleWithEditor(false);
                changed = true;
            }
            if (importer.GetCompatibleWithPlatform(BuildTarget.StandaloneWindows64))
            {
                importer.SetCompatibleWithPlatform(BuildTarget.StandaloneWindows64, false);
                changed = true;
            }

            if (changed)
            {
                importer.SaveAndReimport();
                Debug.Log($"[ExcludeNativePlugins] Excluded: {path}");
            }
        }
    }

    /// <summary>
    /// One-shot cleanup: fix all already-imported files under the exclude paths.
    /// </summary>
    [MenuItem("Tools/Exclude Third-Party Native Plugins")]
    public static void FixAllImported()
    {
        var guids = AssetDatabase.FindAssets("", new[] { "Assets/Imitation/Robot-imitation-learning/third-party" });
        int fixedCount = 0;

        foreach (var guid in guids)
        {
            var path = AssetDatabase.GUIDToAssetPath(guid);
            var importer = AssetImporter.GetAtPath(path) as PluginImporter;
            if (importer == null)
                continue;

            bool changed = false;
            if (importer.GetCompatibleWithAnyPlatform())
            {
                importer.SetCompatibleWithAnyPlatform(false);
                changed = true;
            }
            if (importer.GetCompatibleWithEditor())
            {
                importer.SetCompatibleWithEditor(false);
                changed = true;
            }
            if (importer.GetCompatibleWithPlatform(BuildTarget.StandaloneWindows64))
            {
                importer.SetCompatibleWithPlatform(BuildTarget.StandaloneWindows64, false);
                changed = true;
            }

            if (changed)
            {
                importer.SaveAndReimport();
                fixedCount++;
            }
        }

        AssetDatabase.Refresh();
        Debug.Log($"[ExcludeNativePlugins] Fixed {fixedCount} plugin(s).");
    }
}
