#if UNITY_EDITOR
using System;
using System.IO;
using System.Collections.Generic;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

public sealed class ImitationBuildPostprocessor : IPreprocessBuildWithReport, IPostprocessBuildWithReport
{
    private const string MenuScenePath = "Assets/GewuMenu.unity";
    private const string ImitationScenePath = "Assets/Imitation/G1.unity";

    public int callbackOrder => 1000;

    public void OnPreprocessBuild(BuildReport report)
    {
        EnsureMenuSceneIsBuildEntryPoint();
    }

    public void OnPostprocessBuild(BuildReport report)
    {
        string outputPath = report.summary.outputPath;
        string outputDir = Directory.GetParent(outputPath)?.FullName;
        string exeName = Path.GetFileNameWithoutExtension(outputPath);
        if (string.IsNullOrWhiteSpace(outputDir) || string.IsNullOrWhiteSpace(exeName))
        {
            Debug.LogWarning($"[ImitationBuild] Cannot resolve build output from '{outputPath}'.");
            return;
        }

        string projectRoot = Directory.GetParent(Application.dataPath)?.FullName;
        string sourceRoot = Path.Combine(projectRoot ?? string.Empty, "Assets", "Imitation", "dataset");
        string dataDir = Path.Combine(outputDir, exeName + "_Data");
        string targetRoot = Path.Combine(dataDir, "StreamingAssets", ImitationDatasetPaths.StreamingDatasetFolderName);

        if (!Directory.Exists(sourceRoot))
        {
            Debug.LogWarning($"[ImitationBuild] Dataset source not found: {sourceRoot}");
            return;
        }

        try
        {
            if (Directory.Exists(targetRoot))
            {
                Directory.Delete(targetRoot, recursive: true);
            }

            Directory.CreateDirectory(targetRoot);
            int copied = CopyCsvTree(sourceRoot, targetRoot);
            Debug.Log($"[ImitationBuild] Copied {copied} dataset CSV files to {targetRoot}");
        }
        catch (Exception e)
        {
            Debug.LogError($"[ImitationBuild] Failed to copy dataset CSV files from '{sourceRoot}' to '{targetRoot}': {e}");
        }
    }

    private static int CopyCsvTree(string sourceRoot, string targetRoot)
    {
        int copied = 0;
        foreach (string sourceFile in Directory.GetFiles(sourceRoot, "*.csv", SearchOption.AllDirectories))
        {
            string relativePath = Path.GetRelativePath(sourceRoot, sourceFile);
            string targetFile = Path.Combine(targetRoot, relativePath);
            string targetDir = Path.GetDirectoryName(targetFile);
            if (!string.IsNullOrWhiteSpace(targetDir))
            {
                Directory.CreateDirectory(targetDir);
            }

            File.Copy(sourceFile, targetFile, overwrite: true);
            copied++;
        }

        return copied;
    }

    private static void EnsureMenuSceneIsBuildEntryPoint()
    {
        var scenes = new List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
        bool changed = EnsureSceneAtIndex(scenes, MenuScenePath, 0, enabled: true);
        int imitationIndex = FindSceneIndex(scenes, ImitationScenePath);
        if (imitationIndex < 0)
        {
            scenes.Insert(Mathf.Min(1, scenes.Count), new EditorBuildSettingsScene(ImitationScenePath, true));
            changed = true;
        }
        else if (!scenes[imitationIndex].enabled)
        {
            scenes[imitationIndex].enabled = true;
            changed = true;
        }

        if (changed)
        {
            EditorBuildSettings.scenes = scenes.ToArray();
            Debug.Log($"[ImitationBuild] Build scene order normalized: scene 0 is {MenuScenePath}, imitation scene remains enabled.");
        }
    }

    private static bool EnsureSceneAtIndex(List<EditorBuildSettingsScene> scenes, string path, int index, bool enabled)
    {
        int existingIndex = FindSceneIndex(scenes, path);
        EditorBuildSettingsScene scene = existingIndex >= 0
            ? scenes[existingIndex]
            : new EditorBuildSettingsScene(path, enabled);
        bool wasEnabled = scene.enabled;
        scene.enabled = enabled;

        if (existingIndex >= 0)
        {
            scenes.RemoveAt(existingIndex);
        }

        int clampedIndex = Mathf.Clamp(index, 0, scenes.Count);
        scenes.Insert(clampedIndex, scene);
        return existingIndex != clampedIndex || existingIndex < 0 || wasEnabled != enabled;
    }

    private static int FindSceneIndex(List<EditorBuildSettingsScene> scenes, string path)
    {
        for (int i = 0; i < scenes.Count; i++)
        {
            if (string.Equals(scenes[i].path, path, StringComparison.OrdinalIgnoreCase))
            {
                return i;
            }
        }

        return -1;
    }
}
#endif
