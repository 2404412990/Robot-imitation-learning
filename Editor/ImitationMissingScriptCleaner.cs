using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class ImitationMissingScriptCleaner
{
    private static readonly string[] ScenePaths =
    {
        "Assets/Imitation/G1.unity",
        "Assets/Imitation/G1Replay.unity",
    };

    private static bool cleanupQueued;

    static ImitationMissingScriptCleaner()
    {
        EditorSceneManager.sceneOpened += (_, __) => QueueOpenSceneCleanup("scene opened");
        EditorApplication.playModeStateChanged += state =>
        {
            if (state == PlayModeStateChange.ExitingEditMode)
            {
                int removed = CleanupOpenScenes("[ImitationMissingScriptCleaner] before Play Mode", saveDirtyScenes: true);
                removed += CleanupPrefabsUnder("Assets/Imitation", "[ImitationMissingScriptCleaner] before Play Mode");
                if (removed > 0)
                {
                    AssetDatabase.SaveAssets();
                    AssetDatabase.Refresh();
                }
            }
        };
    }

    [MenuItem("Imitation/Cleanup Missing Scripts")]
    public static void CleanupMissingScripts()
    {
        if (EditorApplication.isPlayingOrWillChangePlaymode)
        {
            Debug.LogWarning("[ImitationMissingScriptCleaner] Stop Play Mode before cleaning missing scripts.");
            return;
        }

        int removed = CleanupConfiguredScenes("[ImitationMissingScriptCleaner] manual cleanup");
        removed += CleanupPrefabsUnder("Assets/Imitation", "[ImitationMissingScriptCleaner] manual cleanup");

        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();
        Debug.Log($"[ImitationMissingScriptCleaner] Finished. Removed {removed} missing script component(s).");
    }

    public static int CleanupOpenScenes(string reason, bool saveDirtyScenes)
    {
        if (EditorApplication.isPlaying)
        {
            return 0;
        }

        int removed = 0;
        for (int i = 0; i < SceneManager.sceneCount; i++)
        {
            Scene scene = SceneManager.GetSceneAt(i);
            if (!scene.IsValid() || !scene.isLoaded || !IsImitationScene(scene.path))
            {
                continue;
            }

            int sceneRemoved = CleanupScene(scene, scene.path);
            if (sceneRemoved <= 0)
            {
                continue;
            }

            removed += sceneRemoved;
            EditorSceneManager.MarkSceneDirty(scene);
            if (saveDirtyScenes && !string.IsNullOrWhiteSpace(scene.path))
            {
                EditorSceneManager.SaveScene(scene);
            }

            Debug.Log($"{reason}: removed {sceneRemoved} missing script component(s) from {scene.path}");
        }

        return removed;
    }

    public static int CleanupConfiguredScenes(string reason)
    {
        if (EditorApplication.isPlaying)
        {
            return 0;
        }

        string originalScene = SceneManager.GetActiveScene().path;
        int removed = 0;

        foreach (string scenePath in ScenePaths)
        {
            if (!File.Exists(scenePath))
            {
                Debug.LogWarning($"[ImitationMissingScriptCleaner] Scene not found: {scenePath}");
                continue;
            }

            Scene scene = EditorSceneManager.OpenScene(scenePath, OpenSceneMode.Single);
            int sceneRemoved = CleanupScene(scene, scenePath);
            if (sceneRemoved > 0)
            {
                removed += sceneRemoved;
                EditorSceneManager.MarkSceneDirty(scene);
                EditorSceneManager.SaveScene(scene);
                Debug.Log($"{reason}: removed {sceneRemoved} missing script component(s) from {scenePath}");
            }
        }

        if (!string.IsNullOrWhiteSpace(originalScene) && File.Exists(originalScene))
        {
            EditorSceneManager.OpenScene(originalScene, OpenSceneMode.Single);
        }

        return removed;
    }

    private static void QueueOpenSceneCleanup(string reason)
    {
        if (cleanupQueued || EditorApplication.isPlayingOrWillChangePlaymode)
        {
            return;
        }

        cleanupQueued = true;
        EditorApplication.delayCall += () =>
        {
            cleanupQueued = false;
            CleanupOpenScenes($"[ImitationMissingScriptCleaner] {reason}", saveDirtyScenes: false);
        };
    }

    private static int CleanupScene(Scene scene, string assetPath)
    {
        int removed = 0;
        foreach (GameObject root in scene.GetRootGameObjects())
        {
            removed += CleanupHierarchy(root, assetPath);
        }

        return removed;
    }

    public static int CleanupPrefabsUnder(string rootFolder, string reason)
    {
        int removed = 0;
        string[] prefabGuids = AssetDatabase.FindAssets("t:Prefab", new[] { rootFolder });
        foreach (string guid in prefabGuids)
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }
            if (ShouldSkipPrefabCleanupPath(path))
            {
                continue;
            }

            GameObject prefabRoot = PrefabUtility.LoadPrefabContents(path);
            if (prefabRoot == null)
            {
                Debug.LogWarning($"[ImitationMissingScriptCleaner] Could not load prefab: {path}");
                continue;
            }

            try
            {
                int prefabRemoved = CleanupHierarchy(prefabRoot, path);
                if (prefabRemoved > 0)
                {
                    removed += prefabRemoved;
                    PrefabUtility.SaveAsPrefabAsset(prefabRoot, path);
                    Debug.Log($"{reason}: removed {prefabRemoved} missing script component(s) from prefab {path}");
                }
            }
            finally
            {
                PrefabUtility.UnloadPrefabContents(prefabRoot);
            }
        }

        return removed;
    }

    private static int CleanupHierarchy(GameObject root, string assetPath)
    {
        int removed = 0;
        foreach (Transform transform in root.GetComponentsInChildren<Transform>(true))
        {
            int missingCount = GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(transform.gameObject);
            if (missingCount <= 0)
            {
                continue;
            }

            int removedNow = GameObjectUtility.RemoveMonoBehavioursWithMissingScript(transform.gameObject);
            if (removedNow <= 0)
            {
                continue;
            }

            removed += removedNow;
            Debug.Log(
                $"[ImitationMissingScriptCleaner] Removed {removedNow}/{missingCount} missing script component(s) at " +
                $"{assetPath}:{GetHierarchyPath(transform)}");
        }

        return removed;
    }

    private static string GetHierarchyPath(Transform transform)
    {
        if (transform == null)
        {
            return "<null>";
        }

        string path = transform.name;
        Transform current = transform.parent;
        while (current != null)
        {
            path = current.name + "/" + path;
            current = current.parent;
        }

        return path;
    }

    private static bool IsImitationScene(string scenePath)
    {
        if (string.IsNullOrWhiteSpace(scenePath))
        {
            return false;
        }

        foreach (string knownScene in ScenePaths)
        {
            if (string.Equals(scenePath.Replace('\\', '/'), knownScene, System.StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    private static bool ShouldSkipPrefabCleanupPath(string assetPath)
    {
        string normalized = assetPath.Replace('\\', '/');
        return normalized.StartsWith("Assets/Imitation/Robot-imitation-learning/", StringComparison.OrdinalIgnoreCase) ||
               normalized.StartsWith("Assets/Imitation/backup/", StringComparison.OrdinalIgnoreCase) ||
               normalized.IndexOf("/third-party/", StringComparison.OrdinalIgnoreCase) >= 0 ||
               normalized.IndexOf("/Library/", StringComparison.OrdinalIgnoreCase) >= 0;
    }
}

public sealed class ImitationMissingScriptBuildCleaner : IPreprocessBuildWithReport
{
    public int callbackOrder => -1000;

    public void OnPreprocessBuild(BuildReport report)
    {
        ImitationMissingScriptCleaner.CleanupOpenScenes("[ImitationMissingScriptCleaner] before Build", saveDirtyScenes: true);
        ImitationMissingScriptCleaner.CleanupConfiguredScenes("[ImitationMissingScriptCleaner] before Build");
        ImitationMissingScriptCleaner.CleanupPrefabsUnder("Assets/Imitation", "[ImitationMissingScriptCleaner] before Build");
    }
}
