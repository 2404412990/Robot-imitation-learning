#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

[InitializeOnLoad]
public static class UnifiedBlurHudSetup
{
    private const string RenderingFolder = "Assets/Imitation/Rendering";
    private const string ResourcesFolder = "Assets/Imitation/Resources";
    private const string PipelineAssetPath = RenderingFolder + "/RetargetingUniversalPipeline.asset";
    private const string RendererAssetPath = RenderingFolder + "/RetargetingUniversalRenderer.asset";
    private const string HudMaterialPath = ResourcesFolder + "/RetargetingUniversalBlur.mat";
    private const string BuiltinHudMaterialPath = ResourcesFolder + "/RetargetingBuiltinUIBlur.mat";

    static UnifiedBlurHudSetup()
    {
        EditorApplication.delayCall += Configure;
    }

    [MenuItem("Imitation/Setup Unified Blur HUD")]
    public static void Configure()
    {
        try
        {
            EnsureFolders();
            Material blurMaterial = EnsureHudMaterial();
            Material builtinMaterial = EnsureBuiltinHudMaterial();
            DisableAutoCreatedPipelineIfActive();
            UniversalRenderPipelineAsset pipeline = GetActiveUniversalPipelineAsset();
            bool featureReady = false;
            if (pipeline != null)
            {
                ScriptableRendererData rendererData = EnsureRendererData(pipeline);
                featureReady = EnsureUniversalBlurFeature(rendererData);
            }

            if (blurMaterial != null && featureReady)
            {
                Debug.Log("[UnifiedBlurHud] Unified Blur material and renderer feature are configured.");
            }
            else if (blurMaterial != null)
            {
                Debug.Log("[UnifiedBlurHud] Unified Blur package material is available. Active render pipeline is not URP, so HUD will use the Built-in UI/Blur fallback.");
            }

            if (builtinMaterial != null)
            {
                Debug.Log("[UnifiedBlurHud] Built-in-compatible open-source UI blur fallback is configured.");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[UnifiedBlurHud] Automatic setup skipped: {e.Message}");
        }
    }

    private static void EnsureFolders()
    {
        EnsureFolder("Assets/Imitation", "Resources");
    }

    private static void EnsureFolder(string parent, string child)
    {
        string path = parent + "/" + child;
        if (!AssetDatabase.IsValidFolder(path))
        {
            AssetDatabase.CreateFolder(parent, child);
        }
    }

    private static Material EnsureHudMaterial()
    {
        Material existing = AssetDatabase.LoadAssetAtPath<Material>(HudMaterialPath);
        if (existing != null)
        {
            return existing;
        }

        string sourcePath = FindPackageMaterialPath();
        if (string.IsNullOrWhiteSpace(sourcePath))
        {
            return null;
        }

        if (!AssetDatabase.CopyAsset(sourcePath, HudMaterialPath))
        {
            Debug.LogWarning($"[UnifiedBlurHud] Failed to copy {sourcePath} to {HudMaterialPath}.");
            return null;
        }

        AssetDatabase.ImportAsset(HudMaterialPath);
        return AssetDatabase.LoadAssetAtPath<Material>(HudMaterialPath);
    }

    private static Material EnsureBuiltinHudMaterial()
    {
        Material existing = AssetDatabase.LoadAssetAtPath<Material>(BuiltinHudMaterialPath);
        if (existing != null)
        {
            return existing;
        }

        Shader shader = Shader.Find("UI/Blur");
        if (shader == null)
        {
            return null;
        }

        Material material = new Material(shader)
        {
            name = "RetargetingBuiltinUIBlur"
        };
        material.SetFloat("_Opacity", 0.72f);
        material.SetFloat("_Size", 2.4f);
        AssetDatabase.CreateAsset(material, BuiltinHudMaterialPath);
        AssetDatabase.ImportAsset(BuiltinHudMaterialPath);
        return AssetDatabase.LoadAssetAtPath<Material>(BuiltinHudMaterialPath);
    }

    private static string FindPackageMaterialPath()
    {
        foreach (string guid in AssetDatabase.FindAssets("UniversalBlurUI t:Material"))
        {
            string path = AssetDatabase.GUIDToAssetPath(guid);
            if (path.EndsWith("/UniversalBlurUI.mat", StringComparison.OrdinalIgnoreCase))
            {
                return path;
            }
        }

        return string.Empty;
    }

    private static UniversalRenderPipelineAsset GetActiveUniversalPipelineAsset()
    {
        UniversalRenderPipelineAsset pipeline = GraphicsSettings.renderPipelineAsset as UniversalRenderPipelineAsset;
        if (pipeline == null)
        {
            pipeline = QualitySettings.renderPipeline as UniversalRenderPipelineAsset;
        }

        return pipeline;
    }

    private static ScriptableRendererData EnsureRendererData(UniversalRenderPipelineAsset pipeline)
    {
        var serializedPipeline = new SerializedObject(pipeline);
        SerializedProperty list = serializedPipeline.FindProperty("m_RendererDataList");
        if (list != null)
        {
            if (list.arraySize == 0)
            {
                list.arraySize = 1;
            }

            SerializedProperty first = list.GetArrayElementAtIndex(0);
            if (first.objectReferenceValue is ScriptableRendererData existingRendererData)
            {
                return existingRendererData;
            }

            ScriptableRendererData rendererData = AssetDatabase.LoadAssetAtPath<ScriptableRendererData>(RendererAssetPath);
            if (rendererData == null)
            {
                EnsureFolder("Assets/Imitation", "Rendering");
                rendererData = ScriptableObject.CreateInstance<UniversalRendererData>();
                rendererData.name = "RetargetingUniversalRenderer";
                AssetDatabase.CreateAsset(rendererData, RendererAssetPath);
            }

            if (first.objectReferenceValue == null)
            {
                first.objectReferenceValue = rendererData;
            }

            SerializedProperty defaultIndex = serializedPipeline.FindProperty("m_DefaultRendererIndex");
            if (defaultIndex != null)
            {
                defaultIndex.intValue = 0;
            }

            serializedPipeline.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(pipeline);

            return rendererData;
        }

        ScriptableRendererData fallbackRendererData = AssetDatabase.LoadAssetAtPath<ScriptableRendererData>(RendererAssetPath);
        if (fallbackRendererData == null)
        {
            EnsureFolder("Assets/Imitation", "Rendering");
            fallbackRendererData = ScriptableObject.CreateInstance<UniversalRendererData>();
            fallbackRendererData.name = "RetargetingUniversalRenderer";
            AssetDatabase.CreateAsset(fallbackRendererData, RendererAssetPath);
        }

        return fallbackRendererData;
    }

    private static bool EnsureUniversalBlurFeature(ScriptableRendererData rendererData)
    {
        if (rendererData == null)
        {
            return false;
        }

        Type featureType = Type.GetType("Unified.UniversalBlur.Runtime.UniversalBlurFeature, Unified.UniversalBlur.Runtime");
        if (featureType == null)
        {
            return false;
        }

        var serializedRenderer = new SerializedObject(rendererData);
        SerializedProperty features = serializedRenderer.FindProperty("m_RendererFeatures");
        SerializedProperty featureMap = serializedRenderer.FindProperty("m_RendererFeatureMap");
        if (features == null)
        {
            return false;
        }

        for (int i = 0; i < features.arraySize; i++)
        {
            UnityEngine.Object current = features.GetArrayElementAtIndex(i).objectReferenceValue;
            if (current != null && current.GetType() == featureType)
            {
                return true;
            }
        }

        var feature = ScriptableObject.CreateInstance(featureType) as ScriptableRendererFeature;
        if (feature == null)
        {
            return false;
        }

        feature.name = "Universal Blur Feature";
        AssetDatabase.AddObjectToAsset(feature, rendererData);
        features.arraySize++;
        int index = features.arraySize - 1;
        features.GetArrayElementAtIndex(index).objectReferenceValue = feature;

        if (featureMap != null)
        {
            featureMap.arraySize = features.arraySize;
            if (AssetDatabase.TryGetGUIDAndLocalFileIdentifier(feature, out _, out long localId))
            {
                featureMap.GetArrayElementAtIndex(index).longValue = localId;
            }
        }

        serializedRenderer.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(rendererData);
        AssetDatabase.SaveAssets();
        return true;
    }

    private static void DisableAutoCreatedPipelineIfActive()
    {
        RenderPipelineAsset autoPipeline = AssetDatabase.LoadAssetAtPath<RenderPipelineAsset>(PipelineAssetPath);
        if (autoPipeline == null)
        {
            return;
        }

        bool changed = false;
        if (GraphicsSettings.renderPipelineAsset == autoPipeline)
        {
            GraphicsSettings.renderPipelineAsset = null;
            changed = true;
        }

        for (int i = 0; i < QualitySettings.names.Length; i++)
        {
            QualitySettings.SetQualityLevel(i, applyExpensiveChanges: false);
            if (QualitySettings.renderPipeline == autoPipeline)
            {
                QualitySettings.renderPipeline = null;
                changed = true;
            }
        }

        int ultra = Array.FindIndex(QualitySettings.names, name => string.Equals(name, "Ultra", StringComparison.OrdinalIgnoreCase));
        if (ultra >= 0)
        {
            QualitySettings.SetQualityLevel(ultra, applyExpensiveChanges: false);
        }

        if (changed)
        {
            Debug.LogWarning("[UnifiedBlurHud] Disabled the auto-created URP pipeline because legacy scene materials were not converted. Unified Blur remains installed but inactive until a valid URP setup is assigned manually.");
        }
    }
}
#endif
