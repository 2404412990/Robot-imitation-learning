using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using Gewu.Imitation;

public sealed partial class RetargetingHudLayout
{
    private RawImage CreateVideoSurface(string name, RectTransform parent, out RectTransform viewport)
    {
        viewport = CreatePanel(name + "Viewport", parent, new Color(0.03f, 0.07f, 0.11f, 0.62f));
        Stretch(viewport);
        Image viewportImage = viewport.GetComponent<Image>();
        if (viewportImage != null)
        {
            viewportImage.raycastTarget = false;
        }

        RectTransform rect = CreateRect(name, viewport);
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = Vector2.zero;
        rect.sizeDelta = Vector2.zero;
        RawImage raw = rect.gameObject.AddComponent<RawImage>();
        raw.raycastTarget = false;
        raw.color = new Color(1f, 1f, 1f, 0f);
        ownedGraphics.Add(raw);
        raw.gameObject.AddComponent<HudVideoAspectFit>().Initialize(viewport, raw);
        return raw;
    }

    private void PushGmrRenderSizeToStartInput(RectTransform videoRect)
    {
        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>(true);
        }

        if (startInput == null || videoRect == null)
        {
            return;
        }

        Rect pixelRect = RectTransformUtility.PixelAdjustRect(videoRect, canvas);
        int width = Mathf.RoundToInt(Mathf.Max(videoRect.rect.width, pixelRect.width));
        int height = Mathf.RoundToInt(Mathf.Max(videoRect.rect.height, pixelRect.height));
        startInput.SetGmrTcpRenderSizeFromHud(width, height);
    }

    private GameObject FindHomeButton()
    {
        GameObject named = GameObject.Find("Home");
        if (named != null)
        {
            Button button = named.GetComponentInParent<Button>(true);
            return button != null ? button.gameObject : named;
        }

        foreach (TMP_Text text in FindObjectsOfType<TMP_Text>(true))
        {
            if (string.Equals(text.text?.Trim(), "Home", StringComparison.OrdinalIgnoreCase))
            {
                Button button = text.GetComponentInParent<Button>(true);
                return button != null ? button.gameObject : text.gameObject;
            }
        }

        return null;
    }

    private GameObject MoveObject(
        string[] candidateNames,
        RectTransform parent,
        Vector2 anchoredPosition,
        Vector2 size,
        HashSet<Transform> movedRoots,
        bool preferInteractiveRoot = true)
    {
        GameObject go = ResolveHudObjectRoot(candidateNames, preferInteractiveRoot);
        if (go == null)
        {
            return null;
        }

        RectTransform rect = EnsureRectTransform(go);
        rect.SetParent(parent, false);
        rect.localRotation = Quaternion.identity;
        rect.localScale = Vector3.one;
        if (size == Vector2.zero)
        {
            Stretch(rect);
        }
        else
        {
            AnchorTopLeft(rect, anchoredPosition, size);
        }

        movedRoots.Add(go.transform);
        return go;
    }

    private static GameObject ResolveHudObjectRoot(string[] candidateNames, bool preferInteractiveRoot)
    {
        foreach (string name in candidateNames)
        {
            if (string.IsNullOrWhiteSpace(name))
            {
                continue;
            }

            GameObject go = GameObject.Find(name);
            if (go == null)
            {
                continue;
            }

            if (!preferInteractiveRoot)
            {
                return go;
            }

            TMP_Dropdown dropdown = go.GetComponent<TMP_Dropdown>() ?? go.GetComponentInParent<TMP_Dropdown>(true);
            if (dropdown != null)
            {
                return dropdown.gameObject;
            }

            Button button = go.GetComponent<Button>() ?? go.GetComponentInParent<Button>(true);
            if (button != null)
            {
                return button.gameObject;
            }

            return go;
        }

        return null;
    }

    private void MoveAsHudButton(GameObject go, RectTransform parent, Vector2 anchoredPosition, Vector2 size, string label, Color color)
    {
        RectTransform rect = EnsureRectTransform(go);
        rect.SetParent(parent, false);
        rect.localRotation = Quaternion.identity;
        rect.localScale = Vector3.one;
        AnchorTopLeft(rect, anchoredPosition, size);
        StyleButton(go, label, color, lockable: false);
    }

    private TMP_Text AddText(
        RectTransform parent,
        string name,
        string textValue,
        float fontSize,
        FontStyles style,
        Vector2 anchoredPosition,
        Vector2 size,
        TextAlignmentOptions alignment = TextAlignmentOptions.Left)
    {
        RectTransform rect = CreateRect(name, parent);
        if (anchoredPosition == Vector2.zero && alignment == TextAlignmentOptions.Center)
        {
            Stretch(rect);
        }
        else
        {
            AnchorTopLeft(rect, anchoredPosition, size);
        }

        TextMeshProUGUI text = rect.gameObject.AddComponent<TextMeshProUGUI>();
        text.text = textValue;
        text.fontSize = fontSize;
        text.fontStyle = style;
        text.alignment = alignment;
        text.color = Color.white;
        text.raycastTarget = false;
        text.enableWordWrapping = true;
        ownedGraphics.Add(text);
        return text;
    }

    private static void ConfigureSingleLine(TMP_Text text)
    {
        if (text == null)
        {
            return;
        }

        text.enableWordWrapping = false;
        text.overflowMode = TextOverflowModes.Ellipsis;
    }

    private static string CompactHudLine(string value, int maxChars)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        string compact = value.Replace('\r', ' ').Replace('\n', ' ').Trim();
        while (compact.Contains("  "))
        {
            compact = compact.Replace("  ", " ");
        }

        return compact.Length <= maxChars ? compact : compact.Substring(0, Mathf.Max(0, maxChars - 3)) + "...";
    }

    private static bool IsImitationScene(Scene scene)
    {
        string path = scene.path ?? string.Empty;
        if (string.Equals(path, MainImitationScenePath, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(path, ReplayImitationScenePath, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        string name = scene.name ?? string.Empty;
        return string.Equals(name, "G1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(name, "G1Replay", StringComparison.OrdinalIgnoreCase);
    }

    private static void CleanupGeneratedHudObjects()
    {
        DestroyGeneratedObject(GameObject.Find(HudRootName));
        DestroyGeneratedObject(GameObject.Find(HudHostName));
    }

    private void DestroyThisComponentHost()
    {
        if (this == null)
        {
            return;
        }

        if (gameObject.name == HudHostName)
        {
            DestroyGeneratedObject(gameObject);
            return;
        }

        DestroyGeneratedObject(this);
    }

    private static void DestroyGeneratedObject(UnityEngine.Object target)
    {
        if (target == null)
        {
            return;
        }

        if (Application.isPlaying)
        {
            Destroy(target);
        }
        else
        {
            DestroyImmediate(target);
        }
    }

    private RectTransform CreatePanel(string name, Transform parent, Color color, bool blur = false)
    {
        RectTransform rect = CreateRect(name, parent);
        Image image = rect.gameObject.AddComponent<Image>();
        image.type = Image.Type.Simple;
        image.fillCenter = true;
        image.preserveAspect = false;
        image.color = color;
        if (blur)
        {
            ApplyUnifiedBlurMaterial(image);
        }
        image.raycastTarget = true;
        ownedGraphics.Add(image);
        return rect;
    }

    private static void ApplyUnifiedBlurMaterial(Graphic graphic)
    {
        Material material = ResolveUnifiedBlurMaterial();
        graphic.material = material;
    }

    private static Material ResolveUnifiedBlurMaterial()
    {
        if (!IsUniversalRenderPipelineActive())
        {
            return ResolveBuiltinBlurMaterial();
        }

        if (unifiedBlurMaterial != null)
        {
            return unifiedBlurMaterial;
        }

        unifiedBlurMaterial = Resources.Load<Material>("RetargetingUniversalBlur");
        if (unifiedBlurMaterial != null)
        {
            return unifiedBlurMaterial;
        }

        Shader shader = Shader.Find("Unify/UI/Tinted Blur");
        if (shader != null)
        {
            unifiedBlurMaterial = new Material(shader)
            {
                name = "Runtime Retargeting Universal Blur"
            };
        }

        return unifiedBlurMaterial;
    }

    private static Material ResolveBuiltinBlurMaterial()
    {
        if (builtinBlurMaterial != null)
        {
            return builtinBlurMaterial;
        }

        builtinBlurMaterial = Resources.Load<Material>("RetargetingBuiltinUIBlur");
        if (builtinBlurMaterial != null)
        {
            return builtinBlurMaterial;
        }

        Shader shader = Shader.Find("UI/Blur");
        if (shader != null)
        {
            builtinBlurMaterial = new Material(shader)
            {
                name = "Runtime Retargeting Builtin UI Blur"
            };
            builtinBlurMaterial.SetFloat("_Opacity", 0.72f);
            builtinBlurMaterial.SetFloat("_Size", 2.4f);
        }

        return builtinBlurMaterial;
    }

    private static bool IsUniversalRenderPipelineActive()
    {
        RenderPipelineAsset active = GraphicsSettings.currentRenderPipeline != null
            ? GraphicsSettings.currentRenderPipeline
            : QualitySettings.renderPipeline;
        string typeName = active != null ? active.GetType().FullName : string.Empty;
        return !string.IsNullOrEmpty(typeName) &&
               typeName.IndexOf("UniversalRenderPipelineAsset", StringComparison.OrdinalIgnoreCase) >= 0;
    }

    private static Canvas FindBestCanvas()
    {
        GameObject named = GameObject.Find("Canvas");
        if (named != null && named.TryGetComponent(out Canvas namedCanvas))
        {
            return namedCanvas;
        }

        named = GameObject.Find("HUDCanvas");
        if (named != null && named.TryGetComponent(out namedCanvas))
        {
            return namedCanvas;
        }

        return FindObjectOfType<Canvas>(true);
    }

    private static Canvas FindBestCanvasStatic()
    {
        return FindBestCanvas();
    }

    private static Canvas CreateHudCanvas()
    {
        var canvasObject = new GameObject("Canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        Canvas created = canvasObject.GetComponent<Canvas>();
        created.renderMode = RenderMode.ScreenSpaceOverlay;
        return created;
    }

    private static void EnsureCanvasScaler(GameObject canvasObject)
    {
        CanvasScaler scaler = canvasObject.GetComponent<CanvasScaler>() ?? canvasObject.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        scaler.matchWidthOrHeight = 0.5f;

        GraphicRaycaster raycaster = canvasObject.GetComponent<GraphicRaycaster>();
        if (raycaster == null)
        {
            raycaster = canvasObject.AddComponent<GraphicRaycaster>();
        }

        raycaster.enabled = true;
    }

    private static void EnsureEventSystem()
    {
        EventSystem eventSystem = FindObjectOfType<EventSystem>(true);
        if (eventSystem != null)
        {
            eventSystem.gameObject.SetActive(true);
            eventSystem.enabled = true;
            StandaloneInputModule module = eventSystem.GetComponent<StandaloneInputModule>() ?? eventSystem.gameObject.AddComponent<StandaloneInputModule>();
            module.enabled = true;
            return;
        }

        var go = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
        go.hideFlags = HideFlags.DontSave;
    }

    private static RectTransform CreateRect(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.hideFlags = HideFlags.DontSave;
        RectTransform rect = go.GetComponent<RectTransform>();
        rect.SetParent(parent, false);
        return rect;
    }

    private static RectTransform EnsureRectTransform(GameObject go)
    {
        RectTransform rect = go.GetComponent<RectTransform>();
        return rect != null ? rect : go.AddComponent<RectTransform>();
    }

    private static void AnchorTopLeft(RectTransform rect, Vector2 anchoredPosition, Vector2 size)
    {
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(0f, 1f);
        rect.pivot = new Vector2(0f, 1f);
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = size;
    }

    private static void Stretch(RectTransform rect)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.offsetMin = Vector2.zero;
        rect.offsetMax = Vector2.zero;
    }

    private static void Stretch(RectTransform rect, float left, float bottom, float right, float top)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.pivot = new Vector2(0.5f, 0.5f);
        rect.offsetMin = new Vector2(left, bottom);
        rect.offsetMax = new Vector2(-right, -top);
    }

    private static void StretchHorizontalTop(RectTransform rect, float left, float right, float top, float height)
    {
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(0.5f, 1f);
        rect.anchoredPosition = new Vector2(0f, -top);
        rect.sizeDelta = new Vector2(-(left + right), height);
        rect.offsetMin = new Vector2(left, rect.offsetMin.y);
        rect.offsetMax = new Vector2(-right, rect.offsetMax.y);
    }

    private static void StretchHorizontalBottom(RectTransform rect, float left, float right, float bottom, float height)
    {
        rect.anchorMin = new Vector2(0f, 0f);
        rect.anchorMax = new Vector2(1f, 0f);
        rect.pivot = new Vector2(0.5f, 0f);
        rect.anchoredPosition = new Vector2(0f, bottom);
        rect.sizeDelta = new Vector2(-(left + right), height);
        rect.offsetMin = new Vector2(left, rect.offsetMin.y);
        rect.offsetMax = new Vector2(-right, rect.offsetMax.y);
    }

    private static void StretchRow(RectTransform rect, float y, float height, float horizontalPadding)
    {
        rect.anchorMin = new Vector2(0f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(0.5f, 1f);
        rect.anchoredPosition = new Vector2(0f, y);
        rect.sizeDelta = new Vector2(-horizontalPadding * 2f, height);
    }

    private static ColorBlock BuildSelectableColors(Color normal)
    {
        Color highlighted = Color.Lerp(normal, Color.white, 0.12f);
        highlighted.a = Mathf.Clamp01(normal.a + 0.10f);
        Color pressed = Color.Lerp(normal, Color.black, 0.24f);
        pressed.a = Mathf.Clamp01(normal.a + 0.12f);
        return new ColorBlock
        {
            normalColor = normal,
            highlightedColor = highlighted,
            pressedColor = pressed,
            selectedColor = normal,
            disabledColor = new Color(0.30f, 0.34f, 0.38f, 0.45f),
            colorMultiplier = 1f,
            fadeDuration = 0.06f
        };
    }

    private void DisableLegacyCanvasGraphics(HashSet<Transform> movedRoots)
    {
        foreach (Graphic graphic in canvas.GetComponentsInChildren<Graphic>(true))
        {
            if (graphic == null || ownedGraphics.Contains(graphic) || graphic.transform.IsChildOf(hudRoot))
            {
                continue;
            }

            bool isMoved = false;
            foreach (Transform moved in movedRoots)
            {
                if (moved != null && (graphic.transform == moved || graphic.transform.IsChildOf(moved)))
                {
                    isMoved = true;
                    break;
                }
            }

            if (!isMoved)
            {
                graphic.enabled = false;
                graphic.raycastTarget = false;
            }
        }
    }

    private static void DestroyLeakedGeneratedHudChildren(Transform canvasTransform)
    {
        if (canvasTransform == null)
        {
            return;
        }

        var toDestroy = new List<GameObject>();
        for (int i = 0; i < canvasTransform.childCount; i++)
        {
            Transform child = canvasTransform.GetChild(i);
            if (child != null && IsGeneratedHudLeakName(child.name))
            {
                toDestroy.Add(child.gameObject);
            }
        }

        for (int i = 0; i < toDestroy.Count; i++)
        {
            DestroyHudObject(toDestroy[i]);
        }
    }

    private static bool IsGeneratedHudLeakName(string objectName)
    {
        if (string.IsNullOrEmpty(objectName))
        {
            return false;
        }

        switch (objectName)
        {
            case "ControlDock":
            case "PluginDock":
            case "StartProgressPanel":
            case "WHAMDrawer":
            case "GMRDrawer":
            case "PARAMSDrawer":
            case "SwitchCameraButton":
            case "SwitchRobotButton":
            case "WHAMEntry":
            case "GMREntry":
            case "PARAMSEntry":
            case "HomeHudButton":
            case "StartHudButton":
            case "ReplayHudButton":
            case "StopHudButton":
                return true;
            default:
                return objectName.StartsWith("RobotRow_", StringComparison.Ordinal) ||
                       objectName.StartsWith("CsvList_", StringComparison.Ordinal);
        }
    }

    private static void DetachPreservedHudObjects(Transform oldRoot, Transform fallbackParent)
    {
        string[] names =
        {
            "Home", "RoboList", "CsvList", "StartButton", "Start", "ReplayButton", "Replay",
            "StopButton", "Stop", "HomeButton", "WHAM", "GMR"
        };

        foreach (string name in names)
        {
            GameObject go = GameObject.Find(name);
            if (go == null || !IsChildOrSelf(go.transform, oldRoot))
            {
                continue;
            }

            go.transform.SetParent(fallbackParent, false);
            go.SetActive(true);
        }
    }

    private static bool IsChildOrSelf(Transform candidate, Transform root)
    {
        return candidate != null && root != null && (candidate == root || candidate.IsChildOf(root));
    }

    private static void DestroyHudObject(GameObject go)
    {
        if (go == null)
        {
            return;
        }

        if (Application.isPlaying)
        {
            Destroy(go);
        }
        else
        {
            DestroyImmediate(go);
        }
    }

    private static string Nicify(string fieldName)
    {
        if (string.IsNullOrEmpty(fieldName))
        {
            return string.Empty;
        }

        var chars = new List<char>(fieldName.Length + 8) { char.ToUpperInvariant(fieldName[0]) };
        for (int i = 1; i < fieldName.Length; i++)
        {
            char c = fieldName[i];
            if (char.IsUpper(c))
            {
                chars.Add(' ');
            }

            chars.Add(c);
        }

        return new string(chars.ToArray());
    }
}
