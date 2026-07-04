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

[ExecuteAlways]
[DefaultExecutionOrder(-10000)]
public sealed partial class RetargetingHudLayout : MonoBehaviour
{
    private const string HudHostName = "RetargetingHudLayoutHost";
    private const string HudRootName = "RuntimeHUDRoot";
    private const string MainImitationScenePath = "Assets/Imitation/G1.unity";
    private const string ReplayImitationScenePath = "Assets/Imitation/G1Replay.unity";

    private static readonly Color PanelBlue = new Color(0.58f, 0.75f, 0.86f, 0.34f);
    private static readonly Color DrawerShell = new Color(0.55f, 0.78f, 0.92f, 0.30f);
    private static readonly Color DrawerContent = new Color(0.06f, 0.12f, 0.18f, 0.68f);
    private static readonly Color DropdownCaption = new Color(0.56f, 0.72f, 0.82f, 0.52f);
    private static readonly Color DropdownList = new Color(0.02f, 0.06f, 0.11f, 0.98f);
    private static readonly Color HomeColor = new Color(0.00f, 0.78f, 0.40f, 0.78f);
    private static readonly Color StartColor = new Color(0.00f, 0.30f, 1.00f, 0.82f);
    private static readonly Color ReplayColor = new Color(0.00f, 0.84f, 0.38f, 0.82f);
    private static readonly Color StopColor = new Color(1.00f, 0.04f, 0.06f, 0.82f);
    private static readonly Color SwitchColor = new Color(0.00f, 0.72f, 1.00f, 0.78f);
    private static readonly Color WhamColor = new Color(0.00f, 0.36f, 1.00f, 0.88f);
    private static readonly Color GmrColor = new Color(0.00f, 0.78f, 0.32f, 0.88f);
    private static readonly Color ParamsColor = new Color(0.72f, 0.16f, 1.00f, 0.88f);
    private static readonly Color CloseColor = new Color(1.0f, 0.12f, 0.14f, 0.96f);

    private readonly List<Graphic> ownedGraphics = new List<Graphic>();
    private readonly List<Button> lockableButtons = new List<Button>();
    private readonly List<TMP_Dropdown> lockableDropdowns = new List<TMP_Dropdown>();
    private readonly List<Toggle> lockableToggles = new List<Toggle>();
    private readonly List<ParamControl> paramControls = new List<ParamControl>();
    private readonly List<RectTransform> paramStaticRows = new List<RectTransform>();
    private readonly List<ParamGroup> paramGroups = new List<ParamGroup>();

    private Canvas canvas;
    private RectTransform hudRoot;
    private CanvasGroup whamDrawer;
    private CanvasGroup gmrDrawer;
    private CanvasGroup paramsDrawer;
    private CanvasGroup progressGroup;
    private TMP_Text progressTitle;
    private TMP_Text progressActivity;
    private TMP_Text progressDetail;
    private Image progressFill;
    private TMP_Text whamStatusText;
    private TMP_Text gmrStatusText;
    private TMP_Text whamPlaceholderText;
    private TMP_Text gmrPlaceholderText;
    private RawImage whamVideoSurface;
    private RawImage gmrVideoSurface;
    private RectTransform whamVideoViewport;
    private RectTransform gmrVideoViewport;
    private TMP_Text cameraStatusText;
    private StreamReceiver streamReceiver;
    private StartInput startInput;
    private bool built;
    private static Material unifiedBlurMaterial;
    private static Material builtinBlurMaterial;

    private sealed class ParamControl
    {
        public TMP_InputField Input;
        public Toggle Toggle;
        public FieldInfo Field;
    }

    private sealed class ParamGroup
    {
        public RectTransform Header;
        public TMP_Text Chevron;
        public readonly List<RectTransform> Rows = new List<RectTransform>();
        public bool Expanded = true;
    }

    private sealed class RobotHudEntry
    {
        public string Label;
        public string Key;
        public bool SelectedByDefault;
    }

    private static readonly RobotHudEntry[] RobotHudEntries =
    {
        new RobotHudEntry { Label = "G1", Key = "unitree_g1", SelectedByDefault = true },
        new RobotHudEntry { Label = "H1", Key = "unitree_h1" },
        new RobotHudEntry { Label = "X02Lite", Key = "x02lite" },
        new RobotHudEntry { Label = "OpenLoong", Key = "openloong" },
    };

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void Bootstrap()
    {
        SceneManager.sceneLoaded -= OnSceneLoaded;
        SceneManager.sceneLoaded += OnSceneLoaded;
        TryBootstrapForScene(SceneManager.GetActiveScene());
    }

    private static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        TryBootstrapForScene(scene);
    }

    private static void TryBootstrapForScene(Scene scene)
    {
        if (!IsImitationScene(scene))
        {
            CleanupGeneratedHudObjects();
            return;
        }

        if (FindObjectOfType<RetargetingHudLayout>(true) != null)
        {
            return;
        }

        Canvas sceneCanvas = FindBestCanvasStatic();
        var host = new GameObject(HudHostName, typeof(RectTransform));
        host.hideFlags = HideFlags.DontSave;
        if (sceneCanvas != null)
        {
            host.transform.SetParent(sceneCanvas.transform, false);
        }

        host.AddComponent<RetargetingHudLayout>();
    }

#if UNITY_EDITOR
    [UnityEditor.InitializeOnLoadMethod]
    private static void EditorBootstrap()
    {
        UnityEditor.EditorApplication.delayCall += EnsureEditorHudHost;
    }

    private static void EnsureEditorHudHost()
    {
        if (Application.isPlaying)
        {
            return;
        }

        Scene activeScene = SceneManager.GetActiveScene();
        if (!IsImitationScene(activeScene))
        {
            CleanupGeneratedHudObjects();
            return;
        }

        if (FindObjectOfType<RetargetingHudLayout>(true) != null)
        {
            return;
        }

        Canvas sceneCanvas = FindBestCanvasStatic();
        if (sceneCanvas == null)
        {
            return;
        }

        var host = GameObject.Find(HudHostName);
        if (host == null)
        {
            host = new GameObject(HudHostName, typeof(RectTransform));
            host.hideFlags = HideFlags.DontSave;
            host.transform.SetParent(sceneCanvas.transform, false);
            UnityEditor.Undo.RegisterCreatedObjectUndo(host, "Create retargeting HUD layout host");
        }
        else
        {
            host.hideFlags = HideFlags.DontSave;
        }

        if (host.GetComponent<RetargetingHudLayout>() == null)
        {
            UnityEditor.Undo.AddComponent<RetargetingHudLayout>(host);
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(host.scene);
        }
    }
#endif

    private void OnEnable()
    {
        if (!IsImitationScene(gameObject.scene))
        {
            CleanupGeneratedHudObjects();
            DestroyThisComponentHost();
            return;
        }

        Build();
    }

    private void Start()
    {
        Build();
    }

    private void Update()
    {
        if (!IsImitationScene(gameObject.scene))
        {
            CleanupGeneratedHudObjects();
            DestroyThisComponentHost();
            return;
        }

        SelectedRobotCameraFollow.EnsureAtLeastOneRenderingCamera();
        SyncRealtimeUiState();
        UpdateProgressPanel();
        UpdateStreamReceiverStatus();
        UpdateCameraStatusText();
    }

    private void Build()
    {
        if (!IsImitationScene(gameObject.scene))
        {
            CleanupGeneratedHudObjects();
            return;
        }

        if (built)
        {
            return;
        }

        startInput = FindObjectOfType<StartInput>(true);
        streamReceiver = ResolveStreamReceiverForHud();
        canvas = FindBestCanvas();
        if (canvas == null)
        {
            canvas = CreateHudCanvas();
        }

        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 500;
        EnsureCanvasScaler(canvas.gameObject);
        EnsureEventSystem();
        DestroyLeakedGeneratedHudChildren(canvas.transform);

        var existingRoot = GameObject.Find(HudRootName);
        if (existingRoot != null)
        {
            DetachPreservedHudObjects(existingRoot.transform, canvas.transform);
            DestroyHudObject(existingRoot);
        }

        ownedGraphics.Clear();
        lockableButtons.Clear();
        lockableDropdowns.Clear();
        lockableToggles.Clear();
        paramControls.Clear();
        paramStaticRows.Clear();
        paramGroups.Clear();

        hudRoot = CreateRect(HudRootName, canvas.transform);
        hudRoot.gameObject.hideFlags = HideFlags.DontSave;
        Stretch(hudRoot);

        var movedRoots = new HashSet<Transform>();
        BuildControlDock(movedRoots);
        BuildPluginDock(movedRoots);
        BuildProgressPanel();
        DisableLegacyCanvasGraphics(movedRoots);
        built = true;
    }

}

public sealed class HudButtonFeedback : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler, IPointerDownHandler, IPointerUpHandler
{
    private Image image;
    private Button button;
    private Color normal;
    private Color hover;
    private Color pressed;
    private Color disabled;
    private bool pointerInside;
    private bool pointerDown;

    public void SetColors(Color normalColor)
    {
        image = GetComponent<Image>();
        button = GetComponent<Button>();
        normal = normalColor;
        hover = Color.Lerp(normalColor, Color.black, 0.22f);
        hover.a = Mathf.Clamp01(normalColor.a + 0.12f);
        pressed = Color.Lerp(normalColor, Color.black, 0.42f);
        pressed.a = Mathf.Clamp01(normalColor.a + 0.14f);
        disabled = new Color(0.30f, 0.34f, 0.38f, 0.42f);
        Apply();
    }

    private void Update()
    {
        Apply();
    }

    public void OnPointerEnter(PointerEventData eventData)
    {
        pointerInside = true;
        Apply();
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        pointerInside = false;
        pointerDown = false;
        Apply();
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        pointerDown = true;
        Apply();
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        pointerDown = false;
        Apply();
    }

    private void Apply()
    {
        if (image == null)
        {
            image = GetComponent<Image>();
        }

        if (button == null)
        {
            button = GetComponent<Button>();
        }

        if (image == null)
        {
            return;
        }

        if (button != null && !button.interactable)
        {
            image.color = disabled;
        }
        else if (pointerDown)
        {
            image.color = pressed;
        }
        else if (pointerInside)
        {
            image.color = hover;
        }
        else
        {
            image.color = normal;
        }
    }
}

public sealed class HudDragHandle : MonoBehaviour, IBeginDragHandler, IDragHandler
{
    private const float MinVisiblePixels = 96f;
    private RectTransform panel;
    private RectTransform clampRoot;
    private Vector2 startPointerLocal;
    private Vector2 startAnchoredPosition;
    private readonly Vector3[] panelWorldCorners = new Vector3[4];

    public void Initialize(RectTransform panelToMove, RectTransform root)
    {
        panel = panelToMove;
        clampRoot = root;
    }

    public void OnBeginDrag(PointerEventData eventData)
    {
        if (panel == null || clampRoot == null)
        {
            return;
        }

        RectTransformUtility.ScreenPointToLocalPointInRectangle(clampRoot, eventData.position, eventData.pressEventCamera, out startPointerLocal);
        startAnchoredPosition = panel.anchoredPosition;
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (panel == null || clampRoot == null)
        {
            return;
        }

        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(clampRoot, eventData.position, eventData.pressEventCamera, out Vector2 currentPointerLocal))
        {
            return;
        }

        panel.anchoredPosition = ClampAnchoredPosition(startAnchoredPosition + currentPointerLocal - startPointerLocal);
    }

    private Vector2 ClampAnchoredPosition(Vector2 candidate)
    {
        Rect root = clampRoot.rect;
        Vector2 original = panel.anchoredPosition;
        panel.anchoredPosition = candidate;
        Canvas.ForceUpdateCanvases();
        panel.ForceUpdateRectTransforms();
        panel.GetWorldCorners(panelWorldCorners);
        panel.anchoredPosition = original;

        Vector2 min = new Vector2(float.PositiveInfinity, float.PositiveInfinity);
        Vector2 max = new Vector2(float.NegativeInfinity, float.NegativeInfinity);
        for (int i = 0; i < panelWorldCorners.Length; i++)
        {
            Vector3 local = clampRoot.InverseTransformPoint(panelWorldCorners[i]);
            min = Vector2.Min(min, local);
            max = Vector2.Max(max, local);
        }

        Vector2 correction = Vector2.zero;
        float visibleX = Mathf.Min(MinVisiblePixels, root.width * 0.5f);
        float visibleY = Mathf.Min(MinVisiblePixels, root.height * 0.5f);

        if (max.x < root.xMin + visibleX)
        {
            correction.x += root.xMin + visibleX - max.x;
        }
        else if (min.x > root.xMax - visibleX)
        {
            correction.x -= min.x - (root.xMax - visibleX);
        }

        if (max.y < root.yMin + visibleY)
        {
            correction.y += root.yMin + visibleY - max.y;
        }
        else if (min.y > root.yMax - visibleY)
        {
            correction.y -= min.y - (root.yMax - visibleY);
        }

        return candidate + correction;
    }
}

public sealed class HudVideoAspectFit : MonoBehaviour
{
    private RectTransform viewport;
    private RawImage rawImage;
    private RectTransform rectTransform;

    public void Initialize(RectTransform viewportRect, RawImage image)
    {
        viewport = viewportRect;
        rawImage = image;
        rectTransform = image != null ? image.rectTransform : GetComponent<RectTransform>();
        Apply();
    }

    private void LateUpdate()
    {
        Apply();
    }

    private void Apply()
    {
        if (rectTransform == null)
        {
            rectTransform = GetComponent<RectTransform>();
        }

        if (rawImage == null)
        {
            rawImage = GetComponent<RawImage>();
        }

        if (viewport == null || rectTransform == null)
        {
            return;
        }

        Vector2 box = viewport.rect.size;
        if (box.x <= 1f || box.y <= 1f)
        {
            return;
        }

        Texture texture = rawImage != null ? rawImage.texture : null;
        if (texture == null || texture.width <= 0 || texture.height <= 0)
        {
            rectTransform.anchorMin = Vector2.zero;
            rectTransform.anchorMax = Vector2.one;
            rectTransform.pivot = new Vector2(0.5f, 0.5f);
            rectTransform.offsetMin = Vector2.zero;
            rectTransform.offsetMax = Vector2.zero;
            return;
        }

        float textureAspect = texture.width / (float)texture.height;
        float boxAspect = box.x / box.y;
        Vector2 fittedSize = textureAspect >= boxAspect
            ? new Vector2(box.x, box.x / textureAspect)
            : new Vector2(box.y * textureAspect, box.y);

        rectTransform.anchorMin = new Vector2(0.5f, 0.5f);
        rectTransform.anchorMax = new Vector2(0.5f, 0.5f);
        rectTransform.pivot = new Vector2(0.5f, 0.5f);
        rectTransform.anchoredPosition = Vector2.zero;
        rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, fittedSize.x);
        rectTransform.SetSizeWithCurrentAnchors(RectTransform.Axis.Vertical, fittedSize.y);
    }
}

public sealed class HudRawImageColorGuard : MonoBehaviour
{
    private RawImage rawImage;

    private void Awake()
    {
        rawImage = GetComponent<RawImage>();
    }

    private void Update()
    {
        if (rawImage == null)
        {
            rawImage = GetComponent<RawImage>();
        }

        if (rawImage == null)
        {
            return;
        }

        rawImage.color = rawImage.texture == null
            ? new Color(1f, 1f, 1f, 0f)
            : Color.white;
    }
}
