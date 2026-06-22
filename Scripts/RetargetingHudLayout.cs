using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

[ExecuteAlways]
[DefaultExecutionOrder(-10000)]
public sealed class RetargetingHudLayout : MonoBehaviour
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
            host.transform.SetParent(sceneCanvas.transform, false);
            UnityEditor.Undo.RegisterCreatedObjectUndo(host, "Create retargeting HUD layout host");
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

        var existingRoot = GameObject.Find(HudRootName);
        if (existingRoot != null)
        {
            DetachPreservedHudObjects(existingRoot.transform, canvas.transform);
            DestroyHudObject(existingRoot);
        }

        ownedGraphics.Clear();
        lockableButtons.Clear();
        lockableDropdowns.Clear();
        paramControls.Clear();
        paramStaticRows.Clear();
        paramGroups.Clear();

        hudRoot = CreateRect(HudRootName, canvas.transform);
        Stretch(hudRoot);

        var movedRoots = new HashSet<Transform>();
        BuildControlDock(movedRoots);
        BuildPluginDock(movedRoots);
        BuildProgressPanel();
        DisableLegacyCanvasGraphics(movedRoots);
        built = true;
    }

    private void BuildControlDock(HashSet<Transform> movedRoots)
    {
        RectTransform controlDock = CreatePanel("ControlDock", hudRoot, PanelBlue, blur: true);
        AnchorTopLeft(controlDock, new Vector2(24f, -18f), new Vector2(420f, 760f));
        controlDock.gameObject.AddComponent<HudDragHandle>().Initialize(controlDock, hudRoot);

        CreateProxyButton("HomeHudButton", new[] { "Home", "HomeButton" }, controlDock, new Vector2(32f, -24f), new Vector2(356f, 72f), "Home", HomeColor, false, movedRoots, OpenHomeScene);

        AddText(controlDock, "Title", "Retargeting", 32, FontStyles.Bold, new Vector2(32f, -128f), new Vector2(356f, 40f));
        AddText(controlDock, "Subtitle", "Robot imitation control", 15, FontStyles.Normal, new Vector2(32f, -164f), new Vector2(356f, 24f));

        AddText(controlDock, "RobotLabel", "Robot", 20, FontStyles.Bold, new Vector2(32f, -214f), new Vector2(356f, 28f));
        GameObject robo = MoveObject(new[] { "RoboList" }, controlDock, new Vector2(32f, -252f), new Vector2(356f, 58f), movedRoots);
        StyleDropdown(robo);

        AddText(controlDock, "CsvLabel", "Motion / CSV", 20, FontStyles.Bold, new Vector2(32f, -330f), new Vector2(356f, 28f));
        GameObject csv = MoveObject(new[] { "CsvList" }, controlDock, new Vector2(32f, -368f), new Vector2(356f, 58f), movedRoots);
        StyleDropdown(csv);

        CreateProxyButton("StartHudButton", new[] { "StartButton", "Start" }, controlDock, new Vector2(32f, -462f), new Vector2(356f, 70f), "Start", StartColor, true, movedRoots, null);
        CreateProxyButton("ReplayHudButton", new[] { "ReplayButton", "Replay" }, controlDock, new Vector2(32f, -560f), new Vector2(164f, 64f), "Replay", ReplayColor, true, movedRoots, null);
        CreateProxyButton("StopHudButton", new[] { "StopButton", "Stop" }, controlDock, new Vector2(224f, -560f), new Vector2(164f, 64f), "Stop", StopColor, false, movedRoots, null);

        Button switchCamera = CreateButton("SwitchCameraButton", controlDock, "Switch Camera", SwitchColor);
        AnchorTopLeft((RectTransform)switchCamera.transform, new Vector2(32f, -646f), new Vector2(356f, 64f));
        switchCamera.onClick.AddListener(SelectedRobotCameraFollow.SwitchNextView);
    }

    private void BuildPluginDock(HashSet<Transform> movedRoots)
    {
        RectTransform dock = CreateRect("PluginDock", hudRoot);
        dock.anchorMin = new Vector2(1f, 0.5f);
        dock.anchorMax = new Vector2(1f, 0.5f);
        dock.pivot = new Vector2(1f, 0.5f);
        dock.anchoredPosition = new Vector2(-30f, 0f);
        dock.sizeDelta = new Vector2(132f, 400f);

        Button whamButton = CreateButton("WHAMEntry", dock, "WHAM", WhamColor);
        AnchorTopLeft((RectTransform)whamButton.transform, new Vector2(0f, -0f), new Vector2(120f, 120f));
        whamButton.onClick.AddListener(() => ShowDrawer(whamDrawer));

        Button gmrButton = CreateButton("GMREntry", dock, "GMR", GmrColor);
        AnchorTopLeft((RectTransform)gmrButton.transform, new Vector2(0f, -140f), new Vector2(120f, 120f));
        gmrButton.onClick.AddListener(() => ShowDrawer(gmrDrawer));

        Button paramsButton = CreateButton("PARAMSEntry", dock, "PARAMS", ParamsColor);
        AnchorTopLeft((RectTransform)paramsButton.transform, new Vector2(0f, -280f), new Vector2(120f, 120f));
        paramsButton.onClick.AddListener(() => ShowDrawer(paramsDrawer));

        whamDrawer = CreateVideoDrawer("WHAMDrawer", "WHAM Input", "Live human/video stream", "WHAM", "WHAM stream waiting", movedRoots);
        gmrDrawer = CreateVideoDrawer("GMRDrawer", "GMR Robot Preview", "MuJoCo retarget preview", "GMR", "GMR stream waiting", movedRoots);
        paramsDrawer = CreateParamsDrawer();
        BindStreamReceiverTargets();
        HideDrawer(whamDrawer);
        HideDrawer(gmrDrawer);
        HideDrawer(paramsDrawer);
    }

    private void BuildProgressPanel()
    {
        RectTransform panel = CreatePanel("StartProgressPanel", hudRoot, new Color(0.02f, 0.05f, 0.08f, 0.72f), blur: true);
        panel.anchorMin = new Vector2(0.5f, 0f);
        panel.anchorMax = new Vector2(0.5f, 0f);
        panel.pivot = new Vector2(0.5f, 0f);
        panel.anchoredPosition = new Vector2(0f, 26f);
        panel.sizeDelta = new Vector2(980f, 122f);

        progressTitle = AddText(panel, "Title", "Starting retargeting", 21, FontStyles.Bold, new Vector2(24f, -12f), new Vector2(920f, 28f));
        progressActivity = AddText(panel, "Stage", "Waiting for Start...", 16, FontStyles.Bold, new Vector2(24f, -42f), new Vector2(920f, 24f));
        progressDetail = AddText(panel, "Detail", string.Empty, 13, FontStyles.Normal, new Vector2(24f, -66f), new Vector2(920f, 20f));
        ConfigureSingleLine(progressTitle);
        ConfigureSingleLine(progressActivity);
        ConfigureSingleLine(progressDetail);

        RectTransform barBack = CreatePanel("ProgressBar", panel, new Color(1f, 1f, 1f, 0.18f));
        StretchHorizontalTop(barBack, 24f, 24f, 94f, 10f);
        RectTransform fillRect = CreatePanel("Fill", barBack, new Color(0.08f, 0.72f, 0.96f, 0.95f));
        fillRect.anchorMin = new Vector2(0f, 0f);
        fillRect.anchorMax = new Vector2(0f, 1f);
        fillRect.pivot = new Vector2(0f, 0.5f);
        fillRect.offsetMin = Vector2.zero;
        fillRect.offsetMax = Vector2.zero;
        progressFill = fillRect.GetComponent<Image>();

        progressGroup = panel.gameObject.AddComponent<CanvasGroup>();
        progressGroup.alpha = 0f;
        progressGroup.blocksRaycasts = false;
        progressGroup.interactable = false;
    }

    private CanvasGroup CreateVideoDrawer(
        string name,
        string title,
        string subtitle,
        string videoObjectName,
        string placeholderText,
        HashSet<Transform> movedRoots)
    {
        RectTransform panel = CreatePanel(name, hudRoot, DrawerShell, blur: true);
        panel.anchorMin = new Vector2(0.5f, 0.5f);
        panel.anchorMax = new Vector2(0.5f, 0.5f);
        panel.pivot = new Vector2(0.5f, 0.5f);
        panel.anchoredPosition = Vector2.zero;
        panel.sizeDelta = new Vector2(1120f, 900f);

        RectTransform header = CreatePanel("Header", panel, new Color(0.62f, 0.82f, 0.94f, 0.32f));
        StretchHorizontalTop(header, 0f, 0f, 0f, 104f);
        header.gameObject.AddComponent<HudDragHandle>().Initialize(panel, hudRoot);
        AddText(header, "Title", title, 30, FontStyles.Bold, new Vector2(28f, -20f), new Vector2(720f, 38f));
        AddText(header, "Subtitle", subtitle, 17, FontStyles.Normal, new Vector2(28f, -62f), new Vector2(720f, 28f));
        CreateCloseButton(header, () => HideDrawer(panel.GetComponent<CanvasGroup>()));

        RectTransform content = CreatePanel("Content", panel, DrawerContent);
        Stretch(content, 28f, 50f, 28f, 124f);
        RawImage surface = CreateVideoSurface(videoObjectName + "VideoSurface", content, out RectTransform viewport);
        TMP_Text placeholder = AddText(content, "Placeholder", placeholderText, 24, FontStyles.Bold, Vector2.zero, new Vector2(900f, 42f), TextAlignmentOptions.Center);

        if (string.Equals(videoObjectName, "WHAM", StringComparison.OrdinalIgnoreCase))
        {
            whamVideoSurface = surface;
            whamVideoViewport = viewport;
            whamPlaceholderText = placeholder;
        }
        else if (string.Equals(videoObjectName, "GMR", StringComparison.OrdinalIgnoreCase))
        {
            gmrVideoSurface = surface;
            gmrVideoViewport = viewport;
            gmrPlaceholderText = placeholder;
            PushGmrRenderSizeToStartInput(viewport);
        }

        RectTransform status = CreatePanel("Status", panel, new Color(0.02f, 0.04f, 0.06f, 0.55f));
        StretchHorizontalBottom(status, 28f, 28f, 18f, 28f);
        TMP_Text statusText = AddText(status, "Text", $"{videoObjectName} | waiting", 13, FontStyles.Normal, new Vector2(10f, -4f), new Vector2(980f, 20f));
        ConfigureSingleLine(statusText);
        if (string.Equals(videoObjectName, "WHAM", StringComparison.OrdinalIgnoreCase))
        {
            whamStatusText = statusText;
        }
        else if (string.Equals(videoObjectName, "GMR", StringComparison.OrdinalIgnoreCase))
        {
            gmrStatusText = statusText;
        }

        CanvasGroup group = panel.gameObject.AddComponent<CanvasGroup>();
        group.alpha = 1f;
        group.blocksRaycasts = true;
        group.interactable = true;
        return group;
    }

    private CanvasGroup CreateParamsDrawer()
    {
        RectTransform panel = CreatePanel("PARAMSDrawer", hudRoot, DrawerShell, blur: true);
        panel.anchorMin = new Vector2(0.5f, 0.5f);
        panel.anchorMax = new Vector2(0.5f, 0.5f);
        panel.pivot = new Vector2(0.5f, 0.5f);
        panel.anchoredPosition = Vector2.zero;
        panel.sizeDelta = new Vector2(1120f, 900f);

        RectTransform header = CreatePanel("Header", panel, new Color(0.62f, 0.82f, 0.94f, 0.32f));
        StretchHorizontalTop(header, 0f, 0f, 0f, 104f);
        header.gameObject.AddComponent<HudDragHandle>().Initialize(panel, hudRoot);
        AddText(header, "Title", "Start Parameters", 30, FontStyles.Bold, new Vector2(28f, -20f), new Vector2(720f, 38f));
        AddText(header, "Subtitle", "Run.ps1 / WHAM / GMR controls", 17, FontStyles.Normal, new Vector2(28f, -62f), new Vector2(780f, 28f));
        CreateCloseButton(header, () => HideDrawer(panel.GetComponent<CanvasGroup>()));

        RectTransform viewport = CreatePanel("ParamsViewport", panel, new Color(0f, 0f, 0f, 0.08f));
        Stretch(viewport, 28f, 26f, 28f, 118f);
        var mask = viewport.gameObject.AddComponent<Mask>();
        mask.showMaskGraphic = false;
        var scroll = viewport.gameObject.AddComponent<ScrollRect>();
        scroll.horizontal = false;
        scroll.vertical = true;
        scroll.scrollSensitivity = 35f;

        RectTransform content = CreateRect("ParamsContent", viewport);
        content.anchorMin = new Vector2(0f, 1f);
        content.anchorMax = new Vector2(1f, 1f);
        content.pivot = new Vector2(0.5f, 1f);
        content.offsetMin = Vector2.zero;
        content.offsetMax = Vector2.zero;
        scroll.content = content;
        scroll.viewport = viewport;

        BuildParamsContent(content);
        CanvasGroup group = panel.gameObject.AddComponent<CanvasGroup>();
        return group;
    }

    private void BuildParamsContent(RectTransform content)
    {
        paramStaticRows.Clear();
        paramGroups.Clear();
        if (startInput == null)
        {
            AddText(content, "MissingStartInput", "StartInput not found in scene.", 22, FontStyles.Bold, new Vector2(24f, -24f), new Vector2(900f, 40f));
            content.sizeDelta = new Vector2(0f, 120f);
            return;
        }

        float y = -16f;
        CreateInfoRow(content, "Most Python parameters apply on next Start. Stop and Start again after changing them.", ref y);
        ParamGroup sourceOutput = CreateGroupHeader(content, "Source / Output", ref y);
        CreateFieldRows(content, sourceOutput, ref y, "videoPath", "outputRoot", "outputCsvFileName", "recordWhamVideo", "recordGmrVideo", "enableTcpStreaming", "disablePreviewVideoWhenTcpStreaming", "track");
        ParamGroup gmrViewer = CreateGroupHeader(content, "GMR / Viewer", ref y);
        CreateFieldRows(content, gmrViewer, ref y, "gmrCameraFollow", "gmrCameraLookatHeightOffset", "gmrCameraElevation", "gmrCameraDistanceScale", "gmrCameraAzimuth", "gmrTcpRenderWidth", "gmrTcpRenderHeight");
        ParamGroup whamPerformance = CreateGroupHeader(content, "WHAM Performance", ref y);
        CreateFieldRows(content, whamPerformance, ref y, "whamDetectInterval", "whamInferInterval", "whamStreamSeqLen", "whamInputScale");
        ParamGroup runtimeLogging = CreateGroupHeader(content, "Runtime / Logging", ref y);
        CreateFieldRows(content, runtimeLogging, ref y, "gmrTorchDevice", "gmrMaxIter", "gmrCsvFlushInterval", "pipelineHeartbeatFrames");
        ReflowParamsContent(content);
    }

    private void CreateInfoRow(RectTransform parent, string message, ref float y)
    {
        RectTransform row = CreatePanel("Info", parent, new Color(0.06f, 0.12f, 0.18f, 0.82f));
        StretchRow(row, y, 70f, 18f);
        AddText(row, "Text", message, 20, FontStyles.Bold, new Vector2(22f, -16f), new Vector2(990f, 34f));
        paramStaticRows.Add(row);
        y -= 86f;
    }

    private ParamGroup CreateGroupHeader(RectTransform parent, string title, ref float y)
    {
        RectTransform row = CreatePanel("Group_" + title.Replace(" ", string.Empty), parent, new Color(0.06f, 0.12f, 0.18f, 0.90f));
        StretchRow(row, y, 58f, 42f);
        AddText(row, "Title", title, 24, FontStyles.Bold, new Vector2(18f, -12f), new Vector2(760f, 34f));
        TMP_Text chevron = AddText(row, "Chevron", "v", 26, FontStyles.Bold, new Vector2(980f, -12f), new Vector2(46f, 34f), TextAlignmentOptions.Center);
        var group = new ParamGroup { Header = row, Chevron = chevron, Expanded = true };
        Image rowImage = row.GetComponent<Image>();
        if (rowImage != null)
        {
            rowImage.raycastTarget = true;
        }
        Button button = row.gameObject.AddComponent<Button>();
        button.targetGraphic = rowImage;
        button.transition = Selectable.Transition.ColorTint;
        button.colors = BuildSelectableColors(rowImage != null ? rowImage.color : new Color(0.06f, 0.12f, 0.18f, 0.90f));
        button.navigation = new Navigation { mode = Navigation.Mode.None };
        button.onClick.AddListener(() =>
        {
            group.Expanded = !group.Expanded;
            ReflowParamsContent(parent);
        });
        paramGroups.Add(group);
        y -= 70f;
        return group;
    }

    private void CreateFieldRows(RectTransform parent, ParamGroup group, ref float y, params string[] fieldNames)
    {
        foreach (string fieldName in fieldNames)
        {
            FieldInfo field = typeof(StartInput).GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            if (field != null)
            {
                RectTransform row = CreateParamRow(parent, Nicify(fieldName), field, ref y);
                group?.Rows.Add(row);
            }
        }
    }

    private RectTransform CreateParamRow(RectTransform parent, string label, FieldInfo field, ref float y)
    {
        RectTransform row = CreatePanel("Param_" + field.Name, parent, new Color(0.12f, 0.19f, 0.25f, 0.62f));
        StretchRow(row, y, 66f, 42f);
        AddText(row, "Label", label, 21, FontStyles.Bold, new Vector2(18f, -16f), new Vector2(430f, 34f));

        if (field.FieldType == typeof(bool))
        {
            Toggle toggle = CreateBooleanToggle(row, field);
            paramControls.Add(new ParamControl { Field = field, Toggle = toggle });
        }
        else
        {
            TMP_InputField input = CreateInputField(row, field);
            paramControls.Add(new ParamControl { Field = field, Input = input });
        }

        y -= 78f;
        return row;
    }

    private void ReflowParamsContent(RectTransform content)
    {
        if (content == null)
        {
            return;
        }

        float y = -16f;
        for (int i = 0; i < paramStaticRows.Count; i++)
        {
            RectTransform row = paramStaticRows[i];
            if (row == null) continue;
            row.gameObject.SetActive(true);
            row.anchoredPosition = new Vector2(0f, y);
            y -= Mathf.Abs(row.sizeDelta.y) + 16f;
        }

        for (int i = 0; i < paramGroups.Count; i++)
        {
            ParamGroup group = paramGroups[i];
            if (group == null || group.Header == null) continue;

            group.Header.gameObject.SetActive(true);
            group.Header.anchoredPosition = new Vector2(0f, y);
            if (group.Chevron != null)
            {
                group.Chevron.text = group.Expanded ? "v" : ">";
            }
            y -= Mathf.Abs(group.Header.sizeDelta.y) + 12f;

            for (int rowIndex = 0; rowIndex < group.Rows.Count; rowIndex++)
            {
                RectTransform row = group.Rows[rowIndex];
                if (row == null) continue;

                row.gameObject.SetActive(group.Expanded);
                if (!group.Expanded)
                {
                    continue;
                }

                row.anchoredPosition = new Vector2(0f, y);
                y -= Mathf.Abs(row.sizeDelta.y) + 12f;
            }
        }

        content.sizeDelta = new Vector2(content.sizeDelta.x, -y + 28f);
    }

    private Toggle CreateBooleanToggle(RectTransform row, FieldInfo field)
    {
        RectTransform box = CreatePanel("Toggle", row, new Color(0.16f, 0.24f, 0.34f, 0.95f));
        box.anchorMin = new Vector2(1f, 0.5f);
        box.anchorMax = new Vector2(1f, 0.5f);
        box.pivot = new Vector2(1f, 0.5f);
        box.anchoredPosition = new Vector2(-22f, 0f);
        box.sizeDelta = new Vector2(42f, 42f);
        RectTransform check = CreatePanel("Checkmark", box, new Color(0.08f, 0.44f, 0.95f, 0.95f));
        Stretch(check, 7f, 7f, 7f, 7f);

        Toggle toggle = row.gameObject.AddComponent<Toggle>();
        toggle.targetGraphic = row.GetComponent<Image>();
        toggle.graphic = check.GetComponent<Image>();
        toggle.transition = Selectable.Transition.ColorTint;
        bool initialValue = ReadRuntimeParameterBool(field);
        toggle.SetIsOnWithoutNotify(initialValue);
        toggle.onValueChanged.AddListener(value =>
        {
            string error = string.Empty;
            if (startInput == null || !startInput.TrySetRuntimeParameter(field.Name, value, out error))
            {
                if (!string.IsNullOrWhiteSpace(error))
                {
                    Debug.LogWarning($"[RetargetingHudLayout] {error}");
                }

                toggle.SetIsOnWithoutNotify(ReadRuntimeParameterBool(field));
            }
        });
        return toggle;
    }

    private TMP_InputField CreateInputField(RectTransform row, FieldInfo field)
    {
        RectTransform root = CreatePanel("Input", row, new Color(0.12f, 0.20f, 0.29f, 0.95f));
        root.anchorMin = new Vector2(0f, 0f);
        root.anchorMax = new Vector2(1f, 1f);
        root.offsetMin = new Vector2(500f, 9f);
        root.offsetMax = new Vector2(-20f, -9f);
        Image inputImage = root.GetComponent<Image>();
        Color normalColor = inputImage != null ? inputImage.color : new Color(0.12f, 0.20f, 0.29f, 0.95f);
        Color focusColor = new Color(0.18f, 0.34f, 0.48f, 0.98f);

        TMP_InputField input = root.gameObject.AddComponent<TMP_InputField>();
        input.targetGraphic = inputImage;
        input.transition = Selectable.Transition.None;
        input.lineType = TMP_InputField.LineType.SingleLine;
        input.text = ReadRuntimeParameterString(field);
        input.customCaretColor = true;
        input.caretColor = Color.white;
        input.caretWidth = 3;
        input.selectionColor = new Color(0.35f, 0.72f, 1.0f, 0.45f);

        RectTransform viewport = CreateRect("TextViewport", root);
        Stretch(viewport, 12f, 4f, 12f, 4f);
        viewport.gameObject.AddComponent<RectMask2D>();

        TMP_Text text = CreateInputText("Text", viewport, input.text, new Color(0.94f, 0.98f, 1f, 1f));
        TMP_Text placeholder = CreateInputText("Placeholder", viewport, "Value", new Color(0.74f, 0.82f, 0.90f, 0.55f));
        placeholder.enabled = string.IsNullOrEmpty(input.text);

        input.textViewport = viewport;
        input.textComponent = text;
        input.placeholder = placeholder;
        input.SetTextWithoutNotify(input.text);
        input.onSelect.AddListener(_ =>
        {
            if (inputImage != null)
            {
                inputImage.color = focusColor;
            }
        });
        input.onDeselect.AddListener(value =>
        {
            if (inputImage != null)
            {
                inputImage.color = normalColor;
            }

            TrySetFieldValue(field, value, live: false);
        });
        input.onValueChanged.AddListener(value =>
        {
            if (placeholder != null)
            {
                placeholder.enabled = string.IsNullOrEmpty(value);
            }

            TrySetFieldValue(field, value, live: true);
        });
        input.onEndEdit.AddListener(value => TrySetFieldValue(field, value, live: false));
        return input;
    }

    private bool ReadRuntimeParameterBool(FieldInfo field)
    {
        if (field == null)
        {
            return false;
        }

        if (startInput != null &&
            startInput.TryGetRuntimeParameterValue(field.Name, out object runtimeValue) &&
            runtimeValue is bool boolValue)
        {
            return boolValue;
        }

        return field.FieldType == typeof(bool) && startInput != null && (bool)field.GetValue(startInput);
    }

    private string ReadRuntimeParameterString(FieldInfo field)
    {
        if (field == null)
        {
            return string.Empty;
        }

        if (startInput != null && startInput.TryGetRuntimeParameterValue(field.Name, out object runtimeValue))
        {
            return Convert.ToString(runtimeValue, CultureInfo.InvariantCulture) ?? string.Empty;
        }

        return startInput != null
            ? Convert.ToString(field.GetValue(startInput), CultureInfo.InvariantCulture) ?? string.Empty
            : string.Empty;
    }

    private TMP_Text CreateInputText(string name, RectTransform parent, string textValue, Color color)
    {
        RectTransform rect = CreateRect(name, parent);
        Stretch(rect);
        TextMeshProUGUI text = rect.gameObject.AddComponent<TextMeshProUGUI>();
        text.text = textValue;
        text.fontSize = 20f;
        text.fontStyle = FontStyles.Bold;
        text.alignment = TextAlignmentOptions.MidlineLeft;
        text.color = color;
        text.enableWordWrapping = false;
        text.raycastTarget = false;
        ownedGraphics.Add(text);
        return text;
    }

    private void TrySetFieldValue(FieldInfo field, string value, bool live)
    {
        if (startInput == null || field == null)
        {
            return;
        }

        string error = string.Empty;
        if (startInput.TrySetRuntimeParameter(field.Name, value, out error))
        {
            return;
        }

        if (!live)
        {
            Debug.LogWarning($"[RetargetingHudLayout] {error}");
        }
    }

    private void StyleButton(GameObject go, string label, Color normalColor, bool lockable)
    {
        if (go == null)
        {
            return;
        }

        RectTransform rect = EnsureRectTransform(go);
        Image image = go.GetComponent<Image>() ?? go.AddComponent<Image>();
        image.enabled = true;
        image.sprite = null;
        image.type = Image.Type.Simple;
        image.fillCenter = true;
        image.preserveAspect = false;
        image.color = normalColor;
        image.raycastTarget = true;
        ownedGraphics.Add(image);

        Button button = go.GetComponent<Button>() ?? go.AddComponent<Button>();
        button.targetGraphic = image;
        button.transition = Selectable.Transition.None;
        if (lockable && !lockableButtons.Contains(button))
        {
            lockableButtons.Add(button);
        }

        foreach (Graphic graphic in go.GetComponentsInChildren<Graphic>(true))
        {
            if (graphic == image)
            {
                continue;
            }

            if (graphic.transform.name == "HudText")
            {
                continue;
            }

            graphic.raycastTarget = false;
            graphic.enabled = false;
        }

        TMP_Text text = go.transform.Find("HudText")?.GetComponent<TMP_Text>();
        if (text == null)
        {
            RectTransform textRect = CreateRect("HudText", rect);
            Stretch(textRect);
            text = textRect.gameObject.AddComponent<TextMeshProUGUI>();
        }

        text.enabled = true;
        text.text = label;
        text.fontSize = label == "PARAMS" ? 24f : 26f;
        text.fontStyle = FontStyles.Bold;
        text.alignment = TextAlignmentOptions.Center;
        text.color = Color.white;
        text.raycastTarget = false;
        ownedGraphics.Add(text);

        HudButtonFeedback feedback = go.GetComponent<HudButtonFeedback>() ?? go.AddComponent<HudButtonFeedback>();
        feedback.SetColors(normalColor);
    }

    private Button CreateButton(string name, RectTransform parent, string label, Color color)
    {
        RectTransform rect = CreatePanel(name, parent, color);
        Button button = rect.gameObject.AddComponent<Button>();
        StyleButton(rect.gameObject, label, color, lockable: false);
        return button;
    }

    private void StyleDropdown(GameObject go)
    {
        if (go == null)
        {
            return;
        }

        Image image = go.GetComponent<Image>() ?? go.AddComponent<Image>();
        image.enabled = true;
        image.sprite = null;
        image.type = Image.Type.Simple;
        image.fillCenter = true;
        image.preserveAspect = false;
        image.color = DropdownCaption;
        image.raycastTarget = true;
        ownedGraphics.Add(image);

        TMP_Dropdown dropdown = go.GetComponent<TMP_Dropdown>();
        if (dropdown == null)
        {
            return;
        }

        if (!lockableDropdowns.Contains(dropdown))
        {
            lockableDropdowns.Add(dropdown);
        }

        dropdown.targetGraphic = image;
        dropdown.colors = BuildSelectableColors(DropdownCaption);

        if (dropdown.captionText != null)
        {
            dropdown.captionText.fontSize = 22;
            dropdown.captionText.color = Color.white;
            dropdown.captionText.fontStyle = FontStyles.Bold;
            dropdown.captionText.raycastTarget = false;
        }

        if (dropdown.itemText != null)
        {
            dropdown.itemText.fontSize = 20;
            dropdown.itemText.color = Color.white;
            dropdown.itemText.fontStyle = FontStyles.Normal;
        }

        if (dropdown.template == null)
        {
            return;
        }

        dropdown.template.sizeDelta = new Vector2(dropdown.template.sizeDelta.x, 430f);
        Image templateImage = dropdown.template.GetComponent<Image>() ?? dropdown.template.gameObject.AddComponent<Image>();
        templateImage.color = DropdownList;
        templateImage.raycastTarget = true;

        foreach (TMP_Text text in dropdown.template.GetComponentsInChildren<TMP_Text>(true))
        {
            text.color = Color.white;
            text.fontSize = Mathf.Max(text.fontSize, 20f);
            text.fontStyle = FontStyles.Normal;
            text.raycastTarget = false;
        }

        foreach (Image childImage in dropdown.template.GetComponentsInChildren<Image>(true))
        {
            if (childImage == templateImage)
            {
                continue;
            }

            if (childImage.GetComponentInParent<Scrollbar>(true) != null)
            {
                continue;
            }

            childImage.color = new Color(0.02f, 0.06f, 0.11f, 0.92f);
        }

        foreach (Scrollbar scrollbar in dropdown.template.GetComponentsInChildren<Scrollbar>(true))
        {
            Image track = scrollbar.GetComponent<Image>();
            if (track != null)
            {
                track.color = new Color(0.07f, 0.12f, 0.17f, 0.92f);
            }

            if (scrollbar.handleRect != null)
            {
                Image handle = scrollbar.handleRect.GetComponent<Image>() ?? scrollbar.handleRect.gameObject.AddComponent<Image>();
                handle.color = new Color(0.72f, 0.88f, 0.94f, 0.96f);
                scrollbar.targetGraphic = handle;
            }
        }
    }

    private void SyncRealtimeUiState()
    {
        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>(true);
        }

        bool locked = startInput != null && startInput.IsRealtimeControlsLocked;
        for (int i = 0; i < lockableButtons.Count; i++)
        {
            if (lockableButtons[i] != null)
            {
                lockableButtons[i].interactable = !locked;
            }
        }

        for (int i = 0; i < lockableDropdowns.Count; i++)
        {
            if (lockableDropdowns[i] != null)
            {
                lockableDropdowns[i].interactable = !locked;
            }
        }

        for (int i = 0; i < paramControls.Count; i++)
        {
            ParamControl control = paramControls[i];
            if (control.Input != null)
            {
                control.Input.interactable = !locked;
            }

            if (control.Toggle != null)
            {
                control.Toggle.interactable = !locked;
            }
        }
    }

    private void UpdateProgressPanel()
    {
        if (progressGroup == null)
        {
            return;
        }

        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>(true);
        }

        bool show = startInput != null && (startInput.PipelineStartupActive || startInput.HasPipelineError);
        progressGroup.alpha = show ? 1f : 0f;
        progressGroup.blocksRaycasts = false;
        progressGroup.interactable = false;
        if (!show)
        {
            return;
        }

        if (progressTitle != null)
        {
            progressTitle.text = startInput.HasPipelineError ? "Retargeting failed" : "Starting retargeting";
        }

        if (progressActivity != null)
        {
            progressActivity.text = startInput.HasPipelineError ? "Startup error" : CompactHudLine(startInput.PipelineStatusText, 96);
        }

        if (progressDetail != null)
        {
            progressDetail.text = CompactHudLine(startInput.HasPipelineError ? startInput.LastPipelineError : startInput.PipelineActivityText, 132);
        }

        if (progressFill != null)
        {
            RectTransform fillRect = progressFill.rectTransform;
            float parentWidth = ((RectTransform)fillRect.parent).rect.width;
            float progress = startInput.HasPipelineError ? 0f : Mathf.Clamp01(startInput.PipelineProgress01);
            fillRect.sizeDelta = new Vector2(parentWidth * progress, 0f);
            progressFill.color = startInput.HasPipelineError
                ? new Color(0.95f, 0.16f, 0.12f, 0.95f)
                : new Color(0.08f, 0.72f, 0.96f, 0.95f);
        }
    }

    private static StreamReceiver ResolveStreamReceiverForHud()
    {
        return Application.isPlaying
            ? StreamReceiver.EnsureReceiverHost()
            : FindObjectOfType<StreamReceiver>(true);
    }

    private void UpdateStreamReceiverStatus()
    {
        if (streamReceiver == null)
        {
            streamReceiver = ResolveStreamReceiverForHud();
            BindStreamReceiverTargets();
        }

        if (streamReceiver != null)
        {
            if (whamStatusText != null)
            {
                whamStatusText.text = streamReceiver.GetStatusLine("WHAM", 0);
            }

            if (gmrStatusText != null)
            {
                gmrStatusText.text = streamReceiver.GetStatusLine("GMR", 1);
            }
        }

        if (whamPlaceholderText != null && whamVideoSurface != null)
        {
            whamPlaceholderText.enabled = whamVideoSurface.texture == null;
        }

        if (gmrPlaceholderText != null && gmrVideoSurface != null)
        {
            gmrPlaceholderText.enabled = gmrVideoSurface.texture == null;
        }
    }

    private void BindStreamReceiverTargets()
    {
        if (streamReceiver == null)
        {
            streamReceiver = ResolveStreamReceiverForHud();
        }

        if (streamReceiver != null && whamVideoSurface != null && gmrVideoSurface != null)
        {
            streamReceiver.SetHudVideoDisplayOptions(hudManagedAspectFit: true, placeholder: new Color(1f, 1f, 1f, 0f));
            streamReceiver.ConfigureTargets(whamVideoSurface, gmrVideoSurface);
        }
    }

    private void ShowDrawer(CanvasGroup drawer)
    {
        if (drawer == null)
        {
            return;
        }

        HideDrawer(whamDrawer);
        HideDrawer(gmrDrawer);
        HideDrawer(paramsDrawer);
        drawer.alpha = 1f;
        drawer.interactable = true;
        drawer.blocksRaycasts = true;
        drawer.transform.SetAsLastSibling();
    }

    private static void HideDrawer(CanvasGroup drawer)
    {
        if (drawer == null)
        {
            return;
        }

        drawer.alpha = 0f;
        drawer.interactable = false;
        drawer.blocksRaycasts = false;
    }

    private void CreateCloseButton(RectTransform header, UnityEngine.Events.UnityAction action)
    {
        Button close = CreateButton("Close", header, "X", CloseColor);
        RectTransform rect = (RectTransform)close.transform;
        rect.anchorMin = new Vector2(1f, 1f);
        rect.anchorMax = new Vector2(1f, 1f);
        rect.pivot = new Vector2(1f, 1f);
        rect.anchoredPosition = new Vector2(-18f, -18f);
        rect.sizeDelta = new Vector2(62f, 62f);
        close.onClick.AddListener(action);
    }

    private Button CreateProxyButton(
        string proxyName,
        string[] sourceNames,
        RectTransform parent,
        Vector2 anchoredPosition,
        Vector2 size,
        string label,
        Color color,
        bool lockable,
        HashSet<Transform> movedRoots,
        UnityEngine.Events.UnityAction fallbackAction)
    {
        GameObject sourceObject = ResolveHudObjectRoot(sourceNames, preferInteractiveRoot: true);
        Button sourceButton = sourceObject != null ? sourceObject.GetComponent<Button>() : null;
        if (sourceObject != null)
        {
            HideLegacyInteractiveObject(sourceObject);
            movedRoots.Add(sourceObject.transform);
        }

        Button proxy = CreateButton(proxyName, parent, label, color);
        AnchorTopLeft((RectTransform)proxy.transform, anchoredPosition, size);
        if (lockable && !lockableButtons.Contains(proxy))
        {
            lockableButtons.Add(proxy);
        }

        proxy.onClick.AddListener(() =>
        {
            if (sourceButton != null)
            {
                sourceButton.onClick.Invoke();
                if (fallbackAction == null || sourceButton.onClick.GetPersistentEventCount() > 0)
                {
                    return;
                }
            }

            fallbackAction?.Invoke();
        });
        return proxy;
    }

    private static bool TryParseBool(string value, out bool result)
    {
        if (bool.TryParse(value, out result))
        {
            return true;
        }

        string normalized = (value ?? string.Empty).Trim();
        if (normalized == "1" || normalized.Equals("yes", StringComparison.OrdinalIgnoreCase) || normalized.Equals("on", StringComparison.OrdinalIgnoreCase))
        {
            result = true;
            return true;
        }

        if (normalized == "0" || normalized.Equals("no", StringComparison.OrdinalIgnoreCase) || normalized.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            result = false;
            return true;
        }

        result = false;
        return false;
    }

    private static void OpenHomeScene()
    {
        string[] candidates = { "GewuMenu", "Home", "Menu", "MainMenu", "StartMenu" };
        for (int i = 0; i < candidates.Length; i++)
        {
            if (Application.CanStreamedLevelBeLoaded(candidates[i]))
            {
                SceneManager.LoadScene(candidates[i]);
                return;
            }
        }

#if UNITY_EDITOR
        string[] editorScenePaths = { "Assets/GewuMenu.unity", "Assets/GewuMenu/NewScene.unity" };
        for (int i = 0; i < editorScenePaths.Length; i++)
        {
            string absolutePath = System.IO.Path.Combine(
                System.IO.Directory.GetParent(Application.dataPath)?.FullName ?? string.Empty,
                editorScenePaths[i]);
            if (System.IO.File.Exists(absolutePath))
            {
                UnityEditor.SceneManagement.EditorSceneManager.LoadSceneInPlayMode(
                    editorScenePaths[i],
                    new LoadSceneParameters(LoadSceneMode.Single));
                return;
            }
        }
#endif

        Scene current = SceneManager.GetActiveScene();
        if (SceneManager.sceneCountInBuildSettings > 0 && current.buildIndex != 0)
        {
            SceneManager.LoadScene(0);
            return;
        }

        Debug.LogWarning("[RetargetingHudLayout] Home button clicked, but no Home/Menu scene is available in Build Settings.");
    }

    private static void HideLegacyInteractiveObject(GameObject sourceObject)
    {
        if (sourceObject == null)
        {
            return;
        }

        foreach (Graphic graphic in sourceObject.GetComponentsInChildren<Graphic>(true))
        {
            graphic.raycastTarget = false;
            graphic.enabled = false;
        }

        Button button = sourceObject.GetComponent<Button>();
        if (button != null)
        {
            button.transition = Selectable.Transition.None;
            button.targetGraphic = null;
        }
    }

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

        if (canvasObject.GetComponent<GraphicRaycaster>() == null)
        {
            canvasObject.AddComponent<GraphicRaycaster>();
        }
    }

    private static void EnsureEventSystem()
    {
        if (FindObjectOfType<EventSystem>(true) != null)
        {
            return;
        }

        var go = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
    }

    private static RectTransform CreateRect(string name, Transform parent)
    {
        var go = new GameObject(name, typeof(RectTransform));
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

    private static void DetachPreservedHudObjects(Transform oldRoot, Transform fallbackParent)
    {
        string[] names =
        {
            "Home", "RoboList", "CsvList", "StartButton", "Start", "ReplayButton", "Replay",
            "StopButton", "Stop", "SwitchCameraButton", "HomeButton", "WHAM", "GMR"
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
