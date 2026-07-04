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
        viewport.gameObject.AddComponent<RectMask2D>();
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
        input.caretWidth = 4;
        input.caretBlinkRate = 0.8f;
        input.selectionColor = new Color(0.35f, 0.72f, 1.0f, 0.45f);
        input.resetOnDeActivation = false;
        input.restoreOriginalTextOnEscape = false;

        RectTransform viewport = CreateRect("TextViewport", root);
        Stretch(viewport, 12f, 4f, 12f, 4f);
        viewport.gameObject.AddComponent<RectMask2D>();

        TMP_Text text = CreateInputText("Text", viewport, input.text, new Color(0.94f, 0.98f, 1f, 1f), raycast: true);
        TMP_Text placeholder = CreateInputText("Placeholder", viewport, "Value", new Color(0.74f, 0.82f, 0.90f, 0.55f), raycast: false);
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

            input.ActivateInputField();
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
        input.ForceLabelUpdate();
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

    private TMP_Text CreateInputText(string name, RectTransform parent, string textValue, Color color, bool raycast = false)
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
        text.raycastTarget = raycast;
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

}
