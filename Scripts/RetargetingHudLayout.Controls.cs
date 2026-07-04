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
    private void BuildControlDock(HashSet<Transform> movedRoots)
    {
        RectTransform controlDock = CreatePanel("ControlDock", hudRoot, PanelBlue, blur: true);
        AnchorTopLeft(controlDock, new Vector2(24f, -18f), new Vector2(420f, 800f));
        controlDock.gameObject.AddComponent<HudDragHandle>().Initialize(controlDock, hudRoot);

        CreateHudCommandButton("HomeHudButton", controlDock, new Vector2(32f, -24f), new Vector2(356f, 72f), "Home", HomeColor, false, OpenHomeScene);

        AddText(controlDock, "Title", "Retargeting", 32, FontStyles.Bold, new Vector2(32f, -128f), new Vector2(356f, 40f));
        AddText(controlDock, "Subtitle", "Robot imitation control", 15, FontStyles.Normal, new Vector2(32f, -164f), new Vector2(356f, 24f));

        AddText(controlDock, "RobotLabel", "Robot", 20, FontStyles.Bold, new Vector2(32f, -214f), new Vector2(128f, 28f));
        AddText(controlDock, "CsvLabel", "Motion / CSV", 20, FontStyles.Bold, new Vector2(188f, -214f), new Vector2(200f, 28f));
        BuildRobotSelectionRows(controlDock);

        CreateHudCommandButton("StartHudButton", controlDock, new Vector2(32f, -500f), new Vector2(356f, 64f), "Start", StartColor, true, InvokeStartInput);
        CreateHudCommandButton("ReplayHudButton", controlDock, new Vector2(32f, -586f), new Vector2(164f, 58f), "Replay", ReplayColor, true, InvokeReplay);
        CreateHudCommandButton("StopHudButton", controlDock, new Vector2(224f, -586f), new Vector2(164f, 58f), "Stop", StopColor, false, InvokeStop);

        Button switchCamera = CreateButton("SwitchCameraButton", controlDock, "Switch Camera", SwitchColor);
        AnchorTopLeft((RectTransform)switchCamera.transform, new Vector2(32f, -662f), new Vector2(164f, 52f));
        switchCamera.onClick.AddListener(SelectedRobotCameraFollow.SwitchNextView);

        Button switchRobot = CreateButton("SwitchRobotButton", controlDock, "Switch Robot", ParamsColor);
        AnchorTopLeft((RectTransform)switchRobot.transform, new Vector2(224f, -662f), new Vector2(164f, 52f));
        switchRobot.onClick.AddListener(SelectedRobotCameraFollow.SwitchNextRobotTarget);

        cameraStatusText = AddText(controlDock, "CameraStatus", SelectedRobotCameraFollow.GetCurrentCameraStatusText(), 14, FontStyles.Bold, new Vector2(32f, -730f), new Vector2(356f, 28f), TextAlignmentOptions.Center);
        ConfigureSingleLine(cameraStatusText);
    }

    private void BuildRobotSelectionRows(RectTransform parent)
    {
        for (int i = 0; i < RobotHudEntries.Length; i++)
        {
            RobotHudEntry entry = RobotHudEntries[i];
            RectTransform row = CreatePanel("RobotRow_" + entry.Key, parent, new Color(0.03f, 0.08f, 0.12f, 0.34f));
            AnchorTopLeft(row, new Vector2(32f, -250f - i * 58f), new Vector2(356f, 50f));

            CreateRobotToggle(row, entry);
            TMP_Text label = AddText(row, "Label", entry.Label, 15, FontStyles.Bold, new Vector2(42f, -12f), new Vector2(108f, 24f));
            ConfigureSingleLine(label);
            CreateRobotCsvDropdown(row, entry);
        }
    }

    private Toggle CreateRobotToggle(RectTransform row, RobotHudEntry entry)
    {
        RectTransform box = CreatePanel("Selected", row, new Color(0.02f, 0.08f, 0.11f, 0.95f));
        AnchorTopLeft(box, new Vector2(10f, -10f), new Vector2(24f, 24f));
        Image boxImage = box.GetComponent<Image>();
        if (boxImage != null)
        {
            boxImage.raycastTarget = false;
        }

        RectTransform check = CreatePanel("Checkmark", box, new Color(0.00f, 0.92f, 0.42f, 0.95f));
        Stretch(check, 5f, 5f, 5f, 5f);
        Image checkImage = check.GetComponent<Image>();
        if (checkImage != null)
        {
            checkImage.raycastTarget = false;
        }

        Toggle toggle = row.gameObject.AddComponent<Toggle>();
        toggle.targetGraphic = row.GetComponent<Image>();
        toggle.graphic = check.GetComponent<Image>();
        toggle.transition = Selectable.Transition.ColorTint;

        bool selected = startInput != null ? startInput.IsRobotSelected(entry.Key) : entry.SelectedByDefault;
        toggle.SetIsOnWithoutNotify(selected);
        check.gameObject.SetActive(selected);
        if (selected && startInput != null)
        {
            startInput.SetRobotSelected(entry.Key, true);
        }

        toggle.onValueChanged.AddListener(value =>
        {
            check.gameObject.SetActive(value);
            if (startInput == null)
            {
                startInput = FindObjectOfType<StartInput>(true);
            }

            if (startInput != null)
            {
                startInput.SetRobotSelected(entry.Key, value);
            }
        });

        lockableToggles.Add(toggle);
        return toggle;
    }

    private TMP_Dropdown CreateRobotCsvDropdown(RectTransform row, RobotHudEntry entry)
    {
        TMP_Dropdown dropdown = CreateRuntimeDropdown("CsvList_" + entry.Key, row);
        GameObject dropdownObject = dropdown.gameObject;
        RectTransform rect = (RectTransform)dropdown.transform;
        AnchorTopLeft(rect, new Vector2(156f, -6f), new Vector2(190f, 38f));

        FileBrowser browser = dropdownObject.AddComponent<FileBrowser>();
        browser.dropdownMode = FileBrowser.DropdownMode.CsvFiles;
        browser.folderPath = "Assets/Imitation/dataset/" + ResolveRobotDatasetFolder(entry.Key);
        browser.fallbackFolderPaths = new List<string>
        {
            "Assets/Gewu/Imitation/dataset/" + ResolveRobotDatasetFolder(entry.Key),
            "Assets/Imitation/dataset/" + ResolveRobotDatasetFolder(entry.Key),
        };
        browser.searchPattern = "*.csv";
        browser.includeSubfolders = false;
        browser.preserveManualOptionsOnFailure = false;
        browser.PopulateDropdown();

        StyleDropdown(dropdownObject);
        EnsureDropdownHasVisibleCaption(dropdown);

        if (startInput != null)
        {
            startInput.RegisterRobotCsvBrowser(entry.Key, browser, dropdown);
        }

        EnsureDropdownHasVisibleCaption(dropdown);
        return dropdown;
    }

    private TMP_Dropdown CreateRuntimeDropdown(string name, RectTransform parent)
    {
        RectTransform root = CreatePanel(name, parent, DropdownCaption);
        TMP_Dropdown dropdown = root.gameObject.AddComponent<TMP_Dropdown>();

        TMP_Text caption = AddText(root, "Label", string.Empty, 18, FontStyles.Bold, new Vector2(10f, -6f), new Vector2(146f, 26f));
        ConfigureSingleLine(caption);

        TMP_Text arrow = AddText(root, "Arrow", "v", 16, FontStyles.Bold, new Vector2(162f, -7f), new Vector2(20f, 24f), TextAlignmentOptions.Center);
        ConfigureSingleLine(arrow);

        RectTransform template = CreatePanel("Template", root, DropdownList);
        template.anchorMin = new Vector2(0f, 1f);
        template.anchorMax = new Vector2(1f, 1f);
        template.pivot = new Vector2(0.5f, 1f);
        template.anchoredPosition = new Vector2(0f, -40f);
        template.sizeDelta = new Vector2(0f, 260f);
        template.gameObject.SetActive(false);

        RectTransform viewport = CreatePanel("Viewport", template, new Color(1f, 1f, 1f, 0.04f));
        Stretch(viewport, 0f, 0f, 0f, 0f);
        Mask mask = viewport.gameObject.AddComponent<Mask>();
        mask.showMaskGraphic = false;

        RectTransform content = CreateRect("Content", viewport);
        content.anchorMin = new Vector2(0f, 1f);
        content.anchorMax = new Vector2(1f, 1f);
        content.pivot = new Vector2(0.5f, 1f);
        content.anchoredPosition = Vector2.zero;
        content.sizeDelta = new Vector2(0f, 34f);

        RectTransform item = CreatePanel("Item", content, new Color(0.02f, 0.06f, 0.11f, 0.92f));
        item.anchorMin = new Vector2(0f, 1f);
        item.anchorMax = new Vector2(1f, 1f);
        item.pivot = new Vector2(0.5f, 1f);
        item.anchoredPosition = Vector2.zero;
        item.sizeDelta = new Vector2(0f, 34f);

        Toggle itemToggle = item.gameObject.AddComponent<Toggle>();
        Image itemImage = item.GetComponent<Image>();
        itemToggle.targetGraphic = itemImage;

        RectTransform itemCheck = CreatePanel("Item Checkmark", item, new Color(0.00f, 0.92f, 0.42f, 0.95f));
        AnchorTopLeft(itemCheck, new Vector2(8f, -8f), new Vector2(18f, 18f));
        itemToggle.graphic = itemCheck.GetComponent<Image>();

        TMP_Text itemLabel = AddText(item, "Item Label", string.Empty, 18, FontStyles.Normal, new Vector2(34f, -5f), new Vector2(140f, 24f));
        ConfigureSingleLine(itemLabel);

        ScrollRect scroll = template.gameObject.AddComponent<ScrollRect>();
        scroll.viewport = viewport;
        scroll.content = content;
        scroll.horizontal = false;
        scroll.vertical = true;
        scroll.movementType = ScrollRect.MovementType.Clamped;

        dropdown.targetGraphic = root.GetComponent<Image>();
        dropdown.captionText = caption;
        dropdown.template = template;
        dropdown.itemText = itemLabel;
        dropdown.itemImage = null;
        dropdown.value = 0;
        dropdown.RefreshShownValue();
        return dropdown;
    }

    private static string ResolveRobotDatasetFolder(string robotKey)
    {
        return RobotCatalog.GetDatasetFolderOrKey(robotKey);
    }

    private static void EnsureDropdownHasVisibleCaption(TMP_Dropdown dropdown)
    {
        if (dropdown == null || dropdown.captionText == null)
        {
            return;
        }

        if (dropdown.options != null && dropdown.options.Count > 0)
        {
            int index = Mathf.Clamp(dropdown.value, 0, dropdown.options.Count - 1);
            dropdown.value = index;
            dropdown.RefreshShownValue();
            dropdown.captionText.text = dropdown.options[index].text;
        }
        else
        {
            dropdown.captionText.text = "No CSV";
        }
    }

}
