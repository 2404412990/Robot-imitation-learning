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

        TMP_Text text = go.transform.Find("HudText")?.GetComponent<TMP_Text>();
        if (text == null)
        {
            text = go.GetComponentInChildren<TMP_Text>(true);
        }

        if (text == null)
        {
            RectTransform textRect = CreateRect("HudText", rect);
            Stretch(textRect);
            text = textRect.gameObject.AddComponent<TextMeshProUGUI>();
        }

        text.gameObject.SetActive(true);
        text.enabled = true;
        text.text = label;
        text.fontSize = label.Length > 8 ? 20f : (label == "PARAMS" ? 24f : 26f);
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

    private Button CreateHudCommandButton(
        string name,
        RectTransform parent,
        Vector2 anchoredPosition,
        Vector2 size,
        string label,
        Color color,
        bool lockable,
        UnityEngine.Events.UnityAction action)
    {
        Button button = CreateButton(name, parent, label, color);
        AnchorTopLeft((RectTransform)button.transform, anchoredPosition, size);
        if (lockable && !lockableButtons.Contains(button))
        {
            lockableButtons.Add(button);
        }

        if (action != null)
        {
            button.onClick.AddListener(action);
        }

        return button;
    }

    private void InvokeStartInput()
    {
        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>(true);
        }

        if (startInput != null)
        {
            startInput.OnStartButtonClicked();
            return;
        }

        Debug.LogError("[RetargetingHudLayout] Start clicked, but no StartInput component was found.");
    }

    private void InvokeReplay()
    {
        Replay replay = FindObjectOfType<Replay>(true);
        if (replay != null)
        {
            replay.OnReplayButtonClicked();
            return;
        }

        Debug.LogError("[RetargetingHudLayout] Replay clicked, but no Replay component was found.");
    }

    private void InvokeStop()
    {
        Stop stop = FindObjectOfType<Stop>(true);
        if (stop != null)
        {
            stop.OnStopButtonClicked();
            return;
        }

        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>(true);
        }

        if (startInput != null)
        {
            startInput.StopStartPipeline();
            return;
        }

        Debug.LogError("[RetargetingHudLayout] Stop clicked, but no Stop or StartInput component was found.");
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
            dropdown.captionText.fontSize = 18;
            dropdown.captionText.color = Color.white;
            dropdown.captionText.fontStyle = FontStyles.Bold;
            dropdown.captionText.raycastTarget = false;
        }

        if (dropdown.itemText != null)
        {
            dropdown.itemText.fontSize = 18;
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
            text.fontSize = Mathf.Max(text.fontSize, 18f);
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

}
