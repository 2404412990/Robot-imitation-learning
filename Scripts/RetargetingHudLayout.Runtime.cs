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

        for (int i = 0; i < lockableToggles.Count; i++)
        {
            if (lockableToggles[i] != null)
            {
                lockableToggles[i].interactable = !locked;
            }
        }

        for (int i = 0; i < paramControls.Count; i++)
        {
            ParamControl control = paramControls[i];
            if (control.Input != null)
            {
                control.Input.interactable = true;
            }

            if (control.Toggle != null)
            {
                control.Toggle.interactable = !locked;
            }
        }
    }

    private void UpdateCameraStatusText()
    {
        if (cameraStatusText != null)
        {
            cameraStatusText.text = SelectedRobotCameraFollow.GetCurrentCameraStatusText();
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

    private Button MoveButton(
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
        GameObject sourceObject = MoveObject(sourceNames, parent, anchoredPosition, size, movedRoots, preferInteractiveRoot: true);
        if (sourceObject == null)
        {
            Button fallbackButton = CreateButton(label + "Button", parent, label, color);
            AnchorTopLeft((RectTransform)fallbackButton.transform, anchoredPosition, size);
            if (fallbackAction != null)
            {
                fallbackButton.onClick.AddListener(fallbackAction);
            }

            if (lockable && !lockableButtons.Contains(fallbackButton))
            {
                lockableButtons.Add(fallbackButton);
            }

            return fallbackButton;
        }

        Button button = sourceObject.GetComponent<Button>() ?? sourceObject.GetComponentInParent<Button>(true);
        if (button == null)
        {
            button = sourceObject.AddComponent<Button>();
        }

        sourceObject.SetActive(true);
        button.enabled = true;
        button.interactable = true;
        StyleButton(button.gameObject, label, color, lockable);

        if (fallbackAction != null && button.onClick.GetPersistentEventCount() == 0)
        {
            button.onClick.AddListener(fallbackAction);
        }

        return button;
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

}
