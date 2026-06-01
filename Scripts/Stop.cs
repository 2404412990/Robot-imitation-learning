using UnityEngine;
using UnityEngine.UI;
using Gewu.Imitation;

public class Stop : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The StartInput component that owns the bash process and CSV monitor. " +
             "Auto-resolved via FindObjectOfType if left empty.")]
    [SerializeField] private StartInput startInput;

    [Tooltip("Optional explicit IMimicAgent to reset after the pipeline is killed. " +
             "When left empty, every registered robot is reset so whichever one was " +
             "consuming the live stream falls back to its own dataset replay.")]
    [SerializeField] private MonoBehaviour targetAgentBehaviour;

    // ── internal state ────────────────────────────────────────────────────────
    private Button stopButton;
    private bool addedRuntimeListener;

    // ── Unity lifecycle ───────────────────────────────────────────────────────

    void Awake()
    {
        stopButton = GetComponent<Button>();
        if (stopButton != null && !HasPersistentStopHandler(stopButton))
        {
            stopButton.onClick.AddListener(OnStopButtonClicked);
            addedRuntimeListener = true;
        }
    }

    void OnDestroy()
    {
        if (stopButton != null && addedRuntimeListener)
        {
            stopButton.onClick.RemoveListener(OnStopButtonClicked);
        }
    }

    // ── public API ────────────────────────────────────────────────────────────

    public void OnStopButtonClicked()
    {
        ExecuteStop();
    }

    // ── core stop logic ───────────────────────────────────────────────────────

    private void ExecuteStop()
    {
        EnsureReferences();

        // 1. Kill bash process and CSV monitor.
        if (startInput != null)
        {
            startInput.StopStartPipeline();
            Debug.Log("[Stop] Bash 进程和 CSV 监听已终止。");
        }
        else
        {
            Debug.LogWarning("[Stop] 未找到 StartInput，无法终止 Bash 进程。");
        }

        // 2. Reset agent(s) to local replay mode.
        //    Clear the live-CSV flag so OnEpisodeBegin reloads motion data from
        //    the dataset directory (the dropdown CSV files). Set replay=true
        //    so the agent uses kinematic teleporting instead of PD-force tracking.
        int resetCount = 0;
        if (targetAgentBehaviour is IMimicAgent pinned && pinned.AgentGameObject != null)
        {
            ResetAgent(pinned);
            resetCount = 1;
        }
        else if (MimicAgentRegistry.Instance != null)
        {
            // Reset every registered robot — the live stream targets exactly
            // one of them, but we don't know which one without re-reading the
            // dropdown, and resetting an already-idle robot is a no-op.
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                ResetAgent(agent);
                resetCount++;
            }
        }

        if (resetCount > 0)
        {
            Debug.Log($"[Stop] {resetCount} 个 Agent 已切换到 replay 模式（EndEpisode 已调用）。");
        }
        else
        {
            Debug.LogWarning("[Stop] 未找到任何已注册的 IMimicAgent，无法重置 Agent。");
        }

        Debug.Log("[Stop] 停止完成。");
    }

    private void ResetAgent(IMimicAgent agent)
    {
        agent.UseExternalReplayData = false;
        agent.ReplayMode = true;
        agent.RequestEndEpisode();
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private void EnsureReferences()
    {
        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>();
            if (startInput == null)
            {
                Debug.LogWarning("[Stop] 场景中未找到 StartInput 组件。" +
                                 "\n请在 Inspector 中手动赋值 startInput 字段。");
            }
        }

        // targetAgentBehaviour is optional. If left empty we fall back to
        // resetting every registered IMimicAgent in ExecuteStop().
        if (targetAgentBehaviour != null && !(targetAgentBehaviour is IMimicAgent))
        {
            Debug.LogWarning("[Stop] targetAgentBehaviour 不是 IMimicAgent 实现，已忽略。" +
                             "\n请确认拖入的是 G1mimicAgent / H1mimicAgent 之类的脚本组件。");
            targetAgentBehaviour = null;
        }
    }

    private bool HasPersistentStopHandler(Button button)
    {
        int eventCount = button.onClick.GetPersistentEventCount();
        for (int i = 0; i < eventCount; i++)
        {
            if (button.onClick.GetPersistentTarget(i) == this &&
                button.onClick.GetPersistentMethodName(i) == nameof(OnStopButtonClicked))
            {
                return true;
            }
        }

        return false;
    }
}