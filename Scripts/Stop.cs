using UnityEngine;
using UnityEngine.UI;
using Gewu.Imitation;

public class Stop : MonoBehaviour
{
    [Header("References")]
    [Tooltip("The StartInput component that owns the shell process and CSV monitor. " +
             "Auto-resolved via FindObjectOfType if left empty.")]
    [SerializeField] private StartInput startInput;

    [Tooltip("Optional explicit IMimicAgent to reset after the pipeline is killed. " +
             "When left empty, every registered robot is reset so whichever one was " +
             "consuming the live stream falls back to its idle gate.")]
    [SerializeField] private MonoBehaviour targetAgentBehaviour;

    private Button stopButton;
    private bool addedRuntimeListener;

    private void Awake()
    {
        stopButton = GetComponent<Button>();
        if (stopButton != null && !HasPersistentStopHandler(stopButton))
        {
            stopButton.onClick.AddListener(OnStopButtonClicked);
            addedRuntimeListener = true;
        }
    }

    private void OnDestroy()
    {
        if (stopButton != null && addedRuntimeListener)
        {
            stopButton.onClick.RemoveListener(OnStopButtonClicked);
        }
    }

    public void OnStopButtonClicked()
    {
        ExecuteStop();
    }

    private void ExecuteStop()
    {
        EnsureReferences();

        if (startInput != null)
        {
            startInput.StopStartPipeline();
            Debug.Log("[Stop] StartInput pipeline stopped.");
        }
        else
        {
            Debug.LogWarning("[Stop] StartInput was not found; no external pipeline was stopped.");
        }

        int resetCount = 0;
        if (targetAgentBehaviour is IMimicAgent pinned && pinned.AgentGameObject != null)
        {
            ResetAgent(pinned);
            resetCount = 1;
        }
        else if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent == null || agent.AgentGameObject == null)
                {
                    continue;
                }

                ResetAgent(agent);
                resetCount++;
            }
        }

        if (resetCount > 0)
        {
            Debug.Log($"[Stop] Reset {resetCount} agent(s) to idle/neutral mode.");
        }
        else
        {
            Debug.LogWarning("[Stop] No registered IMimicAgent was found to reset.");
        }
    }

    private static void ResetAgent(IMimicAgent agent)
    {
        agent.UseExternalReplayData = false;
        agent.ReplayMode = false;
        agent.ResetToInitialState();
        if (agent.AgentGameObject.activeInHierarchy)
        {
            agent.RequestEndEpisode();
        }
    }

    private void EnsureReferences()
    {
        if (startInput == null)
        {
            startInput = FindObjectOfType<StartInput>();
            if (startInput == null)
            {
                Debug.LogWarning("[Stop] Could not find a StartInput component in the scene.");
            }
        }

        if (targetAgentBehaviour != null && !(targetAgentBehaviour is IMimicAgent))
        {
            Debug.LogWarning("[Stop] targetAgentBehaviour does not implement IMimicAgent and was ignored.");
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
