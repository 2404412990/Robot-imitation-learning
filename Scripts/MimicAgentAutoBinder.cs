using System;
using System.Linq;
using Gewu.Imitation;
using UnityEngine;

public static class MimicAgentAutoBinder
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void EnsureRuntimeAgents()
    {
        BindAgent("X02Lite", typeof(X02LiteMimicAgent), "x02lite");
        BindAgent("OpenLoong", typeof(OpenLoongMimicAgent), "openloong");
    }

    private static void BindAgent(string rootName, Type agentType, string robotKey)
    {
        GameObject root = GameObject.Find(rootName);
        if (root == null)
        {
            Debug.LogWarning($"[MimicAgentAutoBinder] Robot root '{rootName}' was not found.");
            return;
        }

        DisableLegacyRobotRlAgents(root);

        bool alreadyHasAgent = root.GetComponents<MonoBehaviour>()
            .OfType<IMimicAgent>()
            .Any(agent => string.Equals(agent.RobotKey, robotKey, StringComparison.OrdinalIgnoreCase));
        if (alreadyHasAgent)
        {
            return;
        }

        Component added = root.AddComponent(agentType);
        if (added == null)
        {
            Debug.LogError($"[MimicAgentAutoBinder] Failed to add {agentType.Name} to {rootName}.");
            return;
        }

        Debug.Log($"[MimicAgentAutoBinder] Added {agentType.Name} to {rootName} for ROBOT={robotKey}.");
    }

    private static void DisableLegacyRobotRlAgents(GameObject root)
    {
        MonoBehaviour[] behaviours = root.GetComponents<MonoBehaviour>();
        for (int i = 0; i < behaviours.Length; i++)
        {
            MonoBehaviour behaviour = behaviours[i];
            if (behaviour == null) continue;

            if (string.Equals(behaviour.GetType().Name, "RobotRLAgent", StringComparison.Ordinal))
            {
                behaviour.enabled = false;
            }
        }
    }
}
