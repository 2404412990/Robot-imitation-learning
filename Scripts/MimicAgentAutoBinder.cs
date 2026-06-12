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
        IMimicAgent existingAgent = FindExistingAgent(robotKey);
        if (existingAgent != null)
        {
            if (existingAgent.AgentGameObject != null)
            {
                DisableLegacyRobotRlAgents(existingAgent.AgentGameObject);
            }
            return;
        }

        GameObject root = FindRobotRoot(rootName);
        if (root == null)
        {
            Debug.LogWarning($"[MimicAgentAutoBinder] Robot root '{rootName}' was not found.");
            return;
        }

        DisableLegacyRobotRlAgents(root);

        bool alreadyHasAgent = root.GetComponentsInChildren<MonoBehaviour>(true)
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

    private static IMimicAgent FindExistingAgent(string robotKey)
    {
        MonoBehaviour[] behaviours = UnityEngine.Object.FindObjectsOfType<MonoBehaviour>(true);
        for (int i = 0; i < behaviours.Length; i++)
        {
            if (behaviours[i] is IMimicAgent agent &&
                string.Equals(agent.RobotKey, robotKey, StringComparison.OrdinalIgnoreCase))
            {
                return agent;
            }
        }

        return null;
    }

    private static GameObject FindRobotRoot(string rootName)
    {
        Transform[] transforms = UnityEngine.Object.FindObjectsOfType<Transform>(true);
        Transform best = null;
        for (int i = 0; i < transforms.Length; i++)
        {
            Transform candidate = transforms[i];
            if (candidate == null || !string.Equals(candidate.name, rootName, StringComparison.Ordinal))
            {
                continue;
            }

            if (best == null)
            {
                best = candidate;
                continue;
            }

            bool candidateIsHigher = best.IsChildOf(candidate);
            bool candidateHasAgent = candidate.GetComponents<MonoBehaviour>().OfType<IMimicAgent>().Any();
            bool bestHasAgent = best.GetComponents<MonoBehaviour>().OfType<IMimicAgent>().Any();
            if (candidateHasAgent || (!bestHasAgent && candidateIsHigher))
            {
                best = candidate;
            }
        }

        return best != null ? best.gameObject : null;
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
