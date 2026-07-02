using System.Collections.Generic;
using UnityEngine;

namespace Gewu.Imitation
{
    /// <summary>
    /// Scene-wide registry of all IMimicAgent instances.
    /// </summary>
    public class MimicAgentRegistry : MonoBehaviour
    {
        private static MimicAgentRegistry _instance;

        private readonly Dictionary<string, IMimicAgent> _agentsByKey =
            new Dictionary<string, IMimicAgent>(System.StringComparer.OrdinalIgnoreCase);

        private readonly List<IMimicAgent> _registrationOrder = new List<IMimicAgent>();

        public static MimicAgentRegistry Instance
        {
            get
            {
                if (_instance != null) return _instance;

                _instance = FindObjectOfType<MimicAgentRegistry>();
                return _instance;
            }
        }

        void Awake()
        {
            if (_instance != null && _instance != this)
            {
                Destroy(gameObject);
                return;
            }

            _instance = this;
        }

        void OnDestroy()
        {
            if (_instance == this) _instance = null;
        }

        private static GameObject SafeAgentGameObject(IMimicAgent agent)
        {
            if (agent == null) return null;
            if (agent is Object unityObject && unityObject == null) return null;

            try
            {
                return agent.AgentGameObject;
            }
            catch (MissingReferenceException)
            {
                return null;
            }
        }

        private static bool IsDestroyed(IMimicAgent agent)
        {
            return SafeAgentGameObject(agent) == null;
        }

        private void PruneDestroyedAgents()
        {
            for (int i = _registrationOrder.Count - 1; i >= 0; i--)
            {
                if (IsDestroyed(_registrationOrder[i]))
                {
                    _registrationOrder.RemoveAt(i);
                }
            }

            var staleKeys = new List<string>();
            foreach (var kvp in _agentsByKey)
            {
                if (IsDestroyed(kvp.Value)) staleKeys.Add(kvp.Key);
            }

            foreach (string key in staleKeys)
            {
                _agentsByKey.Remove(key);
            }
        }

        public void Register(IMimicAgent agent)
        {
            if (agent == null || string.IsNullOrWhiteSpace(agent.RobotKey)) return;

            PruneDestroyedAgents();
            string key = agent.RobotKey.Trim();

            if (_agentsByKey.TryGetValue(key, out IMimicAgent existing) &&
                existing != agent &&
                !IsDestroyed(existing))
            {
                GameObject existingObject = SafeAgentGameObject(existing);
                GameObject agentObject = SafeAgentGameObject(agent);
                Debug.LogWarning($"[MimicAgentRegistry] Duplicate robot key '{key}'. " +
                                 $"Keeping '{(existingObject != null ? existingObject.name : "<null>")}', " +
                                 $"ignoring '{(agentObject != null ? agentObject.name : "<null>")}'.");
                if (agent is MonoBehaviour duplicateBehaviour)
                {
                    duplicateBehaviour.enabled = false;
                }
                return;
            }

            _agentsByKey[key] = agent;
            if (!_registrationOrder.Contains(agent)) _registrationOrder.Add(agent);
        }

        public void Unregister(IMimicAgent agent)
        {
            if (agent == null) return;

            PruneDestroyedAgents();
            string keyToRemove = null;
            foreach (var kvp in _agentsByKey)
            {
                if (kvp.Value == agent)
                {
                    keyToRemove = kvp.Key;
                    break;
                }
            }

            if (keyToRemove != null) _agentsByKey.Remove(keyToRemove);
            _registrationOrder.Remove(agent);
        }

        public IMimicAgent FindByKey(string robotKey)
        {
            if (string.IsNullOrWhiteSpace(robotKey)) return null;

            PruneDestroyedAgents();
            _agentsByKey.TryGetValue(robotKey.Trim(), out IMimicAgent agent);
            return IsDestroyed(agent) ? null : agent;
        }

        public IMimicAgent GetFirstAlive()
        {
            PruneDestroyedAgents();
            foreach (IMimicAgent agent in _registrationOrder)
            {
                if (!IsDestroyed(agent)) return agent;
            }
            return null;
        }

        public IEnumerable<IMimicAgent> All
        {
            get
            {
                PruneDestroyedAgents();
                var snapshot = new List<IMimicAgent>(_registrationOrder);
                foreach (IMimicAgent agent in snapshot)
                {
                    if (!IsDestroyed(agent)) yield return agent;
                }
            }
        }

        public void SetActiveTarget(IMimicAgent active)
        {
            foreach (IMimicAgent agent in All)
            {
                if (agent == active) continue;
                agent.UseExternalReplayData = false;
            }
        }
    }
}
