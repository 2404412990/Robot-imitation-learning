using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace Gewu.Imitation
{
    /// <summary>
    /// Scene-wide registry of all <see cref="IMimicAgent"/> instances. Each
    /// agent registers itself when it wakes up so the UI scripts can look up
    /// "the agent for robot X" without having a hard reference to a specific
    /// concrete type. The registry is implemented as a lazy singleton — the
    /// first lookup will auto-create the holder GameObject if none exists
    /// in the scene.
    /// </summary>
    public class MimicAgentRegistry : MonoBehaviour
    {
        private static MimicAgentRegistry _instance;

        // Keyed by robot key (e.g. "unitree_g1"). Case-insensitive on lookup.
        private readonly Dictionary<string, IMimicAgent> _agentsByKey =
            new Dictionary<string, IMimicAgent>(System.StringComparer.OrdinalIgnoreCase);

        // Linear list preserved in registration order for "first agent" fallback.
        private readonly List<IMimicAgent> _registrationOrder = new List<IMimicAgent>();

        public static MimicAgentRegistry Instance
        {
            get
            {
                if (_instance != null) return _instance;

                _instance = FindObjectOfType<MimicAgentRegistry>();
                if (_instance != null) return _instance;

                var holder = new GameObject("MimicAgentRegistry");
                _instance = holder.AddComponent<MimicAgentRegistry>();
                return _instance;
            }
        }

        void Awake()
        {
            if (_instance != null && _instance != this)
            {
                // A second registry was placed in the scene; prefer the
                // already-cached one and quietly disable the duplicate.
                Destroy(gameObject);
                return;
            }
            _instance = this;
        }

        void OnDestroy()
        {
            if (_instance == this) _instance = null;
        }

        public void Register(IMimicAgent agent)
        {
            if (agent == null || string.IsNullOrWhiteSpace(agent.RobotKey)) return;

            string key = agent.RobotKey.Trim();

            if (_agentsByKey.TryGetValue(key, out IMimicAgent existing) && existing != agent && existing != null)
            {
                // Duplicate robot key — keep the first one to win, warn the user.
                Debug.LogWarning($"[MimicAgentRegistry] Duplicate robot key '{key}'. " +
                                 $"Keeping '{(existing.AgentGameObject != null ? existing.AgentGameObject.name : "<null>")}', " +
                                 $"ignoring '{(agent.AgentGameObject != null ? agent.AgentGameObject.name : "<null>")}'.");
                return;
            }

            _agentsByKey[key] = agent;
            if (!_registrationOrder.Contains(agent)) _registrationOrder.Add(agent);
        }

        public void Unregister(IMimicAgent agent)
        {
            if (agent == null) return;

            // Find by reference (key may not be reliable if RobotKey changed at runtime).
            string keyToRemove = null;
            foreach (var kvp in _agentsByKey)
            {
                if (kvp.Value == agent) { keyToRemove = kvp.Key; break; }
            }
            if (keyToRemove != null) _agentsByKey.Remove(keyToRemove);

            _registrationOrder.Remove(agent);
        }

        /// <summary>
        /// Look up an agent by its robot key. Returns null when no match exists.
        /// </summary>
        public IMimicAgent FindByKey(string robotKey)
        {
            if (string.IsNullOrWhiteSpace(robotKey)) return null;

            _agentsByKey.TryGetValue(robotKey.Trim(), out IMimicAgent agent);

            // Drop stale entries whose Unity object was destroyed.
            if (agent != null && agent.AgentGameObject == null)
            {
                _agentsByKey.Remove(robotKey.Trim());
                _registrationOrder.Remove(agent);
                return null;
            }
            return agent;
        }

        /// <summary>
        /// Returns the first still-alive agent in registration order.
        /// Useful as a fallback when no robot key is supplied.
        /// </summary>
        public IMimicAgent GetFirstAlive()
        {
            return _registrationOrder.FirstOrDefault(a => a != null && a.AgentGameObject != null);
        }

        /// <summary>
        /// All currently-registered, still-alive agents.
        /// </summary>
        public IEnumerable<IMimicAgent> All
        {
            get
            {
                foreach (var a in _registrationOrder)
                    if (a != null && a.AgentGameObject != null) yield return a;
            }
        }

        /// <summary>
        /// Set <paramref name="active"/> as the single retargeting target —
        /// non-matching agents have ReplayMode/UseExternalReplayData cleared
        /// and their GameObject left active in the scene (we don't hide them
        /// because the user may want to compare poses across robots).
        /// </summary>
        public void SetActiveTarget(IMimicAgent active)
        {
            foreach (IMimicAgent agent in All)
            {
                if (agent == active) continue;
                agent.UseExternalReplayData = false;
                // Leave ReplayMode untouched on idle robots — they may still
                // be doing local replay from their own dataset.
            }
        }
    }
}
