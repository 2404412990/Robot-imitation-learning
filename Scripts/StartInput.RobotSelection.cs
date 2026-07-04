using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using System.Threading;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Gewu.Imitation;

using Debug = UnityEngine.Debug;

public partial class StartInput
{
    // RoboList live-switching
    // -------------------------------------------------------------------------

    /// <summary>
    /// Called whenever the user picks a new entry in the RoboList dropdown.
    /// Stops the running pipeline (CSV format is robot-specific, so the
    /// Python side has to be relaunched), resets every registered IMimicAgent
    /// to a clean state, and clears <c>lastResolvedRobotKey</c> so the next
    /// <c>ResolveActiveAgent</c> call logs the switch.
    /// </summary>
    private void OnRoboListChanged(int newIndex)
    {
        if (roboListDropdown == null || roboListDropdown.options == null) return;
        if (newIndex < 0 || newIndex >= roboListDropdown.options.Count) return;

        string newLabel = roboListDropdown.options[newIndex].text;
        string newKey   = TryResolveRobotKeyQuiet(newLabel);
        RefreshCsvListForSelectedRobot(newKey, newLabel);

        if (!switchActiveRobotOnDropdownChange) return;
        if (string.IsNullOrWhiteSpace(newKey))
        {
            // Label is not a WHAM/GMR-supported robot key. We still apply the
            // visual switch (so the user can use the dropdown as a pure
            // visibility toggle for robots like X02Lite that don't have an
            // IMimicAgent / aren't supported by the Python pipeline yet) and
            // bail out without touching the running pipeline.
            Debug.LogWarning($"[StartInput] '{newLabel}' is not supported by WHAM/GMR; applying visibility only. Supported: {string.Join(", ", SupportedRobotNames)}");
            ApplyRobotVisibility(newLabel, newLabel);
            return;
        }

        bool wasRunning = pythonProcess != null && !pythonProcess.HasExited;
        Debug.Log($"[StartInput] RoboList switched to '{newLabel}' -> ROBOT={newKey}." +
                  (wasRunning ? " Stopping current WHAM/GMR pipeline..." : string.Empty));

        // 1) Stop the running pipeline + CSV monitor (if any).
        StopStartPipeline();

        // 2) Restore Physics.gravity. G1mimicAgent.FixedUpdate's replay branch
        //    sets `Physics.gravity = Vector3.zero` (so the kinematic root
        //    teleport does not fight gravity), but that is a global setting;
        //    leaving it zeroed when we switch to another robot makes that
        //    robot float weirdly. Reset to standard Earth gravity here so the
        //    newly-activated robot starts in a clean physics environment.
        Physics.gravity = new Vector3(0f, -9.81f, 0f);

        // 3) Invalidate cached resolution so the next ResolveActiveAgent picks
        //    the new key freshly and logs the change.
        lastResolvedRobotKey = string.Empty;
        targetAgent = null;
        replayBootstrapped = false;

        // 4) Visual switch FIRST. The original order (reset éˆ?SetActive(false))
        //    queued OnEpisodeBegin via RequestEndEpisode but the agent then got
        //    deactivated before ML-Agents could fire it, so the soon-to-be-
        //    hidden robot's reset never ran, and the next time it became
        //    visible it picked up frozen state ("hands reversed" symptom).
        //    Switching visibility first means only currently-active agents
        //    take the reset, which is exactly what we need.
        ApplyRobotVisibility(newLabel, newKey);

        // 5) HARD-RESET every registered agent imperatively via
        //    ResetToInitialState. Only act on the agent matching the newly-
        //    selected RobotKey; the hidden ones had ReplayMode/UseExternalReplayData
        //    cleared in ApplyRobotVisibility above and we deliberately leave
        //    them alone so we don't restart their articulation simulation.
        if (MimicAgentRegistry.Instance != null)
        {
            IMimicAgent selected = MimicAgentRegistry.Instance.FindByKey(newKey);
            if (selected != null && selected.AgentGameObject != null)
            {
                selected.UseExternalReplayData = false;
                if (string.Equals(selected.RobotKey, "x02lite", System.StringComparison.OrdinalIgnoreCase))
                {
                    selected.ReplayMode = false;
                }
                else
                {
                    selected.ReplayMode = true;

                    try { selected.ResetToInitialState(); }
                    catch (System.Exception e)
                    {
                        Debug.LogWarning($"[StartInput] {selected.RobotKey} ResetToInitialState threw: {e.Message}");
                    }

                    selected.RequestEndEpisode();
                }
                if (string.Equals(selected.RobotKey, "x02lite", System.StringComparison.OrdinalIgnoreCase))
                {
                    Debug.Log($"[StartInput] Selected '{selected.RobotKey}': keep grounded neutral pose and skip default replay.");
                }
                else
                {
                    Debug.Log($"[StartInput] Selected '{selected.RobotKey}': articulation reset queued via OnEpisodeBegin.");
                }
            }
            else
            {
                Debug.Log($"[StartInput] No IMimicAgent with RobotKey='{newKey}'; skipping articulation reset for visibility-only robot.");
            }
        }

        // 5) Pre-verify that an agent with this RobotKey exists in the scene.
        //    If not, warn the user; the live CSV will fall back to the first
        //    registered robot (probably G1).
        if (MimicAgentRegistry.Instance != null && MimicAgentRegistry.Instance.FindByKey(newKey) == null)
        {
            Debug.LogWarning($"[StartInput] No IMimicAgent with RobotKey='{newKey}' exists in the scene. If this robot is only in sceneRobots, visibility changes still work but realtime CSV will not be routed to it.");
        }

        // 6) Optionally auto-relaunch the pipeline with the new ROBOT env var.
        if (autoStartOnRobotSwitch && wasRunning)
        {
            Debug.Log("[StartInput] autoStartOnRobotSwitch=true; restarting pipeline automatically.");
            OnStartButtonClicked();
        }
    }

    // Cached child renderers per robot root. We disable/enable Renderer
    // components instead of toggling GameObject.SetActive because SetActive
    // on a hierarchy that contains ArticulationBody bodies forces Unity to
    // tear down and rebuild the articulation simulation. The rebuild seeds
    // from the LAST cache state (which can be a mid-replay frame) rather
    // than the prefab bind pose, and the C# side ML-Agents Agent ends up
    // re-running Initialize against a partially-rebuilt articulation. Both
    // effects compound across switches; see imitation_robot_switch_ordering
    // and articulation_cache_probe memories. The user confirmed even
    // X02Lite (no script) is corrupted by SetActive cycles, ruling out
    // any agent-side fix and pointing at engine behaviour.
    //
    // Renderer-based hiding leaves the GameObject active throughout: the
    // articulation never rebuilds, agents never re-Initialize, cache values
    // never shift, and the visibility toggle has zero physics side-effects.
    // Cached per-root so we don't allocate every dropdown change.
    private readonly Dictionary<GameObject, Renderer[]> _cachedRobotRenderers =
        new Dictionary<GameObject, Renderer[]>();
    private readonly Dictionary<GameObject, Collider[]> _cachedRobotColliders =
        new Dictionary<GameObject, Collider[]>();
    private readonly Dictionary<Collider, bool> _originalColliderEnabled =
        new Dictionary<Collider, bool>();
    private readonly Dictionary<GameObject, Collider[]> _generatedRobotColliders =
        new Dictionary<GameObject, Collider[]>();
    private readonly HashSet<Collider> _runtimeRobotColliders =
        new HashSet<Collider>();
    private readonly Dictionary<GameObject, RuntimeRobotColliderProxy[]> _runtimeColliderProxies =
        new Dictionary<GameObject, RuntimeRobotColliderProxy[]>();
    private Transform runtimeColliderProxyRoot;

    private sealed class RuntimeRobotColliderProxy
    {
        public Transform Source;
        public Transform Proxy;
        public Collider Collider;
    }

    private Renderer[] GetOrCacheRenderers(GameObject root)
    {
        if (root == null) return System.Array.Empty<Renderer>();
        if (_cachedRobotRenderers.TryGetValue(root, out var cached) && cached != null) return cached;
        var fresh = root.GetComponentsInChildren<Renderer>(includeInactive: true);
        _cachedRobotRenderers[root] = fresh;
        return fresh;
    }

    private Collider[] GetOrCacheColliders(GameObject root)
    {
        if (root == null) return System.Array.Empty<Collider>();
        if (_cachedRobotColliders.TryGetValue(root, out var cached) && cached != null) return cached;

        var fresh = root.GetComponentsInChildren<Collider>(includeInactive: true);
        if (_generatedRobotColliders.TryGetValue(root, out Collider[] generated) &&
            generated != null &&
            generated.Length > 0)
        {
            var combined = new Collider[fresh.Length + generated.Length];
            System.Array.Copy(fresh, combined, fresh.Length);
            System.Array.Copy(generated, 0, combined, fresh.Length, generated.Length);
            fresh = combined;
        }

        for (int i = 0; i < fresh.Length; i++)
        {
            Collider collider = fresh[i];
            if (collider != null && !_originalColliderEnabled.ContainsKey(collider))
            {
                _originalColliderEnabled[collider] = collider.enabled;
            }
        }

        _cachedRobotColliders[root] = fresh;
        return fresh;
    }

    private static bool EnsureRootActiveForVisibility(GameObject root)
    {
        if (root == null) return false;

        bool changed = false;
        Transform current = root.transform;
        while (current != null)
        {
            GameObject go = current.gameObject;
            if (!go.activeSelf)
            {
                go.SetActive(true);
                changed = true;
            }

            current = current.parent;
        }

        return changed;
    }

    private static int EnsureRendererHierarchyActive(GameObject root, Renderer[] renderers)
    {
        if (root == null || renderers == null) return 0;

        int changed = 0;
        Transform rootTransform = root.transform;
        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer renderer = renderers[i];
            if (renderer == null) continue;

            Transform current = renderer.transform;
            while (current != null)
            {
                GameObject go = current.gameObject;
                if (!go.activeSelf)
                {
                    go.SetActive(true);
                    changed++;
                }

                if (current == rootTransform)
                {
                    break;
                }

                current = current.parent;
            }
        }

        return changed;
    }

    private bool GetOriginalColliderEnabled(Collider collider)
    {
        return collider != null &&
               (!_originalColliderEnabled.TryGetValue(collider, out bool originalEnabled) || originalEnabled);
    }

    private void EnsureRuntimeCollidersForSelectedRobot(GameObject root)
    {
        if (!addRuntimeCollidersForSelectedRobot || root == null) return;
        if (_generatedRobotColliders.ContainsKey(root)) return;

        ArticulationBody[] bodies = root.GetComponentsInChildren<ArticulationBody>(includeInactive: true);
        var generated = new List<Collider>();
        var proxies = new List<RuntimeRobotColliderProxy>();

        for (int i = 0; i < bodies.Length; i++)
        {
            ArticulationBody body = bodies[i];
            if (body == null || BodyHasUsableLocalCollider(body))
            {
                continue;
            }

            BoxCollider box = CreateRuntimeColliderProxy(root, body, out RuntimeRobotColliderProxy proxy);
            if (box == null || proxy == null)
            {
                continue;
            }

            if (TryComputeLocalRendererBounds(body, out Bounds bounds))
            {
                box.center = bounds.center;
                box.size = ClampRuntimeColliderSize(bounds.size);
            }
            else
            {
                box.center = Vector3.zero;
                box.size = ClampRuntimeColliderSize(fallbackRuntimeColliderSize);
            }

            ConfigureRuntimeRobotCollider(box);
            box.enabled = false;
            _originalColliderEnabled[box] = true;
            generated.Add(box);
            proxies.Add(proxy);
        }

        _generatedRobotColliders[root] = generated.ToArray();
        _runtimeColliderProxies[root] = proxies.ToArray();
        if (generated.Count > 0)
        {
            _cachedRobotColliders.Remove(root);
            SyncRuntimeColliderProxies(root);
            Debug.Log($"[StartInput] Added {generated.Count} runtime trigger proxy BoxCollider(s) for selected robot '{root.name}'.");
        }
    }

    private BoxCollider CreateRuntimeColliderProxy(
        GameObject robotRoot,
        ArticulationBody sourceBody,
        out RuntimeRobotColliderProxy proxy)
    {
        proxy = null;
        if (robotRoot == null || sourceBody == null) return null;

        Transform proxyRoot = GetRuntimeColliderProxyRoot();
        if (proxyRoot == null) return null;

        string safeRobotName = robotRoot.name.Replace('/', '_');
        string safeBodyName = sourceBody.name.Replace('/', '_');
        var proxyObject = new GameObject($"{safeRobotName}_{safeBodyName}_QueryCollider");
        proxyObject.hideFlags = HideFlags.DontSave;
        proxyObject.transform.SetParent(proxyRoot, worldPositionStays: false);
        CopyWorldTransform(sourceBody.transform, proxyObject.transform);

        BoxCollider box = proxyObject.AddComponent<BoxCollider>();
        proxy = new RuntimeRobotColliderProxy
        {
            Source = sourceBody.transform,
            Proxy = proxyObject.transform,
            Collider = box,
        };
        return box;
    }

    private Transform GetRuntimeColliderProxyRoot()
    {
        if (runtimeColliderProxyRoot != null) return runtimeColliderProxyRoot;

        var root = new GameObject("__RuntimeRobotQueryColliders");
        root.hideFlags = HideFlags.DontSave;
        root.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
        root.transform.localScale = Vector3.one;
        runtimeColliderProxyRoot = root.transform;
        return runtimeColliderProxyRoot;
    }

    private void ConfigureRuntimeRobotCollider(Collider collider)
    {
        if (collider == null) return;
        collider.isTrigger = runtimeRobotCollidersAreTriggers;
        _runtimeRobotColliders.Add(collider);
    }

    private void SyncRuntimeColliderProxies()
    {
        if (_runtimeColliderProxies.Count == 0) return;

        foreach (RuntimeRobotColliderProxy[] proxies in _runtimeColliderProxies.Values)
        {
            SyncRuntimeColliderProxies(proxies);
        }
    }

    private void SyncRuntimeColliderProxies(GameObject root)
    {
        if (root == null) return;
        if (_runtimeColliderProxies.TryGetValue(root, out RuntimeRobotColliderProxy[] proxies))
        {
            SyncRuntimeColliderProxies(proxies);
        }
    }

    private void SetRuntimeColliderProxiesEnabledForCurrentState()
    {
        bool allowForSelected = !IsRealtimeControlsLocked;
        foreach (var kv in _runtimeColliderProxies)
        {
            bool enableForRoot = allowForSelected && kv.Key != null && kv.Key == CurrentSelectedRobotRoot;
            RuntimeRobotColliderProxy[] proxies = kv.Value;
            if (proxies == null) continue;

            for (int i = 0; i < proxies.Length; i++)
            {
                Collider collider = proxies[i]?.Collider;
                if (collider == null) continue;
                collider.enabled = enableForRoot && GetOriginalColliderEnabled(collider);
            }
        }
    }

    private static void SyncRuntimeColliderProxies(RuntimeRobotColliderProxy[] proxies)
    {
        if (proxies == null) return;

        for (int i = 0; i < proxies.Length; i++)
        {
            RuntimeRobotColliderProxy proxy = proxies[i];
            if (proxy == null || proxy.Source == null || proxy.Proxy == null)
            {
                continue;
            }

            CopyWorldTransform(proxy.Source, proxy.Proxy);
        }
    }

    private static void CopyWorldTransform(Transform source, Transform target)
    {
        if (source == null || target == null) return;
        target.SetPositionAndRotation(source.position, source.rotation);
        Vector3 scale = source.lossyScale;
        target.localScale = new Vector3(
            Mathf.Max(Mathf.Abs(scale.x), 0.0001f),
            Mathf.Max(Mathf.Abs(scale.y), 0.0001f),
            Mathf.Max(Mathf.Abs(scale.z), 0.0001f));
    }

    private bool BodyHasUsableLocalCollider(ArticulationBody body)
    {
        if (body == null) return true;
        Collider[] colliders = body.GetComponents<Collider>();
        if (colliders == null || colliders.Length == 0) return false;

        for (int i = 0; i < colliders.Length; i++)
        {
            Collider collider = colliders[i];
            if (collider == null) continue;
            if (collider.enabled || GetOriginalColliderEnabled(collider))
            {
                return true;
            }
        }

        return false;
    }

    private bool TryComputeLocalRendererBounds(ArticulationBody body, out Bounds localBounds)
    {
        localBounds = new Bounds(Vector3.zero, ClampRuntimeColliderSize(fallbackRuntimeColliderSize));
        if (body == null) return false;

        Renderer[] renderers = body.GetComponentsInChildren<Renderer>(includeInactive: true);
        bool hasBounds = false;
        Bounds result = default(Bounds);

        for (int i = 0; i < renderers.Length; i++)
        {
            Renderer renderer = renderers[i];
            if (renderer == null || FindNearestArticulationBody(renderer.transform) != body)
            {
                continue;
            }

            EncapsulateWorldBoundsInLocal(body.transform, renderer.bounds, ref result, ref hasBounds);
        }

        if (!hasBounds) return false;
        localBounds = result;
        return true;
    }

    private static ArticulationBody FindNearestArticulationBody(Transform transform)
    {
        Transform current = transform;
        while (current != null)
        {
            ArticulationBody body = current.GetComponent<ArticulationBody>();
            if (body != null) return body;
            current = current.parent;
        }

        return null;
    }

    private static void EncapsulateWorldBoundsInLocal(
        Transform localFrame,
        Bounds worldBounds,
        ref Bounds localBounds,
        ref bool hasBounds)
    {
        Vector3 center = worldBounds.center;
        Vector3 extents = worldBounds.extents;

        for (int x = -1; x <= 1; x += 2)
        for (int y = -1; y <= 1; y += 2)
        for (int z = -1; z <= 1; z += 2)
        {
            Vector3 worldCorner = center + Vector3.Scale(extents, new Vector3(x, y, z));
            Vector3 localCorner = localFrame.InverseTransformPoint(worldCorner);

            if (!hasBounds)
            {
                localBounds = new Bounds(localCorner, Vector3.zero);
                hasBounds = true;
            }
            else
            {
                localBounds.Encapsulate(localCorner);
            }
        }
    }

    private static Vector3 ClampRuntimeColliderSize(Vector3 size)
    {
        const float minSize = 0.02f;
        const float maxSize = 2.0f;
        return new Vector3(
            Mathf.Clamp(Mathf.Abs(size.x), minSize, maxSize),
            Mathf.Clamp(Mathf.Abs(size.y), minSize, maxSize),
            Mathf.Clamp(Mathf.Abs(size.z), minSize, maxSize));
    }

    /// <summary>
    /// Toggle visibility of the registered/listed robots by enabling or
    /// disabling their <see cref="Renderer"/> components. The matching
    /// robot becomes visible (renderers on); every other tracked robot
    /// becomes invisible (renderers off). GameObjects stay active in the
    /// scene so their ArticulationBody simulation, ML-Agents Agent
    /// initialization, and replay logic are NOT disturbed, which is the
    /// whole point of switching off SetActive (see the long comment on
    /// _cachedRobotRenderers above).
    ///
    /// Side-effect: also pauses the replay loop on hidden IMimicAgent
    /// instances by clearing their ReplayMode and UseExternalReplayData,
    /// so they don't keep running TeleportRoot every FixedUpdate while
    /// invisible. The newly-visible robot's replay state is set up by the
    /// rest of OnRoboListChanged after this function returns.
    ///
    /// Resolution order: (1) sceneRobots Inspector mapping (matches against
    /// label OR RobotKey), (2) registered IMimicAgent.AgentGameObject.
    /// Robots that are neither listed nor registered are left untouched.
    /// </summary>
    private void ApplyRobotVisibility(string selectedLabel, string selectedRobotKey)
    {
        List<string> selectedKeys = BuildSelectedRobotKeyList();
        HashSet<string> activeKeys = new HashSet<string>(selectedKeys, System.StringComparer.OrdinalIgnoreCase);
        CurrentSelectedRobotKey = selectedKeys.Count > 0 ? selectedKeys[0] : string.Empty;

        if (!hideInactiveRobotsOnSwitch)
        {
            CurrentSelectedRobotRoot = ResolveRobotRootByKey(CurrentSelectedRobotKey);
            return;
        }

        bool useLegacySingleSelection = !hasExplicitRobotSelection;

        // Collect every (GameObject, isSelected) pair we know about.
        // Using a dictionary keyed by GameObject so a robot listed in BOTH
        // sceneRobots and the registry is processed only once.
        var roster = new Dictionary<GameObject, bool>();

        // 1) sceneRobots: match against either label or robot key.
        if (sceneRobots != null)
        {
            foreach (RobotSceneEntry entry in sceneRobots)
            {
                if (entry == null || entry.robotRoot == null) continue;
                string entryLabel = (entry.label ?? string.Empty).Trim();

                bool match = false;
                if (useLegacySingleSelection)
                {
                    match =
                        !string.IsNullOrEmpty(entryLabel) && (
                            string.Equals(entryLabel, selectedLabel, System.StringComparison.OrdinalIgnoreCase) ||
                            string.Equals(entryLabel, selectedRobotKey, System.StringComparison.OrdinalIgnoreCase));
                }

                // Or match by the resolved key going through the alias table
                // (so "G1" label still matches a "unitree_g1" entry).
                if (!match && !string.IsNullOrEmpty(entryLabel))
                {
                    string entryKey = TryResolveRobotKeyQuiet(entryLabel);
                    if (!string.IsNullOrEmpty(entryKey) && activeKeys.Contains(entryKey))
                    {
                        match = true;
                    }
                }

                if (!roster.ContainsKey(entry.robotRoot) || match)
                    roster[entry.robotRoot] = match;
            }
        }

        // 2) Registered IMimicAgents: match against RobotKey.
        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                GameObject go = agent.AgentGameObject;
                if (go == null) continue;
                bool match = activeKeys.Contains(agent.RobotKey);
                if (!roster.ContainsKey(go) || match)
                    roster[go] = match;
            }
        }

        AddFallbackSceneRobotRoster(roster, selectedLabel, CurrentSelectedRobotKey);

        if (roster.Count == 0)
        {
            Debug.LogWarning("[StartInput] No visibility roster is available; skip robot visibility switch.");
            return;
        }

        // Build a set of agent GameObjects so we can also pause the matching
        // IMimicAgent's replay loop when its robot is being hidden.
        var agentsByGo = new Dictionary<GameObject, IMimicAgent>();
        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent != null && agent.AgentGameObject != null)
                    agentsByGo[agent.AgentGameObject] = agent;
            }
        }

        int shown = 0, hidden = 0, bootstrapped = 0;
        bool allowRuntimeColliderProxies = !IsRealtimeControlsLocked;
        GameObject selectedRoot = null;
        foreach (var kv in roster)
        {
            GameObject root = kv.Key;
            bool isSelected = kv.Value;
            if (root == null) continue;
            if (isSelected)
            {
                selectedRoot = root;
            }

            // One-time SetActive(true) bootstrap for never-shown robots and
            // inactive visual children. We still never SetActive(false) here:
            // hidden robots are renderer-disabled only, so articulation state
            // is not rebuilt on every visibility change.
            if (isSelected)
            {
                bool rootActivated = EnsureRootActiveForVisibility(root);
                Renderer[] freshRenderers = root.GetComponentsInChildren<Renderer>(includeInactive: true);
                int visualNodesActivated = EnsureRendererHierarchyActive(root, freshRenderers);
                if (rootActivated || visualNodesActivated > 0)
                {
                    bootstrapped++;
                }

                // Selection is the only path that can resurrect a previously
                // inactive hierarchy, so rebuild the cached component arrays
                // before enabling renderers/colliders below.
                _cachedRobotRenderers.Remove(root);
                _cachedRobotColliders.Remove(root);
            }

            // Renderer toggle: visibility-only, no SetActive(false) on the
            // hierarchy. Leaving GameObjects active is what prevents the
            // ArticulationBody rebuild that corrupted joint poses on every
            // dropdown switch in the SetActive-based implementation.
            Renderer[] renderers = GetOrCacheRenderers(root);
            for (int i = 0; i < renderers.Length; i++)
            {
                if (renderers[i] != null) renderers[i].enabled = isSelected;
            }

            if (isSelected && allowRuntimeColliderProxies)
            {
                EnsureRuntimeCollidersForSelectedRobot(root);
                SyncRuntimeColliderProxies(root);
            }

            Collider[] colliders = GetOrCacheColliders(root);
            for (int i = 0; i < colliders.Length; i++)
            {
                Collider collider = colliders[i];
                if (collider != null)
                {
                    if (_runtimeRobotColliders.Contains(collider))
                    {
                        ConfigureRuntimeRobotCollider(collider);
                        collider.enabled = isSelected && allowRuntimeColliderProxies && GetOriginalColliderEnabled(collider);
                        continue;
                    }
                    collider.enabled = isSelected && GetOriginalColliderEnabled(collider);
                }
            }

            if (agentsByGo.TryGetValue(root, out IMimicAgent visibilityAgent) &&
                visibilityAgent is ISelectableMimicAgent selectableAgent)
            {
                selectableAgent.SetRobotSelectedInScene(isSelected);
            }

            // Pause hidden agents' replay loop so they don't keep
            // TeleportRoot-ing themselves around while invisible. The
            // newly-visible robot's replay state is set fresh later in
            // OnRoboListChanged (RequestEndEpisode éˆ?OnEpisodeBegin).
            if (!isSelected && agentsByGo.TryGetValue(root, out IMimicAgent hiddenAgent))
            {
                hiddenAgent.ReplayMode = false;
                hiddenAgent.UseExternalReplayData = false;
            }

            if (isSelected) shown++; else hidden++;
        }

        // If we just activated a robot for the first time, give the IMimicAgent
        // registry a chance to pick up its newly-registered entry before the
        // caller (OnRoboListChanged) tries to FindByKey.
        if (bootstrapped > 0)
        {
            Debug.Log($"[StartInput] First-time activated {bootstrapped} robot(s) with SetActive(true) bootstrap.");
        }
        CurrentSelectedRobotRoot = ResolveFirstSelectedRobotRoot(selectedKeys) ?? selectedRoot;
        Debug.Log($"[StartInput] Visibility switch (Renderer+Collider mode): shown={shown}, hidden={hidden}, selectedKeys='{string.Join(",", selectedKeys)}'.");
    }

    private GameObject ResolveFirstSelectedRobotRoot(List<string> selectedKeys)
    {
        if (selectedKeys == null)
        {
            return null;
        }

        for (int i = 0; i < selectedKeys.Count; i++)
        {
            GameObject root = ResolveRobotRootByKey(selectedKeys[i]);
            if (root != null)
            {
                return root;
            }
        }

        return null;
    }

    private void ApplySelectedRobotDisplayOffsets(HashSet<string> activeKeys)
    {
        if (activeKeys == null)
        {
            activeKeys = BuildSelectedRobotKeySet();
        }

        List<string> selectedKeys = BuildSelectedRobotKeyList();
        float spacing = Mathf.Max(0f, multiRobotDisplaySpacingMeters);
        float centerIndex = (selectedKeys.Count - 1) * 0.5f;

        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent is IReplayRootOffsetMimicAgent offsetAgent)
                {
                    offsetAgent.SetReplayRootOffset(Vector3.zero);
                }
            }
        }

        for (int i = 0; i < selectedKeys.Count; i++)
        {
            string key = selectedKeys[i];
            IMimicAgent agent = ResolveAgentByRobotKey(key);
            if (agent is IReplayRootOffsetMimicAgent offsetAgent)
            {
                offsetAgent.SetReplayRootOffset(new Vector3((i - centerIndex) * spacing, 0f, 0f));
            }
        }
    }

    private void ResetRobotDisplayOffsets()
    {
        if (MimicAgentRegistry.Instance == null)
        {
            return;
        }

        foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
        {
            if (agent is IReplayRootOffsetMimicAgent offsetAgent)
            {
                offsetAgent.SetReplayRootOffset(Vector3.zero);
            }
        }
    }

    private GameObject ResolveRobotRootByKey(string robotKeyOrLabel)
    {
        string key = TryResolveRobotKeyQuiet(robotKeyOrLabel);
        return ResolveSelectedRobotRootForCamera(key, key);
    }

    private GameObject ResolveSelectedRobotRootForCamera(string selectedLabel, string selectedRobotKey)
    {
        string key = !string.IsNullOrWhiteSpace(selectedRobotKey)
            ? selectedRobotKey.Trim()
            : TryResolveRobotKeyQuiet(selectedLabel);

        if (sceneRobots != null)
        {
            foreach (RobotSceneEntry entry in sceneRobots)
            {
                if (entry == null || entry.robotRoot == null)
                {
                    continue;
                }

                string entryLabel = (entry.label ?? string.Empty).Trim();
                string entryKey = TryResolveRobotKeyQuiet(entryLabel);
                if ((!string.IsNullOrWhiteSpace(entryLabel) &&
                     string.Equals(entryLabel, selectedLabel, System.StringComparison.OrdinalIgnoreCase)) ||
                    (!string.IsNullOrWhiteSpace(entryLabel) &&
                     string.Equals(entryLabel, key, System.StringComparison.OrdinalIgnoreCase)) ||
                    (!string.IsNullOrWhiteSpace(entryKey) &&
                     string.Equals(entryKey, key, System.StringComparison.OrdinalIgnoreCase)))
                {
                    return entry.robotRoot;
                }
            }
        }

        if (!string.IsNullOrWhiteSpace(key) && MimicAgentRegistry.Instance != null)
        {
            IMimicAgent agent = MimicAgentRegistry.Instance.FindByKey(key);
            if (agent?.AgentGameObject != null)
            {
                return agent.AgentGameObject;
            }
        }

        switch ((key ?? string.Empty).Trim().ToLowerInvariant())
        {
            case "unitree_g1":
                return FindFallbackRobotRoot("G1", "unitree_g1", "g1_29dof_rev_1_0");
            case "unitree_h1":
                return FindFallbackRobotRoot("H1", "h1", "unitree_h1");
            case "x02lite":
                return FindFallbackRobotRoot("X02Lite", "x02lite", "X02");
            case "openloong":
                return FindFallbackRobotRoot("OpenLoong", "openloong");
            default:
                return FindFallbackRobotRoot(selectedLabel, key);
        }
    }

    private void AddFallbackSceneRobotRoster(Dictionary<GameObject, bool> roster, string selectedLabel, string selectedRobotKey)
    {
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "unitree_g1", "G1", "unitree_g1", "g1_29dof_rev_1_0");
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "unitree_h1", "H1", "h1", "unitree_h1");
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "x02lite", "X02Lite", "x02lite", "X02");
        AddFallbackRobotRoot(roster, selectedLabel, selectedRobotKey, "openloong", "OpenLoong", "openloong");
    }

    private void AddFallbackRobotRoot(Dictionary<GameObject, bool> roster, string selectedLabel, string selectedRobotKey, string robotKey, params string[] sceneNames)
    {
        GameObject root = FindFallbackRobotRoot(sceneNames);
        if (root == null)
        {
            return;
        }

        HashSet<string> activeKeys = BuildSelectedRobotKeySet();
        bool useLegacySingleSelection = !hasExplicitRobotSelection;
        bool match = activeKeys.Contains(robotKey) ||
                     (useLegacySingleSelection && string.Equals(selectedRobotKey, robotKey, System.StringComparison.OrdinalIgnoreCase));
        if (!match && useLegacySingleSelection && !string.IsNullOrWhiteSpace(selectedLabel))
        {
            string labelKey = TryResolveRobotKeyQuiet(selectedLabel);
            match = string.Equals(labelKey, robotKey, System.StringComparison.OrdinalIgnoreCase);
        }

        if (!roster.ContainsKey(root) || match)
        {
            roster[root] = match;
        }
    }

    private static GameObject FindFallbackRobotRoot(params string[] sceneNames)
    {
        for (int pass = 0; pass < 2; pass++)
        {
            bool requireActive = pass == 0;
            foreach (string name in sceneNames)
            {
                if (string.IsNullOrWhiteSpace(name))
                {
                    continue;
                }

                foreach (Transform transform in FindObjectsOfType<Transform>(true))
                {
                    if (transform == null ||
                        !string.Equals(transform.name, name, System.StringComparison.OrdinalIgnoreCase) ||
                        (requireActive && !transform.gameObject.activeInHierarchy))
                    {
                        continue;
                    }

                    if (transform.GetComponentInChildren<ArticulationBody>(true) != null ||
                        transform.GetComponentInChildren<Renderer>(true) != null)
                    {
                        return transform.gameObject;
                    }
                }
            }
        }

        return null;
    }

    private IEnumerator ApplyInitialRobotSelectionStateNextFrame()
    {
        yield return null;
        ApplyInitialRobotSelectionState();
    }

    private void ApplyInitialRobotSelectionState()
    {
        if (initialRobotSelectionApplied)
        {
            return;
        }

        ResolveRoboListReferences();
        ResolveCsvListReferences();

        if (roboListFileBrowser != null)
        {
            roboListFileBrowser.PopulateDropdown();
        }

        string selectedLabel = ResolveSelectedRobotName();
        string selectedKey = TryResolveRobotKeyQuiet(selectedLabel);

        if ((string.IsNullOrWhiteSpace(selectedLabel) || string.IsNullOrWhiteSpace(selectedKey)) &&
            TrySelectRoboListOption(defaultRobotName))
        {
            selectedLabel = ResolveSelectedRobotName();
            selectedKey = TryResolveRobotKeyQuiet(selectedLabel);
        }

        if (string.IsNullOrWhiteSpace(selectedLabel))
        {
            selectedLabel = defaultRobotName.Trim();
        }

        if (string.IsNullOrWhiteSpace(selectedKey))
        {
            selectedKey = TryResolveRobotKeyQuiet(defaultRobotName);
        }

        if (!hasExplicitRobotSelection && !string.IsNullOrWhiteSpace(selectedKey))
        {
            selectedRobotKeys.Clear();
            selectedRobotKeys.Add(selectedKey);
        }

        RefreshCsvListForSelectedRobot(selectedKey, selectedLabel);

        if (!string.IsNullOrWhiteSpace(selectedLabel) || !string.IsNullOrWhiteSpace(selectedKey))
        {
            ApplyRobotVisibility(selectedLabel, selectedKey);
            Debug.Log($"[StartInput] Initial robot selection applied: label='{selectedLabel}', key='{selectedKey}'.");
        }
        else
        {
            LogVerbose("ApplyInitialRobotSelectionState: no label/key resolved.");
        }

        initialRobotSelectionApplied = true;
    }

    private bool TrySelectRoboListOption(string robotKeyOrLabel)
    {
        if (roboListDropdown == null || roboListDropdown.options == null || roboListDropdown.options.Count == 0)
        {
            return false;
        }

        string desired = (robotKeyOrLabel ?? string.Empty).Trim();
        if (string.IsNullOrWhiteSpace(desired))
        {
            return false;
        }

        string desiredKey = TryResolveRobotKeyQuiet(desired);
        for (int i = 0; i < roboListDropdown.options.Count; i++)
        {
            string optionText = roboListDropdown.options[i].text?.Trim() ?? string.Empty;
            if (string.Equals(optionText, desired, System.StringComparison.OrdinalIgnoreCase))
            {
                roboListDropdown.SetValueWithoutNotify(i);
                roboListDropdown.RefreshShownValue();
                LogVerbose($"TrySelectRoboListOption: matched desired='{desired}' with option='{optionText}' at index={i}");
                return true;
            }

            string optionKey = TryResolveRobotKeyQuiet(optionText);
            if (!string.IsNullOrWhiteSpace(desiredKey) &&
                string.Equals(optionKey, desiredKey, System.StringComparison.OrdinalIgnoreCase))
            {
                roboListDropdown.SetValueWithoutNotify(i);
                roboListDropdown.RefreshShownValue();
                LogVerbose($"TrySelectRoboListOption: matched desiredKey='{desiredKey}' with option='{optionText}' at index={i}");
                return true;
            }
        }

        LogVerbose($"TrySelectRoboListOption: no match for '{desired}' (desiredKey='{desiredKey}').");
        return false;
    }

}
