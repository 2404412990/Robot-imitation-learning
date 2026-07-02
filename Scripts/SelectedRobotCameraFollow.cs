using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Gewu.Imitation;

[ExecuteAlways]
[DefaultExecutionOrder(-9000)]
public sealed class SelectedRobotCameraFollow : MonoBehaviour
{
    [SerializeField] private float frontDistance = 6.0f;
    [SerializeField] private float height = 2.1f;
    [SerializeField] private float lookAtHeight = 1.05f;
    [SerializeField] private float smoothTime = 0.08f;
    [SerializeField] private float viewTransitionSeconds = 0.55f;
    [SerializeField] private float fieldOfView = 45f;
    [SerializeField] private float boundsDistanceMultiplier = 1.55f;
    [SerializeField] private float minViewDistance = 4.2f;
    [SerializeField] private float maxViewDistance = 11.5f;
    [SerializeField] private float targetFramePadding = 1.18f;
    [SerializeField] private float boundsHeightMultiplier = 0.25f;
    [SerializeField] private bool invertFrontDirection;

    private static readonly string[] CameraNames = { "Main Camera", "RightCamera", "BackCamera", "LeftCamera" };
    private static readonly string[] LegacyCameraNames = { "Main Camera", "Main Camera (1)", "Main Camera (2)", "Main Camera (3)" };
    private static readonly string[] AnchorTargetKeys = { "overview", "unitree_g1", "unitree_h1", "x02lite", "openloong" };
    private static readonly string[] AnchorDirectionNames = { "Front", "Right", "Back", "Left" };
    private static int viewIndex;
    private static int robotTargetIndex = 0;
    private static bool robotTargetIsOverview = true;
    private static SelectedRobotCameraFollow primaryFollower;

    private readonly Camera[] viewCameras = new Camera[CameraNames.Length];
    private readonly Vector3[] desiredCameraPositions = new Vector3[CameraNames.Length];
    private readonly Quaternion[] desiredCameraRotations = new Quaternion[CameraNames.Length];
    private readonly Dictionary<string, Transform[]> cameraAnchors = new Dictionary<string, Transform[]>(System.StringComparer.OrdinalIgnoreCase);
    private Vector3 followVelocity;
    private Transform lastTarget;
    private Transform anchorRigRoot;
    private string activeAnchorTargetKey = "overview";
    private int lastViewIndex = -1;
    private bool transitionActive;
    private Vector3 transitionStartPosition;
    private Quaternion transitionStartRotation;
    private float transitionStartTime;

    public static bool FixedCameraModeActive => FindObjectOfType<SelectedRobotCameraFollow>(true) != null;

    public static string GetCurrentCameraStatusText()
    {
        SelectedRobotCameraFollow follower = ResolvePrimaryFollower();
        return follower != null ? follower.BuildCameraStatusText() : "View: Overview | Front";
    }

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    private static void BootstrapRuntime()
    {
        AttachToMainCamera();
    }

#if UNITY_EDITOR
    [UnityEditor.InitializeOnLoadMethod]
    private static void BootstrapEditor()
    {
        UnityEditor.EditorApplication.delayCall += () =>
        {
            if (!Application.isPlaying)
            {
                AttachToMainCamera();
            }
        };
    }
#endif

    private static void AttachToMainCamera()
    {
        Camera main = FindFrontCamera();
        if (main == null)
        {
            return;
        }

        main.gameObject.name = "Main Camera";
        main.tag = "MainCamera";
        main.gameObject.SetActive(true);
        main.enabled = true;
        main.targetDisplay = 0;

        if (main.GetComponent<SelectedRobotCameraFollow>() != null)
        {
            return;
        }

#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            UnityEditor.Undo.AddComponent<SelectedRobotCameraFollow>(main.gameObject);
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(main.gameObject.scene);
            return;
        }
#endif
        main.gameObject.AddComponent<SelectedRobotCameraFollow>();
    }

    public static void EnsureAtLeastOneRenderingCamera()
    {
        AttachToMainCamera();
        var follower = ResolvePrimaryFollower();
        if (follower != null)
        {
            follower.EnsureCameraRig();
            follower.ApplyActiveCamera();
            return;
        }

        Camera main = FindFrontCamera();
        if (main != null)
        {
            main.gameObject.SetActive(true);
            main.enabled = true;
            main.targetDisplay = 0;
        }
    }

    public static void SwitchNextView()
    {
        int next = (viewIndex + 1) % CameraNames.Length;
        SelectedRobotCameraFollow follower = ResolvePrimaryFollower();
        if (follower != null)
        {
            follower.BeginViewTransition(next);
        }
        else
        {
            viewIndex = next;
        }
    }

    public static void SwitchNextRobotTarget()
    {
        SelectedRobotCameraFollow follower = ResolvePrimaryFollower();
        int targetCount = follower != null ? follower.CountSelectableRobotTargets() : 0;
        int totalTargets = Mathf.Max(1, targetCount + 1); // selected robots + overview
        int current = follower != null ? follower.NormalizeRobotTargetIndex(targetCount) : 0;
        int next = (current + 1) % totalTargets;

        if (follower != null)
        {
            follower.BeginRobotTargetTransition(next);
        }
        else
        {
            robotTargetIndex = next;
        }
        robotTargetIsOverview = next == 0;
    }

    public static void NotifyRobotSelectionChanged(string robotKeyOrLabel, bool selected)
    {
        SelectedRobotCameraFollow follower = ResolvePrimaryFollower();
        if (follower == null)
        {
            robotTargetIndex = 0;
            robotTargetIsOverview = true;
            return;
        }

        follower.EnsureCameraRig();
        int targetCount = follower.CountSelectableRobotTargets();
        if (targetCount <= 0)
        {
            robotTargetIndex = 0;
            robotTargetIsOverview = true;
            follower.followVelocity = Vector3.zero;
            follower.lastTarget = null;
            return;
        }

        int normalized = follower.NormalizeRobotTargetIndex(targetCount);
        if (normalized > targetCount)
        {
            robotTargetIndex = 0;
            robotTargetIsOverview = true;
        }
    }

    public static void SetFrontView()
    {
        SelectedRobotCameraFollow follower = ResolvePrimaryFollower();
        if (follower != null)
        {
            follower.BeginViewTransition(0);
        }
        else
        {
            viewIndex = 0;
        }
    }

    private string BuildCameraStatusText()
    {
        string direction = ResolveDirectionLabel(viewIndex);
        string targetLabel = "Overview";

        if (!robotTargetIsOverview)
        {
            List<GameObject> roots = ResolveSelectableRoots();
            int normalized = NormalizeRobotTargetIndex(roots.Count);
            if (normalized > 0 && roots.Count > 0)
            {
                GameObject root = roots[Mathf.Clamp(normalized - 1, 0, roots.Count - 1)];
                targetLabel = FormatRobotLabel(ResolveRobotKeyForRoot(root));
            }
            else if (!string.IsNullOrWhiteSpace(activeAnchorTargetKey))
            {
                targetLabel = FormatRobotLabel(activeAnchorTargetKey);
            }
        }

        return $"View: {targetLabel} | {direction}";
    }

    private static string ResolveDirectionLabel(int index)
    {
        switch (Mathf.Abs(index) % 4)
        {
            case 1:
                return "Right";
            case 2:
                return "Back";
            case 3:
                return "Left";
            default:
                return "Front";
        }
    }

    private static string FormatRobotLabel(string robotKey)
    {
        string key = robotKey?.Trim().ToLowerInvariant() ?? string.Empty;
        switch (key)
        {
            case "unitree_g1":
            case "g1":
                return "G1";
            case "unitree_h1":
            case "h1":
                return "H1";
            case "x02lite":
            case "x02":
                return "X02Lite";
            case "openloong":
                return "OpenLoong";
            default:
                return string.IsNullOrWhiteSpace(robotKey) || key == "overview" ? "总视角" : robotKey.Trim();
        }
    }

    private void BeginViewTransition(int nextViewIndex)
    {
        EnsureCameraRig();
        Camera renderCamera = viewCameras[0] != null ? viewCameras[0] : GetComponent<Camera>();
        if (renderCamera != null)
        {
            transitionStartPosition = renderCamera.transform.position;
            transitionStartRotation = renderCamera.transform.rotation;
        }

        viewIndex = Mathf.Abs(nextViewIndex) % CameraNames.Length;
        transitionStartTime = Time.unscaledTime;
        transitionActive = Application.isPlaying && viewTransitionSeconds > 0.001f;
        followVelocity = Vector3.zero;
        lastViewIndex = viewIndex;
    }

    private void BeginRobotTargetTransition(int nextRobotTargetIndex)
    {
        EnsureCameraRig();
        CaptureTransitionStart();
        int targetCount = CountSelectableRobotTargets();
        int totalTargets = Mathf.Max(1, targetCount + 1);
        robotTargetIndex = Mathf.Clamp(nextRobotTargetIndex, 0, totalTargets - 1);
        robotTargetIsOverview = robotTargetIndex == 0;
        transitionStartTime = Time.unscaledTime;
        transitionActive = Application.isPlaying && viewTransitionSeconds > 0.001f;
        followVelocity = Vector3.zero;
    }

    private void CaptureTransitionStart()
    {
        Camera renderCamera = viewCameras[0] != null ? viewCameras[0] : GetComponent<Camera>();
        if (renderCamera != null)
        {
            transitionStartPosition = renderCamera.transform.position;
            transitionStartRotation = renderCamera.transform.rotation;
        }
    }

    private void OnEnable()
    {
        primaryFollower = ResolvePrimaryFollower();
        EnsureCameraRig();
        DisableRoamingControls();
        ApplyCursorState();
    }

    private void LateUpdate()
    {
        if (ResolvePrimaryFollower() != this)
        {
            DisableLocalCamera();
            return;
        }

        EnsureCameraRig();
        DisableRoamingControls();
        ApplyCursorState();
        UpdateAllCameraAnchors();

        if (TryResolveTarget(out GameObject targetRoot, out Bounds targetBounds))
        {
            Transform target = targetRoot.transform;
            bool targetChanged = lastTarget != target;
            if (targetChanged)
            {
                followVelocity = Vector3.zero;
                lastTarget = target;
            }
            lastViewIndex = viewIndex;

            UpdateCameraTransforms(target, targetBounds, targetChanged && !transitionActive);
        }

        ApplyActiveCamera();
    }

    private static SelectedRobotCameraFollow ResolvePrimaryFollower()
    {
        SelectedRobotCameraFollow mainFollower = null;
        SelectedRobotCameraFollow first = null;
        foreach (var follower in FindObjectsOfType<SelectedRobotCameraFollow>(true))
        {
            if (follower == null || !follower.enabled)
            {
                continue;
            }

            if (first == null)
            {
                first = follower;
            }

            if (follower.gameObject.name == "Main Camera")
            {
                mainFollower = follower;
                break;
            }
        }

        primaryFollower = mainFollower != null ? mainFollower : first;
        return primaryFollower;
    }

    private void EnsureCameraRig()
    {
        Camera front = GetComponent<Camera>();
        if (front == null)
        {
            return;
        }

        front.gameObject.name = "Main Camera";
        front.tag = "MainCamera";
        front.targetDisplay = 0;
        viewCameras[0] = front;

        for (int i = 1; i < CameraNames.Length; i++)
        {
            viewCameras[i] = FindOrCreateViewCamera(i, front);
        }

        for (int i = 0; i < viewCameras.Length; i++)
        {
            Camera camera = viewCameras[i];
            if (camera == null)
            {
                continue;
            }

            camera.fieldOfView = fieldOfView;
            camera.targetDisplay = 0;
            camera.gameObject.SetActive(true);
            ConfigureAudioListener(camera, i == 0);
        }

        EnsureAnchorRig();
    }

    private Camera FindOrCreateViewCamera(int index, Camera template)
    {
        Camera existing = FindCameraByName(CameraNames[index]);
        if (existing == null && index < LegacyCameraNames.Length)
        {
            existing = FindCameraByName(LegacyCameraNames[index]);
        }

        if (existing == null)
        {
            existing = CreateViewCamera(CameraNames[index], template);
        }

        existing.gameObject.name = CameraNames[index];
        existing.tag = "Untagged";
        CopyCameraSettings(template, existing);
        return existing;
    }

    private static Camera CreateViewCamera(string cameraName, Camera template)
    {
        var go = new GameObject(cameraName);
#if UNITY_EDITOR
        if (!Application.isPlaying)
        {
            UnityEditor.Undo.RegisterCreatedObjectUndo(go, "Create fixed robot view camera");
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(template.gameObject.scene);
        }
#endif
        var camera = go.AddComponent<Camera>();
        CopyCameraSettings(template, camera);
        return camera;
    }

    private static void CopyCameraSettings(Camera source, Camera target)
    {
        if (source == null || target == null || source == target)
        {
            return;
        }

        target.clearFlags = source.clearFlags;
        target.backgroundColor = source.backgroundColor;
        target.cullingMask = source.cullingMask;
        target.orthographic = source.orthographic;
        target.orthographicSize = source.orthographicSize;
        target.nearClipPlane = source.nearClipPlane;
        target.farClipPlane = source.farClipPlane;
        target.depth = source.depth;
        target.allowHDR = source.allowHDR;
        target.allowMSAA = source.allowMSAA;
    }

    private void UpdateCameraTransforms(Transform target, Bounds targetBounds, bool snapActive)
    {
        Camera renderCamera = viewCameras[0] != null ? viewCameras[0] : GetComponent<Camera>();
        Vector3 frontDirection = ResolveFrontDirection(target);
        for (int i = 0; i < viewCameras.Length; i++)
        {
            Camera camera = viewCameras[i];
            ComputeCameraPose(targetBounds, ResolveAnchorDirection(frontDirection, i), out Vector3 desired, out Quaternion desiredRotation);
            desiredCameraPositions[i] = desired;
            desiredCameraRotations[i] = desiredRotation;

            if (camera != null && i > 0)
            {
                camera.transform.SetPositionAndRotation(desired, desiredRotation);
            }
        }

        if (renderCamera == null)
        {
            return;
        }

        int activeIndex = Mathf.Abs(viewIndex) % CameraNames.Length;
        Vector3 targetPosition = desiredCameraPositions[activeIndex];
        Quaternion targetRotation = desiredCameraRotations[activeIndex];
        Transform activeAnchor = GetCurrentAnchor(activeIndex);
        if (activeAnchor != null)
        {
            targetPosition = activeAnchor.position;
            targetRotation = activeAnchor.rotation;
        }

        if (!Application.isPlaying || snapActive)
        {
            renderCamera.transform.SetPositionAndRotation(targetPosition, targetRotation);
            transitionActive = false;
            return;
        }

        if (transitionActive)
        {
            float t = Mathf.Clamp01((Time.unscaledTime - transitionStartTime) / Mathf.Max(0.001f, viewTransitionSeconds));
            float eased = t * t * (3f - 2f * t);
            renderCamera.transform.position = Vector3.Lerp(transitionStartPosition, targetPosition, eased);
            renderCamera.transform.rotation = Quaternion.Slerp(transitionStartRotation, targetRotation, eased);
            if (t >= 1f)
            {
                transitionActive = false;
            }

            return;
        }

        if (smoothTime > 0f)
        {
            renderCamera.transform.position = Vector3.SmoothDamp(renderCamera.transform.position, targetPosition, ref followVelocity, smoothTime);
            float lerp = 1f - Mathf.Exp(-Time.deltaTime / Mathf.Max(0.001f, smoothTime));
            renderCamera.transform.rotation = Quaternion.Slerp(renderCamera.transform.rotation, targetRotation, lerp);
        }
        else
        {
            renderCamera.transform.SetPositionAndRotation(targetPosition, targetRotation);
        }
    }

    private void ApplyActiveCamera()
    {
        foreach (Camera otherCamera in FindObjectsOfType<Camera>(true))
        {
            if (otherCamera == null || IsManagedCamera(otherCamera))
            {
                continue;
            }

            if (otherCamera.targetDisplay == 0 && otherCamera.targetTexture == null)
            {
                otherCamera.enabled = false;
                ConfigureAudioListener(otherCamera, false);
            }
        }

        for (int i = 0; i < viewCameras.Length; i++)
        {
            Camera camera = viewCameras[i];
            if (camera == null)
            {
                continue;
            }

            bool active = i == 0;
            camera.enabled = active;
            camera.targetDisplay = 0;
            ConfigureAudioListener(camera, active);
        }
    }

    private bool IsManagedCamera(Camera candidate)
    {
        for (int i = 0; i < viewCameras.Length; i++)
        {
            if (viewCameras[i] == candidate)
            {
                return true;
            }
        }

        return false;
    }

    private void DisableLocalCamera()
    {
        Camera camera = GetComponent<Camera>();
        if (camera != null && gameObject.name != "Main Camera")
        {
            camera.enabled = false;
        }
    }

    private static void ConfigureAudioListener(Camera camera, bool enabled)
    {
        if (camera == null)
        {
            return;
        }

        var listener = camera.GetComponent<AudioListener>();
        if (listener != null)
        {
            listener.enabled = enabled;
        }
    }

    private void OnDestroy()
    {
        if (Application.isPlaying && anchorRigRoot != null)
        {
            Destroy(anchorRigRoot.gameObject);
            anchorRigRoot = null;
        }
    }

    private void EnsureAnchorRig()
    {
        if (!Application.isPlaying)
        {
            return;
        }

        if (anchorRigRoot == null)
        {
            GameObject existing = GameObject.Find("CameraAnchorRigRoot");
            GameObject root = existing != null ? existing : new GameObject("CameraAnchorRigRoot");
            anchorRigRoot = root.transform;
        }

        for (int i = 0; i < AnchorTargetKeys.Length; i++)
        {
            string targetKey = AnchorTargetKeys[i];
            Transform targetFolder = anchorRigRoot.Find(targetKey);
            if (targetFolder == null)
            {
                targetFolder = new GameObject(targetKey).transform;
                targetFolder.SetParent(anchorRigRoot, false);
            }

            if (!cameraAnchors.TryGetValue(targetKey, out Transform[] anchors) || anchors == null || anchors.Length != AnchorDirectionNames.Length)
            {
                anchors = new Transform[AnchorDirectionNames.Length];
                cameraAnchors[targetKey] = anchors;
            }

            for (int d = 0; d < AnchorDirectionNames.Length; d++)
            {
                Transform anchor = targetFolder.Find(AnchorDirectionNames[d]);
                if (anchor == null)
                {
                    anchor = new GameObject(AnchorDirectionNames[d]).transform;
                    anchor.SetParent(targetFolder, false);
                }

                anchors[d] = anchor;
            }
        }
    }

    private void UpdateAllCameraAnchors()
    {
        if (!Application.isPlaying)
        {
            return;
        }

        EnsureAnchorRig();
        var startInput = FindObjectOfType<StartInput>(true);
        if (startInput != null)
        {
            var selectedRoots = new List<GameObject>();
            foreach (GameObject root in startInput.GetSelectedRobotRoots())
            {
                if (root != null && root.activeInHierarchy && !selectedRoots.Contains(root))
                {
                    selectedRoots.Add(root);
                }
            }

            if (selectedRoots.Count > 0 && TryBuildCombinedBounds(selectedRoots, out Bounds overviewBounds))
            {
                UpdateAnchorGroup("overview", overviewBounds, ResolveOverviewFrontDirection(selectedRoots));
            }
        }

        for (int i = 1; i < AnchorTargetKeys.Length; i++)
        {
            Transform root = ResolveSceneObjectByRobotKey(AnchorTargetKeys[i]);
            if (root != null && TryBuildTargetBounds(root.gameObject, out Bounds bounds))
            {
                UpdateAnchorGroup(AnchorTargetKeys[i], bounds, ResolveFrontDirection(root));
            }
        }
    }

    private void UpdateAnchorGroup(string targetKey, Bounds bounds, Vector3 frontDirection)
    {
        if (!cameraAnchors.TryGetValue(targetKey, out Transform[] anchors) || anchors == null)
        {
            return;
        }

        for (int i = 0; i < anchors.Length && i < AnchorDirectionNames.Length; i++)
        {
            if (anchors[i] == null)
            {
                continue;
            }

            ComputeCameraPose(bounds, ResolveAnchorDirection(frontDirection, i), out Vector3 position, out Quaternion rotation);
            anchors[i].SetPositionAndRotation(position, rotation);
        }
    }

    private Vector3 ResolveOverviewFrontDirection(IReadOnlyList<GameObject> roots)
    {
        Vector3 sum = Vector3.zero;
        if (roots != null)
        {
            for (int i = 0; i < roots.Count; i++)
            {
                if (roots[i] == null)
                {
                    continue;
                }

                sum += ProjectHorizontal(roots[i].transform.forward);
            }
        }

        return NormalizeOrDefault(sum, Vector3.forward);
    }

    private Vector3 ResolveFrontDirection(Transform target)
    {
        Vector3 direction = target != null ? ProjectHorizontal(target.forward) : Vector3.forward;
        direction = NormalizeOrDefault(direction, Vector3.forward);
        return invertFrontDirection ? -direction : direction;
    }

    private static Vector3 ResolveAnchorDirection(Vector3 frontDirection, int directionIndex)
    {
        Vector3 front = NormalizeOrDefault(frontDirection, Vector3.forward);
        Vector3 right = NormalizeOrDefault(Vector3.Cross(Vector3.up, front), Vector3.right);
        switch (Mathf.Abs(directionIndex) % 4)
        {
            case 1:
                return right;
            case 2:
                return -front;
            case 3:
                return -right;
            default:
                return front;
        }
    }

    private static Vector3 ProjectHorizontal(Vector3 value)
    {
        value.y = 0f;
        return value;
    }

    private static Vector3 NormalizeOrDefault(Vector3 value, Vector3 fallback)
    {
        return value.sqrMagnitude > 0.0001f ? value.normalized : fallback.normalized;
    }

    private void ComputeCameraPose(Bounds targetBounds, Vector3 direction, out Vector3 position, out Quaternion rotation)
    {
        Camera renderCamera = viewCameras[0] != null ? viewCameras[0] : GetComponent<Camera>();
        float aspect = renderCamera != null ? Mathf.Max(0.7f, renderCamera.aspect) : 1.6f;
        float verticalExtent = Mathf.Max(0.75f, targetBounds.extents.y);
        float horizontalExtent = Mathf.Max(0.75f, Mathf.Max(targetBounds.extents.x, targetBounds.extents.z));
        float maxExtent = Mathf.Max(verticalExtent, horizontalExtent);
        float halfFov = Mathf.Max(10f, fieldOfView) * Mathf.Deg2Rad * 0.5f;
        float distanceForHeight = (verticalExtent * targetFramePadding) / Mathf.Tan(halfFov);
        float distanceForWidth = (horizontalExtent * targetFramePadding) / (Mathf.Tan(halfFov) * aspect);
        float multiplier = Mathf.Clamp(boundsDistanceMultiplier, 1.1f, 1.8f);
        float distance = Mathf.Max(minViewDistance, Mathf.Max(frontDistance, Mathf.Max(distanceForHeight, distanceForWidth)));
        distance = Mathf.Max(distance, maxExtent * multiplier);
        distance = Mathf.Clamp(distance, minViewDistance, Mathf.Max(minViewDistance, maxViewDistance));
        float viewHeight = Mathf.Max(height, targetBounds.extents.y * 0.55f + boundsHeightMultiplier * maxExtent);
        Vector3 center = targetBounds.center;
        Vector3 lookAt = center + Vector3.up * lookAtHeight;
        Vector3 safeDirection = direction.sqrMagnitude > 0.0001f ? direction.normalized : Vector3.back;
        position = center + safeDirection * distance + Vector3.up * viewHeight;
        Vector3 lookDirection = lookAt - position;
        rotation = lookDirection.sqrMagnitude > 0.0001f
            ? Quaternion.LookRotation(lookDirection.normalized, Vector3.up)
            : Quaternion.identity;
    }

    private Transform GetCurrentAnchor(int activeIndex)
    {
        if (string.IsNullOrWhiteSpace(activeAnchorTargetKey))
        {
            activeAnchorTargetKey = "overview";
        }

        if (!cameraAnchors.TryGetValue(activeAnchorTargetKey, out Transform[] anchors) || anchors == null)
        {
            return null;
        }

        int index = Mathf.Clamp(activeIndex, 0, anchors.Length - 1);
        return anchors[index];
    }

    private static Camera FindFrontCamera()
    {
        Camera exact = FindCameraByName("Main Camera");
        if (exact != null)
        {
            return exact;
        }

        Camera main = Camera.main;
        if (main != null)
        {
            return main;
        }

        foreach (var camera in FindObjectsOfType<Camera>(true))
        {
            if (camera != null && camera.gameObject.name.Contains("Main Camera"))
            {
                return camera;
            }
        }

        foreach (var camera in FindObjectsOfType<Camera>(true))
        {
            if (camera != null)
            {
                return camera;
            }
        }

        return null;
    }

    private static Camera FindCameraByName(string cameraName)
    {
        foreach (var camera in FindObjectsOfType<Camera>(true))
        {
            if (camera != null && camera.gameObject.name == cameraName)
            {
                return camera;
            }
        }

        return null;
    }

    private static void DisableRoamingControls()
    {
        foreach (var move in FindObjectsOfType<Move>(true))
        {
            if (move != null)
            {
                move.SetCursorState(false);
                move.enabled = false;
            }
        }

        foreach (var interactor in FindObjectsOfType<FPSUiInteractor>(true))
        {
            if (interactor == null)
            {
                continue;
            }

            var crosshair = interactor.crosshairImage;
            if (crosshair != null)
            {
                crosshair.enabled = false;
                crosshair.gameObject.SetActive(false);
            }

            interactor.enabled = false;
        }
    }

    private static void ApplyCursorState()
    {
        Cursor.lockState = CursorLockMode.None;
        Cursor.visible = true;
    }

    private bool TryResolveTarget(out GameObject targetRoot, out Bounds bounds)
    {
        targetRoot = null;
        bounds = default;

        var startInput = FindObjectOfType<StartInput>(true);
        if (TryResolveTargetFromSelectedRobots(startInput, out targetRoot, out bounds))
        {
            return true;
        }

        string selectedKey = ResolveSelectedRobotKey();
        Transform selected = ResolveSelectedTransform(selectedKey);
        if (selected != null)
        {
            targetRoot = selected.gameObject;
            activeAnchorTargetKey = ResolveRobotKeyForRoot(targetRoot);
            return TryBuildTargetBounds(targetRoot, out bounds);
        }

        if (!string.IsNullOrWhiteSpace(selectedKey))
        {
            if (lastTarget != null && lastTarget.gameObject.activeInHierarchy && TryBuildTargetBounds(lastTarget.gameObject, out bounds))
            {
                targetRoot = lastTarget.gameObject;
                return true;
            }

            return false;
        }

        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent?.AgentGameObject != null && agent.AgentGameObject.activeInHierarchy)
                {
                    targetRoot = agent.AgentGameObject;
                    activeAnchorTargetKey = ResolveRobotKeyForRoot(targetRoot);
                    return TryBuildTargetBounds(targetRoot, out bounds);
                }
            }
        }

        string[] fallbackNames = { "G1", "H1", "X02Lite", "OpenLoong" };
        foreach (string name in fallbackNames)
        {
            Transform transform = FindBestActiveTransformByName(name);
            if (transform != null)
            {
                targetRoot = transform.gameObject;
                activeAnchorTargetKey = ResolveRobotKeyForRoot(targetRoot);
                return TryBuildTargetBounds(targetRoot, out bounds);
            }
        }

        return false;
    }

    private int CountSelectableRobotTargets()
    {
        return ResolveSelectableRoots().Count;
    }

    private int NormalizeRobotTargetIndex(int targetCount)
    {
        int totalTargets = Mathf.Max(1, targetCount + 1);
        if (robotTargetIsOverview)
        {
            return 0;
        }

        if (robotTargetIndex <= 0 || robotTargetIndex >= totalTargets)
        {
            return 0;
        }

        return robotTargetIndex;
    }

    private bool TryResolveTargetFromSelectedRobots(StartInput startInput, out GameObject targetRoot, out Bounds bounds)
    {
        targetRoot = null;
        bounds = default;
        var roots = ResolveSelectableRoots(startInput);

        if (roots.Count == 0)
        {
            return false;
        }

        robotTargetIndex = NormalizeRobotTargetIndex(roots.Count);
        robotTargetIsOverview = robotTargetIndex == 0;

        if (robotTargetIndex == 0)
        {
            targetRoot = roots[0];
            activeAnchorTargetKey = "overview";
            return TryBuildCombinedBounds(roots, out bounds);
        }

        int index = Mathf.Clamp(robotTargetIndex - 1, 0, roots.Count - 1);
        targetRoot = roots[index];
        activeAnchorTargetKey = ResolveRobotKeyForRoot(targetRoot);
        return TryBuildTargetBounds(targetRoot, out bounds);
    }

    private static List<GameObject> ResolveSelectableRoots(StartInput startInput = null)
    {
        var roots = new List<GameObject>();
        StartInput input = startInput != null ? startInput : FindObjectOfType<StartInput>(true);
        if (input == null)
        {
            return roots;
        }

        foreach (GameObject root in input.GetSelectedRobotRoots())
        {
            if (root != null && root.activeInHierarchy && !roots.Contains(root))
            {
                roots.Add(root);
            }
        }

        if (roots.Count == 0 &&
            input.CurrentSelectedRobotRoot != null &&
            input.CurrentSelectedRobotRoot.activeInHierarchy)
        {
            roots.Add(input.CurrentSelectedRobotRoot);
        }

        return roots;
    }

    private static bool TryBuildCombinedBounds(IReadOnlyList<GameObject> roots, out Bounds bounds)
    {
        bounds = default;
        bool hasBounds = false;
        if (roots == null)
        {
            return false;
        }

        for (int i = 0; i < roots.Count; i++)
        {
            if (roots[i] == null || !roots[i].activeInHierarchy)
            {
                continue;
            }

            if (!TryBuildTargetBounds(roots[i], out Bounds robotBounds))
            {
                continue;
            }

            if (!hasBounds)
            {
                bounds = robotBounds;
                hasBounds = true;
            }
            else
            {
                bounds.Encapsulate(robotBounds);
            }
        }

        return hasBounds;
    }

    private static bool TryBuildTargetBounds(GameObject root, out Bounds bounds)
    {
        bounds = default;
        if (root == null)
        {
            return false;
        }

        if (TryBuildArticulationBounds(root, out bounds))
        {
            return true;
        }

        bool hasBounds = false;
        foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>(false))
        {
            if (!ShouldUseRendererForCameraBounds(root, renderer))
            {
                continue;
            }

            if (!hasBounds)
            {
                bounds = renderer.bounds;
                hasBounds = true;
            }
            else
            {
                bounds.Encapsulate(renderer.bounds);
            }
        }

        if (hasBounds && IsSaneRobotBounds(bounds))
        {
            return true;
        }

        bounds = BuildDefaultRobotBounds(root);
        return true;
    }

    private static bool TryBuildArticulationBounds(GameObject root, out Bounds bounds)
    {
        bounds = default;
        bool hasBounds = false;
        foreach (ArticulationBody body in root.GetComponentsInChildren<ArticulationBody>(false))
        {
            if (body == null || !body.gameObject.activeInHierarchy || !IsFinite(body.transform.position))
            {
                continue;
            }

            var pointBounds = new Bounds(body.transform.position, Vector3.one * 0.12f);
            if (!hasBounds)
            {
                bounds = pointBounds;
                hasBounds = true;
            }
            else
            {
                bounds.Encapsulate(pointBounds);
            }
        }

        return hasBounds && IsSaneRobotBounds(bounds);
    }

    private static Bounds BuildDefaultRobotBounds(GameObject root)
    {
        return new Bounds(root.transform.position + Vector3.up, new Vector3(1.2f, 2.0f, 1.2f));
    }

    private static bool IsSaneRobotBounds(Bounds bounds)
    {
        Vector3 center = bounds.center;
        Vector3 size = bounds.size;
        if (!IsFinite(center) || !IsFinite(size))
        {
            return false;
        }

        float maxSize = Mathf.Max(size.x, Mathf.Max(size.y, size.z));
        return maxSize >= 0.05f && maxSize <= 7.0f;
    }

    private static bool ShouldUseRendererForCameraBounds(GameObject root, Renderer renderer)
    {
        if (root == null ||
            renderer == null ||
            !renderer.enabled ||
            !renderer.gameObject.activeInHierarchy)
        {
            return false;
        }

        Transform transform = renderer.transform;
        while (transform != null && transform != root.transform)
        {
            string name = transform.name;
            if (name.IndexOf("RuntimeHUD", System.StringComparison.OrdinalIgnoreCase) >= 0 ||
                name.IndexOf("Drawer", System.StringComparison.OrdinalIgnoreCase) >= 0 ||
                name.IndexOf("VideoSurface", System.StringComparison.OrdinalIgnoreCase) >= 0 ||
                name.IndexOf("ColliderProxy", System.StringComparison.OrdinalIgnoreCase) >= 0)
            {
                return false;
            }

            transform = transform.parent;
        }

        Bounds bounds = renderer.bounds;
        if (!IsSaneRobotBounds(bounds))
        {
            return false;
        }

        // Only use renderer fallback for visual-only robots. Articulated robots
        // are handled from ArticulationBody positions first because replay/live
        // teleports the articulation root while the authored GameObject root may
        // stay far away in the scene.
        return renderer.GetComponentInParent<ArticulationBody>() == null;
    }

    private static bool IsFinite(float value)
    {
        return !float.IsNaN(value) && !float.IsInfinity(value);
    }

    private static bool IsFinite(Vector3 value)
    {
        return IsFinite(value.x) && IsFinite(value.y) && IsFinite(value.z);
    }

    private static Transform ResolveSelectedTransform(string selectedKey)
    {
        if (string.IsNullOrWhiteSpace(selectedKey))
        {
            return null;
        }

        if (MimicAgentRegistry.Instance != null)
        {
            IMimicAgent agent = MimicAgentRegistry.Instance.FindByKey(selectedKey);
            if (agent?.AgentGameObject != null && agent.AgentGameObject.activeInHierarchy)
            {
                return agent.AgentGameObject.transform;
            }
        }

        return ResolveSceneObjectByRobotKey(selectedKey);
    }

    private static Transform ResolveSceneObjectByRobotKey(string selectedKey)
    {
        string normalized = selectedKey?.Trim().ToLowerInvariant() ?? string.Empty;
        Transform agentRoot = ResolveAgentTransformByRobotKey(normalized);
        if (agentRoot != null)
        {
            return agentRoot;
        }

        string[] names;
        switch (normalized)
        {
            case "unitree_g1":
            case "g1":
                names = new[] { "G1", "unitree_g1", "g1_29dof_rev_1_0" };
                break;
            case "unitree_h1":
            case "h1":
                names = new[] { "H1", "h1", "unitree_h1" };
                break;
            case "x02lite":
            case "x02":
                names = new[] { "X02Lite", "x02lite", "X02" };
                break;
            case "openloong":
                names = new[] { "OpenLoong", "openloong" };
                break;
            default:
                names = new[] { selectedKey.Trim() };
                break;
        }

        foreach (string name in names)
        {
            Transform best = FindBestActiveTransformByName(name);
            if (best != null)
            {
                return best;
            }
        }

        return null;
    }

    private static Transform ResolveAgentTransformByRobotKey(string normalizedKey)
    {
        if (string.IsNullOrWhiteSpace(normalizedKey))
        {
            return null;
        }

        if (MimicAgentRegistry.Instance != null)
        {
            IMimicAgent registered = MimicAgentRegistry.Instance.FindByKey(normalizedKey);
            if (registered?.AgentGameObject != null && registered.AgentGameObject.activeInHierarchy)
            {
                return registered.AgentGameObject.transform;
            }
        }

        foreach (MonoBehaviour behaviour in FindObjectsOfType<MonoBehaviour>(true))
        {
            if (behaviour == null || !behaviour.gameObject.activeInHierarchy)
            {
                continue;
            }

            var agent = behaviour as IMimicAgent;
            if (agent == null)
            {
                continue;
            }

            try
            {
                if (string.Equals(agent.RobotKey?.Trim(), normalizedKey, System.StringComparison.OrdinalIgnoreCase) &&
                    agent.AgentGameObject != null &&
                    agent.AgentGameObject.activeInHierarchy)
                {
                    return agent.AgentGameObject.transform;
                }
            }
            catch (MissingReferenceException)
            {
                // Ignore stale scene objects during play-mode teardown.
            }
        }

        return null;
    }

    private static string ResolveRobotKeyForRoot(GameObject root)
    {
        if (root == null)
        {
            return "overview";
        }

        if (MimicAgentRegistry.Instance != null)
        {
            foreach (IMimicAgent agent in MimicAgentRegistry.Instance.All)
            {
                if (agent?.AgentGameObject == root && !string.IsNullOrWhiteSpace(agent.RobotKey))
                {
                    return agent.RobotKey.Trim();
                }
            }
        }

        string name = root.name.ToLowerInvariant();
        if (name.Contains("g1")) return "unitree_g1";
        if (name.Contains("h1")) return "unitree_h1";
        if (name.Contains("x02")) return "x02lite";
        if (name.Contains("openloong")) return "openloong";
        return "overview";
    }

    private static Transform FindActiveTransformByName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        foreach (var transform in FindObjectsOfType<Transform>(true))
        {
            if (transform == null ||
                !transform.gameObject.activeInHierarchy ||
                !string.Equals(transform.name, name, System.StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            return transform;
        }

        return null;
    }

    private static Transform FindBestActiveTransformByName(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            return null;
        }

        Transform first = null;
        foreach (var transform in FindObjectsOfType<Transform>(true))
        {
            if (transform == null ||
                !transform.gameObject.activeInHierarchy ||
                !string.Equals(transform.name, name, System.StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (first == null)
            {
                first = transform;
            }

            foreach (MonoBehaviour behaviour in transform.GetComponents<MonoBehaviour>())
            {
                if (behaviour is IMimicAgent)
                {
                    return transform;
                }
            }
        }

        return first;
    }

    private static string ResolveSelectedRobotKey()
    {
        var dropdownObject = GameObject.Find("RoboList");
        var dropdown = dropdownObject != null ? dropdownObject.GetComponent<TMP_Dropdown>() : null;
        if (dropdown == null || dropdown.options == null || dropdown.options.Count == 0)
        {
            return string.Empty;
        }

        int index = Mathf.Clamp(dropdown.value, 0, dropdown.options.Count - 1);
        string label = dropdown.options[index].text.Trim();
        if (string.Equals(label, "G1", System.StringComparison.OrdinalIgnoreCase))
        {
            return "unitree_g1";
        }
        if (string.Equals(label, "H1", System.StringComparison.OrdinalIgnoreCase))
        {
            return "unitree_h1";
        }
        if (string.Equals(label, "X02", System.StringComparison.OrdinalIgnoreCase) ||
            string.Equals(label, "X02Lite", System.StringComparison.OrdinalIgnoreCase))
        {
            return "x02lite";
        }
        if (string.Equals(label, "OpenLoong", System.StringComparison.OrdinalIgnoreCase))
        {
            return "openloong";
        }

        return label;
    }
}
