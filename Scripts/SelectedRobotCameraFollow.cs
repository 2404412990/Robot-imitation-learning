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
    [SerializeField] private float boundsDistanceMultiplier = 2.8f;
    [SerializeField] private float minViewDistance = 4.8f;
    [SerializeField] private float boundsHeightMultiplier = 0.25f;
    [SerializeField] private bool invertFrontDirection;

    private static readonly string[] CameraNames = { "Main Camera", "RightCamera", "BackCamera", "LeftCamera" };
    private static readonly string[] LegacyCameraNames = { "Main Camera", "Main Camera (1)", "Main Camera (2)", "Main Camera (3)" };
    private static int viewIndex;
    private static SelectedRobotCameraFollow primaryFollower;

    private readonly Camera[] viewCameras = new Camera[CameraNames.Length];
    private readonly Vector3[] desiredCameraPositions = new Vector3[CameraNames.Length];
    private readonly Quaternion[] desiredCameraRotations = new Quaternion[CameraNames.Length];
    private Vector3 followVelocity;
    private Transform lastTarget;
    private int lastViewIndex = -1;
    private bool transitionActive;
    private Vector3 transitionStartPosition;
    private Quaternion transitionStartRotation;
    private float transitionStartTime;

    public static bool FixedCameraModeActive => FindObjectOfType<SelectedRobotCameraFollow>(true) != null;

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

        if (TryResolveTarget(out GameObject targetRoot, out Bounds targetBounds))
        {
            Transform target = targetRoot.transform;
            bool targetChanged = lastTarget != target;
            bool viewChanged = lastViewIndex != viewIndex;
            if (targetChanged)
            {
                followVelocity = Vector3.zero;
                transitionActive = false;
                lastTarget = target;
            }
            lastViewIndex = viewIndex;

            UpdateCameraTransforms(target, targetBounds, targetChanged);
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
        Vector3 front = invertFrontDirection ? -target.forward : target.forward;
        if (front.sqrMagnitude < 0.001f)
        {
            front = Vector3.forward;
        }

        front.Normalize();
        Vector3 right = target.right.sqrMagnitude > 0.001f ? target.right.normalized : Vector3.right;
        Vector3[] directions =
        {
            front,
            right,
            -front,
            -right
        };

        float radius = Mathf.Max(0.8f, targetBounds.extents.magnitude);
        float distance = Mathf.Max(frontDistance, minViewDistance, radius * boundsDistanceMultiplier);
        float viewHeight = Mathf.Max(height, targetBounds.extents.y * 0.65f + boundsHeightMultiplier * radius);
        Vector3 center = targetBounds.center;
        Vector3 lookAt = center + Vector3.up * lookAtHeight;
        for (int i = 0; i < viewCameras.Length; i++)
        {
            Camera camera = viewCameras[i];
            Vector3 desired = center + directions[i] * distance + Vector3.up * viewHeight;
            Vector3 lookDirection = lookAt - desired;
            Quaternion desiredRotation = lookDirection.sqrMagnitude > 0.0001f
                ? Quaternion.LookRotation(lookDirection.normalized, Vector3.up)
                : Quaternion.identity;
            desiredCameraPositions[i] = desired;
            desiredCameraRotations[i] = desiredRotation;

            if (camera != null && i > 0)
            {
                camera.transform.SetPositionAndRotation(desired, desiredRotation);
            }
        }

        Camera renderCamera = viewCameras[0] != null ? viewCameras[0] : GetComponent<Camera>();
        if (renderCamera == null)
        {
            return;
        }

        int activeIndex = Mathf.Abs(viewIndex) % CameraNames.Length;
        Vector3 targetPosition = desiredCameraPositions[activeIndex];
        Quaternion targetRotation = desiredCameraRotations[activeIndex];

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
        if (startInput != null && startInput.CurrentSelectedRobotRoot != null)
        {
            targetRoot = startInput.CurrentSelectedRobotRoot;
            if (TryBuildTargetBounds(targetRoot, out bounds))
            {
                return true;
            }
        }

        string selectedKey = ResolveSelectedRobotKey();
        Transform selected = ResolveSelectedTransform(selectedKey);
        if (selected != null)
        {
            targetRoot = selected.gameObject;
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
                    return TryBuildTargetBounds(targetRoot, out bounds);
                }
            }
        }

        string[] fallbackNames = { "G1", "H1", "X02Lite", "OpenLoong" };
        foreach (string name in fallbackNames)
        {
            Transform transform = FindActiveTransformByName(name);
            if (transform != null)
            {
                targetRoot = transform.gameObject;
                return TryBuildTargetBounds(targetRoot, out bounds);
            }
        }

        return false;
    }

    private static bool TryBuildTargetBounds(GameObject root, out Bounds bounds)
    {
        bounds = default;
        if (root == null)
        {
            return false;
        }

        bool hasBounds = false;
        foreach (Renderer renderer in root.GetComponentsInChildren<Renderer>(true))
        {
            if (renderer == null || !renderer.enabled)
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

        if (hasBounds)
        {
            return true;
        }

        foreach (ArticulationBody body in root.GetComponentsInChildren<ArticulationBody>(true))
        {
            if (body == null)
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

        if (hasBounds)
        {
            return true;
        }

        bounds = new Bounds(root.transform.position + Vector3.up, new Vector3(1.2f, 2.0f, 1.2f));
        return true;
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
            Transform best = FindActiveTransformByName(name);
            if (best != null)
            {
                return best;
            }
        }

        return null;
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
