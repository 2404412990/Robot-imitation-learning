using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.EventSystems;

public class Move : MonoBehaviour
{
    [Header("移动设置")]
    public float moveSpeed = 5f;
    public CharacterController controller;

    [Header("视角设置")]
    public float mouseSensitivity = 200f;
    public Transform cameraTransform; 

    [Header("物理设置")]
    public float gravity = -9.81f;    // 重力加速度
    private Vector3 velocity;         // 当前的速度（包含垂直速度）

    private float xRotation = 0f;
    private bool cursorLocked = false; 

    void Awake()
    {
        // 自动容错：如果没有拖入相机，尝试查找主相机
        if (cameraTransform == null && Camera.main != null)
        {
            cameraTransform = Camera.main.transform;
        }
        
        if (controller == null)
        {
            controller = GetComponent<CharacterController>();
        }
    }

    void Start()
    {
        // 进入场景时自动锁定鼠标
        if (SelectedRobotCameraFollow.FixedCameraModeActive)
        {
            SetCursorState(false);
            enabled = false;
            return;
        }

        SetCursorState(true);
    }

    void Update()
    {
        HandleEscape();

        // 只有在鼠标锁定的情况下才允许移动和旋转
        if (cursorLocked)
        {
            ClearUiFocusForMovement();
            HandleMovement();
            HandleLook();
        }
    }

    void HandleMovement()
    {
        if (controller == null) return;

        // 1. 处理水平移动
        float x = Input.GetAxis("Horizontal");
        float z = Input.GetAxis("Vertical");
        Vector3 move = transform.right * x + transform.forward * z;
        controller.Move(move * moveSpeed * Time.deltaTime);

        // 2. 处理重力逻辑
        // 如果在地面上，重力速度重置（给个小的向下力 -2f 是为了让 isGrounded 检测更稳定）
        if (controller.isGrounded && velocity.y < 0)
        {
            velocity.y = -2f; 
        }

        // 自由落体公式：v = g * t
        velocity.y += gravity * Time.deltaTime;

        // 应用垂直位移：Δx = v * t
        controller.Move(velocity * Time.deltaTime);
    }

    void HandleLook()
    {
        if (cameraTransform == null) return;

        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity * Time.deltaTime;
        float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity * Time.deltaTime;

        transform.Rotate(Vector3.up * mouseX);

        xRotation -= mouseY;
        xRotation = Mathf.Clamp(xRotation, -90f, 90f);
        cameraTransform.localRotation = Quaternion.Euler(xRotation, 0f, 0f);
    }

    void HandleEscape()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }

    // 统一管理鼠标状态
    void ClearUiFocusForMovement()
    {
        var eventSystem = EventSystem.current;
        if (eventSystem == null || eventSystem.currentSelectedGameObject == null) return;
        if (!IsMovementInputPressed()) return;

        eventSystem.SetSelectedGameObject(null);
    }

    static bool IsMovementInputPressed()
    {
        return Input.GetKey(KeyCode.W) || Input.GetKey(KeyCode.A) ||
               Input.GetKey(KeyCode.S) || Input.GetKey(KeyCode.D) ||
               Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.DownArrow) ||
               Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.RightArrow);
    }

    public void SetCursorState(bool locked)
    {
        cursorLocked = locked;
        if (locked)
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }
        else
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }
    }

    void OnDisable()
    {
        // 离开场景（对象销毁）时必须释放鼠标，否则下个场景（如菜单）鼠标会消失
        SetCursorState(false);
    }
}
