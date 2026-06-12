using UnityEngine;

[DisallowMultipleComponent]
[RequireComponent(typeof(Light))]
public sealed class DirectionalLightSunCycle : MonoBehaviour
{
    [Tooltip("Rotation speed around the X axis, in degrees per second.")]
    public float rotationSpeedDegreesPerSecond = 5f;

    [Tooltip("When enabled, rotate around the world X axis. Disable to rotate around the light's local X axis.")]
    public bool rotateInWorldSpace = true;

    private Light cachedLight;

    private void Awake()
    {
        cachedLight = GetComponent<Light>();
        if (cachedLight != null && cachedLight.type != LightType.Directional)
        {
            Debug.LogWarning($"{nameof(DirectionalLightSunCycle)} is intended for Directional Light objects.", this);
        }
    }

    private void Update()
    {
        float deltaDegrees = rotationSpeedDegreesPerSecond * Time.deltaTime;
        Space space = rotateInWorldSpace ? Space.World : Space.Self;
        transform.Rotate(Vector3.right, deltaDegrees, space);
    }
}
