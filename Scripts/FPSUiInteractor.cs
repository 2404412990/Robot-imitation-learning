using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using System.Collections.Generic;
using TMPro;

public class FPSUiInteractor : MonoBehaviour
{
    public float interactDistance = 5f;
    public Image crosshairImage;
    
    private PointerEventData pointerData;
    private GameObject currentHoveredObject;

    void Start()
    {
        DisableKeyboardNavigation();
        pointerData = new PointerEventData(EventSystem.current);
    }

    void Update()
    {
        HandleRaycast();
        HandleClick();
        HandleScroll(); // 新增滚轮支持
    }

    void HandleRaycast()
	{
	    pointerData.position = new Vector2(Screen.width / 2, Screen.height / 2);

	    List<RaycastResult> results = new List<RaycastResult>();
	    EventSystem.current.RaycastAll(pointerData, results);

	    currentHoveredObject = null;

	    foreach (var result in results)
	    {
		float dist = Vector3.Distance(transform.position, result.gameObject.transform.position);
		if (dist <= interactDistance)
		{
		    currentHoveredObject = result.gameObject;

		    // ⭐关键：必须赋值
		    pointerData.pointerCurrentRaycast = result;

		    break;
		}
	    }

	    UpdateCrosshair(currentHoveredObject != null);
	}

    void HandleClick()
	{
	    if (currentHoveredObject == null) return;

	    if (Input.GetMouseButtonDown(0))
	    {
		pointerData.pressPosition = pointerData.position;
		pointerData.pointerPressRaycast = pointerData.pointerCurrentRaycast;

		ExecuteEvents.ExecuteHierarchy(currentHoveredObject, pointerData, ExecuteEvents.pointerDownHandler);
	    }

	    if (Input.GetMouseButtonUp(0))
	    {
		ExecuteEvents.ExecuteHierarchy(currentHoveredObject, pointerData, ExecuteEvents.pointerUpHandler);
		ExecuteEvents.ExecuteHierarchy(currentHoveredObject, pointerData, ExecuteEvents.pointerClickHandler);

		EventSystem.current.SetSelectedGameObject(ResolveKeyboardFocusTarget(currentHoveredObject));
	    }
	}

    // --- 新增：滚轮模拟 ---
    void HandleScroll()
    {
        if (currentHoveredObject == null) return;

        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) > 0.01f)
        {
            // 将滚轮数值传递给 UI 系统
            pointerData.scrollDelta = new Vector2(0, scroll * 30f); // 10f 是倍率，根据手感调整
            ExecuteEvents.ExecuteHierarchy(currentHoveredObject, pointerData, ExecuteEvents.scrollHandler);
        }
    }

    void UpdateCrosshair(bool highlight)
    {
        if (crosshairImage == null) return;
        crosshairImage.color = highlight ? Color.red : Color.white;
        crosshairImage.transform.localScale = highlight ? Vector3.one * 1.5f : Vector3.one;
    }

    static void DisableKeyboardNavigation()
    {
        var eventSystem = EventSystem.current;
        if (eventSystem == null) return;

        eventSystem.sendNavigationEvents = false;
    }

    static GameObject ResolveKeyboardFocusTarget(GameObject target)
    {
        if (target == null) return null;

        TMP_InputField tmpInput = target.GetComponentInParent<TMP_InputField>();
        if (tmpInput != null) return tmpInput.gameObject;

        InputField input = target.GetComponentInParent<InputField>();
        return input != null ? input.gameObject : null;
    }
}
