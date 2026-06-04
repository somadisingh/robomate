from __future__ import annotations

from typing import Any


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value.tolist()


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _class_name(names: Any, class_id: int | None) -> str | None:
    if class_id is None:
        return None
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def result_to_record(
    result: Any,
    frame_index: int,
    *,
    include_masks: bool,
    mask_format: str = "polygon_xyn",
) -> dict[str, Any]:
    names = getattr(result, "names", {}) or {}
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    orig_shape = getattr(result, "orig_shape", None)

    instances: list[dict[str, Any]] = []
    if boxes is not None:
        xyxy = _as_list(boxes.xyxy)
        confs = _as_list(boxes.conf) if getattr(boxes, "conf", None) is not None else [None] * len(xyxy)
        classes = _as_list(boxes.cls) if getattr(boxes, "cls", None) is not None else [None] * len(xyxy)

        polygons = None
        if include_masks and masks is not None and mask_format == "polygon_xyn":
            polygons = getattr(masks, "xyn", None)

        for index, coords in enumerate(xyxy):
            class_id = None if classes[index] is None else int(_scalar(classes[index]))
            record: dict[str, Any] = {
                "box_xyxy": [float(value) for value in coords],
                "confidence": None if confs[index] is None else _scalar(confs[index]),
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
            }
            if include_masks and polygons is not None and index < len(polygons):
                record["mask_polygon_xyn"] = [
                    [float(x), float(y)] for x, y in _as_list(polygons[index])
                ]
            instances.append(record)

    return {
        "frame_index": frame_index,
        "orig_shape": None if orig_shape is None else [int(value) for value in orig_shape],
        "instances": instances,
    }
