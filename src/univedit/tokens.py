"""Modality special tokens and the task they belong to: <T_IMG> (i2i, ti_i2i),
<T_VID> (v2v, tiv2v) and <T_3D> (3d23d)."""

from typing import Optional


T_IMG = "<T_IMG>"
T_VID = "<T_VID>"
T_3D = "<T_3D>"

SPECIAL_TOKENS = (T_IMG, T_VID, T_3D)

TASK_TYPE_TO_SPECIAL_TOKEN = {
    "i2i": T_IMG,
    "ti_i2i": T_IMG,
    "v2v": T_VID,
    "tiv2v": T_VID,
    "3d23d": T_3D,
}

SUPPORTED_TASKS = tuple(TASK_TYPE_TO_SPECIAL_TOKEN)


def get_special_token_from_task(
    task_type: str,
    *,
    strict: bool = False,
) -> Optional[str]:
    """Modality special token of a task type; an unknown task gives None, or
    ValueError with ``strict``."""
    if not isinstance(task_type, str) or not task_type:
        if strict:
            raise ValueError("task_type must be a non-empty string.")
        return None

    token = TASK_TYPE_TO_SPECIAL_TOKEN.get(task_type)
    if token is None and strict:
        raise ValueError(f"No special token is defined for task '{task_type}'.")

    return token
