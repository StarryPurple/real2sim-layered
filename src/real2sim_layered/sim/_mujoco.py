"""Thin typed adapter for MuJoCo C-extension dynamic exports."""

from __future__ import annotations

from typing import Any, cast

import mujoco

_mj: Any = cast(Any, mujoco)


def load_mjcf_model(xml: str) -> Any:
    """Compile an MJCF XML string into a `mujoco.MjModel`."""
    return _mj.MjModel.from_xml_string(xml)


def new_mj_data(model: Any) -> Any:
    """Allocate a `mujoco.MjData` instance for `model`."""
    return _mj.MjData(model)


def mj_step(model: Any, data: Any) -> None:
    """Advance MuJoCo simulation by one timestep."""
    _mj.mj_step(model, data)


def mj_forward(model: Any, data: Any) -> None:
    """Recompute MuJoCo kinematics and derived state without advancing time."""
    _mj.mj_forward(model, data)


def joint_qpos_adr(model: Any, joint_name: str) -> int | None:
    """Return the starting `qpos` index for a named joint, or None if absent."""
    jnt_id = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_JOINT, joint_name))
    return int(model.jnt_qposadr[jnt_id]) if jnt_id >= 0 else None


def geom_name_id(model: Any, geom_name: str) -> int | None:
    """Return the geom ID for `geom_name`, or None if absent."""
    gid = int(_mj.mj_name2id(model, _mj.mjtObj.mjOBJ_GEOM, geom_name))
    return gid if gid >= 0 else None
