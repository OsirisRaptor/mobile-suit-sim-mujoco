from typing import Dict, List
import mujoco
from .config import ThrusterConfig

def load_model(mjcf_path: str):
    """Load an MJCF model from file."""
    with open(mjcf_path, "r") as f:
        xml = f.read()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data

def apply_thruster_config(model: mujoco.MjModel,
                          thrusters: List[ThrusterConfig],
                          site_name_prefix: str = "thruster_") -> Dict[str, int]:
    """
    Update site positions for thrusters in the compiled model.
    Returns a mapping from thruster name -> site id.
    """
    site_ids: Dict[str, int] = {}
    for cfg in thrusters:
        site_name = f"{site_name_prefix}{cfg.name}"
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise ValueError(f"Site {site_name} not found in model.")
        # Update site position (in parent body frame)
        model.site_pos[sid, :] = cfg.pos
        site_ids[cfg.name] = sid

    return site_ids
