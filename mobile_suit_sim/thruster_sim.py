"""
Physics simulation integration for stick figure with thrusters.
Uses MuJoCo's xfrc_applied for flexible runtime force application.
"""

from typing import Dict, List, Callable, Optional, Tuple
import numpy as np
import mujoco
from .thruster_ui import ThrusterSetup, ThrusterPlacement, BODY_PARTS


def get_body_id(model: mujoco.MjModel, body_name: str) -> int:
    """Get body ID from name, with error handling."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found in model")
    return bid


def get_site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    """Get site position in world coordinates."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found in model")
    return data.site_xpos[sid].copy()


class ThrusterSimulator:
    """
    Manages thruster forces on a stick figure during simulation.

    Usage:
        sim = ThrusterSimulator(model, data, setup)
        sim.set_thrust('thruster_1', 0.5)  # 50% power
        sim.apply_forces()  # Call each timestep
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        setup: ThrusterSetup
    ):
        self.model = model
        self.data = data
        self.setup = setup

        # Map thruster name -> body ID for fast lookup
        self._body_ids: Dict[str, int] = {}
        # Thrust levels (0-1 normalized)
        self._thrust_levels: Dict[str, float] = {}

        for thruster in setup.thrusters:
            bid = get_body_id(model, thruster.body_name)
            self._body_ids[thruster.name] = bid
            self._thrust_levels[thruster.name] = 0.0

    def set_thrust(self, name: str, level: float):
        """Set thrust level for a thruster (0.0 to 1.0)."""
        if name not in self._thrust_levels:
            raise ValueError(f"Unknown thruster: {name}")
        self._thrust_levels[name] = max(0.0, min(1.0, level))

    def set_all_thrust(self, level: float):
        """Set all thrusters to the same level."""
        for name in self._thrust_levels:
            self._thrust_levels[name] = max(0.0, min(1.0, level))

    def get_thrust(self, name: str) -> float:
        """Get current thrust level."""
        return self._thrust_levels.get(name, 0.0)

    def apply_forces(self):
        """
        Apply thruster forces to the simulation.
        Call this BEFORE mj_step().
        """
        # Clear external forces
        self.data.xfrc_applied[:] = 0

        for thruster in self.setup.thrusters:
            level = self._thrust_levels[thruster.name]
            if level <= 0:
                continue

            bid = self._body_ids[thruster.name]
            force_magnitude = level * thruster.max_force

            # Direction is in body frame, convert to world frame
            # Get body rotation matrix
            body_xmat = self.data.xmat[bid].reshape(3, 3)

            # Transform direction from body frame to world frame
            direction = np.array(thruster.direction)
            world_dir = body_xmat @ direction
            world_dir = world_dir / np.linalg.norm(world_dir)  # Normalize

            # Apply force (first 3 components are force, last 3 are torque)
            force = world_dir * force_magnitude
            self.data.xfrc_applied[bid, :3] += force

    def get_state(self) -> Dict:
        """Get current simulation state."""
        return {
            'time': self.data.time,
            'com_pos': self.data.subtree_com[1].copy(),  # Body 1 is usually the root
            'com_vel': self.data.cvel[1, 3:6].copy() if hasattr(self.data, 'cvel') else np.zeros(3),
            'thrust_levels': dict(self._thrust_levels),
        }


def create_control_function(
    sim: ThrusterSimulator,
    thrust_schedule: Dict[str, List[Tuple[int, int, float]]]
) -> Callable:
    """
    Create a control function from a thrust schedule.

    thrust_schedule format:
    {
        'thruster_1': [(start_step, end_step, level), ...],
        'thruster_2': [(start_step, end_step, level), ...],
    }

    Example:
        schedule = {
            'thruster_1': [(0, 100, 1.0), (200, 300, 0.5)],
            'thruster_2': [(50, 150, 0.8)],
        }
    """
    def control_fn(step: int, model: mujoco.MjModel, data: mujoco.MjData):
        # Reset all thrusters
        sim.set_all_thrust(0.0)

        # Apply scheduled thrust
        for name, schedule in thrust_schedule.items():
            for start, end, level in schedule:
                if start <= step < end:
                    sim.set_thrust(name, level)
                    break

        # Apply forces to simulation
        sim.apply_forces()

    return control_fn


def run_thruster_sim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    setup: ThrusterSetup,
    n_steps: int = 500,
    control_fn: Optional[Callable] = None,
) -> List[Dict]:
    """
    Run physics simulation with thrusters.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        setup: ThrusterSetup with thruster configurations
        n_steps: Number of simulation steps
        control_fn: Optional control function (step, model, data) -> None
                   If None, all thrusters fire at 100% the whole time

    Returns:
        List of state dictionaries for each step
    """
    sim = ThrusterSimulator(model, data, setup)
    states = []

    for step in range(n_steps):
        if control_fn:
            control_fn(step, model, data)
        else:
            # Default: all thrusters at full power
            sim.set_all_thrust(1.0)
            sim.apply_forces()

        # Step physics
        mujoco.mj_step(model, data)

        # Record state
        states.append(sim.get_state())

    return states


def load_stick_figure(model_path: str = "models/stick_figure.mjcf"):
    """Load the stick figure model."""
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    return model, data
