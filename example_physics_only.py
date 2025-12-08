#!/usr/bin/env python3
"""
Example: Physics-only simulation (no rendering).

This script demonstrates how to use the step_only() function
for batch simulations, CI/CD pipelines, or parameter sweeps
without the overhead of rendering.
"""

from mobile_suit_sim import (
    load_model,
    ThrusterConfig,
    apply_thruster_config,
    step_only,
)
import numpy as np


def main():
    # Load the MJCF model
    model, data = load_model("models/ambac_test.mjcf")

    # Define thruster configuration
    thrusters = [
        ThrusterConfig(
            name="main",
            body_name="torso",
            pos=[0.0, -0.6, 0.0],
            dir_world=[0.0, -1.0, 0.0],
            max_force=2000.0
        )
    ]

    # Apply configuration to model
    apply_thruster_config(model, thrusters)

    # Define control function
    def control_fn(step, model, data):
        if step < 100:
            data.ctrl[:] = 1500.0
        else:
            data.ctrl[:] = 0.0

    # Run 300 steps of pure physics (no rendering)
    n_completed = step_only(
        model=model,
        data=data,
        n_steps=300,
        control_fn=control_fn
    )

    # Print final state
    print(f"✓ Completed {n_completed} simulation steps")
    print(f"  Final joint position: {data.qpos}")
    print(f"  Final joint velocity: {data.qvel}")
    print(f"  Final body COM position: {data.xpos[1][:3]}")  # Left upper arm COM


if __name__ == "__main__":
    main()
