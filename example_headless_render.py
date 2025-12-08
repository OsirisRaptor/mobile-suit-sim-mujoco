#!/usr/bin/env python3
"""
Example: Headless rendering with EGL backend.

Run with:
    MUJOCO_GL=egl python example_headless_render.py

This script demonstrates how to use the mobile-suit-sim framework
in a headless environment (dev container, CI/CD, etc.) to render
simulation output as a GIF without a display.
"""

from mobile_suit_sim import (
    load_model,
    ThrusterConfig,
    apply_thruster_config,
    simulate_and_render,
)


def main():
    # Load the MJCF model
    model, data = load_model("models/ambac_test.mjcf")

    # Define thruster configuration
    thrusters = [
        ThrusterConfig(
            name="main",
            body_name="torso",
            pos=[0.0, -0.6, 0.0],      # Position in torso frame
            dir_world=[0.0, -1.0, 0.0],  # Force direction
            max_force=2000.0
        )
    ]

    # Apply configuration to model
    apply_thruster_config(model, thrusters)

    # Define control function: thrust for first 150 steps, then coast
    def control_fn(step, model, data):
        if step < 150:
            data.ctrl[:] = 1500.0  # Thrust on
        else:
            data.ctrl[:] = 0.0      # Thrust off

    # Run simulation and render to GIF
    output_file = simulate_and_render(
        model=model,
        data=data,
        n_steps=300,
        control_fn=control_fn,
        width=640,
        height=480,
        camera="fixed",
        output_path="example_headless_output.gif"
    )

    print(f"✓ Simulation complete. Output saved to: {output_file}")


if __name__ == "__main__":
    main()
