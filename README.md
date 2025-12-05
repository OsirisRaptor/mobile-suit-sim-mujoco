# Mobile Suit Simulation with MuJoCo

A physics-based mobile suit simulation framework using MuJoCo, designed for exploring AMBAC (Active Mass Balance Auto-Control) systems and thruster dynamics in zero-gravity environments.

## Overview

This project provides a modular Python framework for simulating mobile suits with parametric thruster configurations. It's designed to be:

- **GitHub-ready**: Clone and run immediately
- **Colab-friendly**: Full support for headless rendering and notebook workflows
- **Extensible**: Easy to add more limbs, thrusters, and control logic
- **Experimental**: Built for rapid iteration on thruster placement and control strategies

## Features

- Wireframe mobile suit model (torso + articulated limbs)
- Parametric thruster placement via Python configuration
- Zero-gravity physics simulation
- Offscreen rendering (EGL/OSMesa) for headless environments
- GIF output for visualization
- Colab notebook for interactive experimentation

## Repository Structure

```
mobile-suit-sim-mujoco/
│
├─ mobile_suit_sim/          # Main Python package
│  ├─ __init__.py
│  ├─ config.py              # ThrusterConfig and parameter definitions
│  ├─ model_builder.py       # MJCF loading and thruster positioning
│  └─ sim_runner.py          # Simulation stepping and rendering
│
├─ models/
│  └─ ambac_test.mjcf        # Minimal wireframe mobile suit model
│
├─ notebooks/
│  └─ colab_ambac_wireframe.ipynb   # Colab-ready demo notebook
│
├─ requirements.txt
└─ README.md
```

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/OsirisRaptor/mobile-suit-sim-mujoco.git
cd mobile-suit-sim-mujoco

# Install dependencies
pip install -r requirements.txt

# Run a quick test (optional)
python -c "from mobile_suit_sim import load_model; print('Import successful!')"
```

### Google Colab

The easiest way to get started is with Google Colab:

1. Open [`notebooks/colab_ambac_wireframe.ipynb`](notebooks/colab_ambac_wireframe.ipynb) in Colab
2. Run all cells
3. Experiment with thruster positions and control strategies

**Pro tip**: Enable GPU acceleration in Colab (Runtime > Change runtime type > Hardware accelerator > GPU) for better performance.

## Usage Example

```python
import os
from mobile_suit_sim.config import ThrusterConfig
from mobile_suit_sim.model_builder import load_model, apply_thruster_config
from mobile_suit_sim.sim_runner import simulate_and_render

# Load the MJCF model
model, data = load_model("models/ambac_test.mjcf")

# Define thruster configuration
thrusters = [
    ThrusterConfig(
        name="main",
        body_name="torso",
        pos=[0.0, -0.6, 0.0],      # Position in torso frame
        dir_world=[0.0, -1.0, 0.0], # Force direction
        max_force=2000.0
    )
]

# Apply configuration to model
apply_thruster_config(model, thrusters)

# Define control function
def control_fn(step, model, data):
    if step < 150:
        data.ctrl[:] = 1500.0  # Thrust on
    else:
        data.ctrl[:] = 0.0      # Thrust off

# Run simulation and render
simulate_and_render(
    model=model,
    data=data,
    n_steps=300,
    control_fn=control_fn,
    output_path="demo.gif"
)
```

## Model Details

The `ambac_test.mjcf` model includes:

- **Torso**: Box-shaped main body (0.5 × 0.3 × 0.8 m)
- **Left arm**: Capsule-shaped limb with shoulder pitch joint
- **Thruster site**: Configurable position on torso
- **General actuator**: Force application at thruster site
- **Zero gravity**: Perfect for testing AMBAC dynamics

### Extending the Model

To add more complexity:

1. **More thrusters**: Add additional `<site>` elements in the MJCF and corresponding `ThrusterConfig` entries
2. **More limbs**: Add bodies with joints (shoulders, elbows, hips, knees, etc.)
3. **Cameras**: Add `<camera>` elements for different viewing angles
4. **Sensors**: Add `<sensor>` elements to track orientation, velocity, etc.

## Workflow for Parameter Exploration

This framework is designed for rapid experimentation:

1. **Define a parameter vector**: `θ = [thruster1_pos, thruster2_pos, ...]`
2. **Update model**: Use `apply_thruster_config()` to move thrusters
3. **Simulate**: Run physics with different control strategies
4. **Measure**: Track attitude change, angular momentum, etc.
5. **Iterate**: Sweep over configurations to find optimal placements

This is essentially "Children of a Dead Earth but for mobile suits" — parameterize everything, simulate, and optimize.

## Dependencies

- **mujoco** (≥3.1.0): Physics simulation engine
- **dm-control** (≥1.0.0): DeepMind's control suite (includes PyMJCF)
- **imageio**: GIF/video output
- **matplotlib**: Debugging and visualization (optional)

## Future Directions

- Multi-thruster RCS systems
- Full mobile suit skeleton (head, arms, legs, backpack)
- AMBAC control policies (PID, MPC, RL)
- Fuel consumption models
- Trajectory optimization
- PyMJCF procedural model generation

## Contributing

This is an experimental research project. Feel free to fork, extend, and experiment!

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with [MuJoCo](https://mujoco.org/) by DeepMind. Inspired by realistic space combat mechanics and the engineering of Gundam's AMBAC system.
