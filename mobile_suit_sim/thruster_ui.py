"""
Jupyter widget-based UI for placing thrusters on a stick figure.
Supports bilateral symmetry (auto-mirror left<->right).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np

# Body part definitions with their mirrored counterparts
BODY_PARTS = {
    # Center line (no mirror)
    "torso": {"mirror": None, "sites": ["torso_front", "torso_back", "torso_left", "torso_right", "torso_top", "torso_bottom"]},
    "head": {"mirror": None, "sites": ["head_top", "head_front"]},
    "backpack": {"mirror": None, "sites": ["backpack_top", "backpack_bottom", "backpack_left", "backpack_right", "backpack_rear"]},

    # Left side (mirrors to right)
    "left_upper_arm": {"mirror": "right_upper_arm", "sites": ["left_upper_arm_site"]},
    "left_lower_arm": {"mirror": "right_lower_arm", "sites": ["left_lower_arm_site"]},
    "left_hand": {"mirror": "right_hand", "sites": ["left_hand_site"]},
    "left_upper_leg": {"mirror": "right_upper_leg", "sites": ["left_upper_leg_site"]},
    "left_lower_leg": {"mirror": "right_lower_leg", "sites": ["left_lower_leg_site"]},
    "left_foot": {"mirror": "right_foot", "sites": ["left_foot_site"]},

    # Right side (mirrors to left)
    "right_upper_arm": {"mirror": "left_upper_arm", "sites": ["right_upper_arm_site"]},
    "right_lower_arm": {"mirror": "left_lower_arm", "sites": ["right_lower_arm_site"]},
    "right_hand": {"mirror": "left_hand", "sites": ["right_hand_site"]},
    "right_upper_leg": {"mirror": "left_upper_leg", "sites": ["right_upper_leg_site"]},
    "right_lower_leg": {"mirror": "left_lower_leg", "sites": ["right_lower_leg_site"]},
    "right_foot": {"mirror": "left_foot", "sites": ["right_foot_site"]},
}

# Preset thrust directions
THRUST_DIRECTIONS = {
    "forward": (1, 0, 0),
    "backward": (-1, 0, 0),
    "left": (0, 1, 0),
    "right": (0, -1, 0),
    "up": (0, 0, 1),
    "down": (0, 0, -1),
}


@dataclass
class ThrusterPlacement:
    """A single thruster attached to a body part."""
    name: str
    body_name: str
    site_name: str
    direction: Tuple[float, float, float]  # Thrust direction in body frame
    max_force: float = 1000.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "body_name": self.body_name,
            "site_name": self.site_name,
            "direction": list(self.direction),
            "max_force": self.max_force,
        }


@dataclass
class ThrusterSetup:
    """Collection of thrusters with bilateral symmetry support."""
    thrusters: List[ThrusterPlacement] = field(default_factory=list)
    bilateral_symmetry: bool = True
    _counter: int = 0

    def add_thruster(
        self,
        body_name: str,
        direction: Tuple[float, float, float],
        max_force: float = 1000.0,
        site_name: Optional[str] = None,
    ) -> List[ThrusterPlacement]:
        """
        Add a thruster to a body part.
        If bilateral_symmetry is enabled and the body has a mirror,
        automatically adds a mirrored thruster.

        Returns list of added thrusters.
        """
        added = []

        if body_name not in BODY_PARTS:
            raise ValueError(f"Unknown body part: {body_name}")

        body_info = BODY_PARTS[body_name]

        # Use first site if not specified
        if site_name is None:
            site_name = body_info["sites"][0]

        # Create primary thruster
        self._counter += 1
        thruster = ThrusterPlacement(
            name=f"thruster_{self._counter}",
            body_name=body_name,
            site_name=site_name,
            direction=direction,
            max_force=max_force,
        )
        self.thrusters.append(thruster)
        added.append(thruster)

        # Create mirrored thruster if symmetry enabled
        if self.bilateral_symmetry and body_info["mirror"]:
            mirror_body = body_info["mirror"]
            mirror_info = BODY_PARTS[mirror_body]
            mirror_site = mirror_info["sites"][0]

            # Mirror the direction (flip Y axis for left/right)
            mirror_dir = (direction[0], -direction[1], direction[2])

            self._counter += 1
            mirror_thruster = ThrusterPlacement(
                name=f"thruster_{self._counter}",
                body_name=mirror_body,
                site_name=mirror_site,
                direction=mirror_dir,
                max_force=max_force,
            )
            self.thrusters.append(mirror_thruster)
            added.append(mirror_thruster)

        return added

    def remove_thruster(self, name: str) -> bool:
        """Remove a thruster by name."""
        for i, t in enumerate(self.thrusters):
            if t.name == name:
                self.thrusters.pop(i)
                return True
        return False

    def clear(self):
        """Remove all thrusters."""
        self.thrusters.clear()
        self._counter = 0

    def to_list(self) -> List[dict]:
        """Export as list of dicts for simulation."""
        return [t.to_dict() for t in self.thrusters]

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"ThrusterSetup ({len(self.thrusters)} thrusters, symmetry={'ON' if self.bilateral_symmetry else 'OFF'})"]
        for t in self.thrusters:
            lines.append(f"  - {t.name}: {t.body_name} @ {t.site_name}, dir={t.direction}, max={t.max_force}N")
        return "\n".join(lines)


def create_thruster_ui(setup: Optional[ThrusterSetup] = None):
    """
    Create interactive Jupyter widgets for thruster placement.

    Returns (setup, ui_container) where:
    - setup: ThrusterSetup object (modified in-place by UI)
    - ui_container: ipywidgets container to display
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        raise ImportError("ipywidgets required. Install with: pip install ipywidgets")

    if setup is None:
        setup = ThrusterSetup()

    # === Widgets ===

    # Body part selector
    body_dropdown = widgets.Dropdown(
        options=list(BODY_PARTS.keys()),
        value="torso",
        description="Body Part:",
        style={'description_width': '100px'}
    )

    # Direction selector
    direction_dropdown = widgets.Dropdown(
        options=list(THRUST_DIRECTIONS.keys()),
        value="backward",
        description="Direction:",
        style={'description_width': '100px'}
    )

    # Custom direction inputs
    dir_x = widgets.FloatText(value=0, description="Dir X:", style={'description_width': '50px'}, layout=widgets.Layout(width='120px'))
    dir_y = widgets.FloatText(value=0, description="Y:", style={'description_width': '30px'}, layout=widgets.Layout(width='100px'))
    dir_z = widgets.FloatText(value=0, description="Z:", style={'description_width': '30px'}, layout=widgets.Layout(width='100px'))

    use_custom_dir = widgets.Checkbox(value=False, description="Custom direction")

    # Force slider
    force_slider = widgets.FloatSlider(
        value=1000,
        min=100,
        max=5000,
        step=100,
        description="Max Force (N):",
        style={'description_width': '100px'},
        layout=widgets.Layout(width='400px')
    )

    # Bilateral symmetry toggle
    symmetry_toggle = widgets.Checkbox(
        value=setup.bilateral_symmetry,
        description="Bilateral Symmetry (auto-mirror)",
    )

    # Buttons
    add_btn = widgets.Button(description="Add Thruster", button_style="success", icon="plus")
    clear_btn = widgets.Button(description="Clear All", button_style="danger", icon="trash")

    # Output area for thruster list
    output = widgets.Output()

    # Status message
    status = widgets.HTML(value="<i>No thrusters added yet</i>")

    # === Callbacks ===

    def update_display():
        with output:
            clear_output(wait=True)
            if setup.thrusters:
                print(setup.summary())
            else:
                print("No thrusters configured.")
        status.value = f"<b>{len(setup.thrusters)}</b> thrusters configured"

    def on_add_click(b):
        body = body_dropdown.value

        if use_custom_dir.value:
            direction = (dir_x.value, dir_y.value, dir_z.value)
            # Normalize if non-zero
            mag = np.sqrt(sum(d**2 for d in direction))
            if mag > 0:
                direction = tuple(d/mag for d in direction)
            else:
                direction = (0, 0, -1)  # Default to down if zero
        else:
            direction = THRUST_DIRECTIONS[direction_dropdown.value]

        added = setup.add_thruster(
            body_name=body,
            direction=direction,
            max_force=force_slider.value,
        )

        update_display()

        # Show what was added
        names = [t.name for t in added]
        if len(names) == 2:
            status.value = f"Added <b>{names[0]}</b> + mirrored <b>{names[1]}</b>"
        else:
            status.value = f"Added <b>{names[0]}</b>"

    def on_clear_click(b):
        setup.clear()
        update_display()
        status.value = "<i>All thrusters cleared</i>"

    def on_symmetry_change(change):
        setup.bilateral_symmetry = change['new']

    def on_preset_direction_change(change):
        if not use_custom_dir.value:
            d = THRUST_DIRECTIONS[change['new']]
            dir_x.value, dir_y.value, dir_z.value = d

    # Connect callbacks
    add_btn.on_click(on_add_click)
    clear_btn.on_click(on_clear_click)
    symmetry_toggle.observe(on_symmetry_change, names='value')
    direction_dropdown.observe(on_preset_direction_change, names='value')

    # Initialize direction display
    on_preset_direction_change({'new': direction_dropdown.value})

    # === Layout ===

    custom_dir_box = widgets.HBox([use_custom_dir, dir_x, dir_y, dir_z])

    controls = widgets.VBox([
        widgets.HTML("<h3>Thruster Placement</h3>"),
        widgets.HBox([body_dropdown, direction_dropdown]),
        custom_dir_box,
        force_slider,
        symmetry_toggle,
        widgets.HBox([add_btn, clear_btn]),
        widgets.HTML("<hr>"),
        status,
        output,
    ])

    update_display()

    return setup, controls


def create_figure_diagram():
    """
    Create a matplotlib figure showing the stick figure body parts.
    Useful for reference when placing thrusters.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        raise ImportError("matplotlib required. Install with: pip install matplotlib")

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Body part positions (2D representation)
    positions = {
        "head": (0, 1.8),
        "torso": (0, 1.2),
        "backpack": (-0.3, 1.3),
        "left_upper_arm": (0.4, 1.4),
        "left_lower_arm": (0.7, 1.4),
        "left_hand": (0.95, 1.4),
        "right_upper_arm": (-0.4, 1.4),
        "right_lower_arm": (-0.7, 1.4),
        "right_hand": (-0.95, 1.4),
        "left_upper_leg": (0.15, 0.6),
        "left_lower_leg": (0.15, 0.2),
        "left_foot": (0.2, 0.0),
        "right_upper_leg": (-0.15, 0.6),
        "right_lower_leg": (-0.15, 0.2),
        "right_foot": (-0.2, 0.0),
    }

    # Draw body parts
    for name, (x, y) in positions.items():
        if "head" in name:
            circle = plt.Circle((x, y), 0.12, fill=False, color='blue', linewidth=2)
            ax.add_patch(circle)
        elif "torso" in name:
            rect = patches.Rectangle((x-0.15, y-0.25), 0.3, 0.5, fill=False, color='blue', linewidth=2)
            ax.add_patch(rect)
        elif "backpack" in name:
            rect = patches.Rectangle((x-0.08, y-0.15), 0.16, 0.3, fill=False, color='gray', linewidth=2)
            ax.add_patch(rect)
        elif "hand" in name:
            rect = patches.Rectangle((x-0.04, y-0.06), 0.08, 0.12, fill=False, color='orange', linewidth=2)
            ax.add_patch(rect)
        elif "foot" in name:
            rect = patches.Rectangle((x-0.08, y-0.03), 0.16, 0.06, fill=False, color='green', linewidth=2)
            ax.add_patch(rect)
        else:
            # Limb segments
            ax.plot([x-0.05, x+0.05], [y, y], 'b-', linewidth=8, solid_capstyle='round')

        # Label
        ax.annotate(name.replace("_", "\n"), (x, y), fontsize=7, ha='center', va='center')

    # Draw connections
    connections = [
        ("torso", "head"),
        ("torso", "left_upper_arm"),
        ("left_upper_arm", "left_lower_arm"),
        ("left_lower_arm", "left_hand"),
        ("torso", "right_upper_arm"),
        ("right_upper_arm", "right_lower_arm"),
        ("right_lower_arm", "right_hand"),
        ("torso", "left_upper_leg"),
        ("left_upper_leg", "left_lower_leg"),
        ("left_lower_leg", "left_foot"),
        ("torso", "right_upper_leg"),
        ("right_upper_leg", "right_lower_leg"),
        ("right_lower_leg", "right_foot"),
        ("torso", "backpack"),
    ]

    for a, b in connections:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1, alpha=0.3)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.3, 2.2)
    ax.set_aspect('equal')
    ax.set_title("Stick Figure Body Parts\n(Left side mirrors to Right)")
    ax.axis('off')

    plt.tight_layout()
    return fig, ax
