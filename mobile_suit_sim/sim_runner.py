import numpy as np
import mujoco
from mujoco import viewer as mj_viewer  # not used in Colab, but fine locally
import imageio
from typing import Callable, Optional

def simulate_and_render(model,
                        data,
                        n_steps: int = 300,
                        control_fn: Optional[Callable[[int, mujoco.MjModel, mujoco.MjData], None]] = None,
                        width: int = 640,
                        height: int = 480,
                        camera: str = "fixed",
                        output_path: str = "ambac_test.gif"):
    """
    Run a short sim and render frames to a GIF using offscreen rendering.
    control_fn(step, model, data) can apply controls each step.
    """
    # Create an offscreen renderer. In headless/Colab this typically uses EGL.
    renderer = mujoco.Renderer(model, width=width, height=height)

    frames = []

    # Optionally find a camera
    cam_id = None
    if isinstance(camera, str):
        try:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            if cam_id < 0:
                cam_id = None
        except Exception:
            cam_id = None
    elif isinstance(camera, int):
        cam_id = camera

    for step in range(n_steps):
        if control_fn is not None:
            control_fn(step, model, data)

        mujoco.mj_step(model, data)

        if cam_id is not None:
            renderer.update_scene(data, camera=cam_id)
        else:
            renderer.update_scene(data)

        img = renderer.render()
        frames.append(img)

    renderer.close()

    # Save to GIF
    imageio.mimsave(output_path, frames, fps=30)

    return output_path
