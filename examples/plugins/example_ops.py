"""Example MicroSegWorkbench plugin: extra image-only augmentations.

Use:  set MSW_PLUGINS=examples/plugins/example_ops.py   (Windows: set; Linux/macOS: export)
then start msw / msw-gui. The operations appear in step 2 under "Plugin operation".

Rules for plugin operations (enforced or recorded by the workbench):
* they receive and return a float32 (H, W, C) image in [0, 1] of the same shape;
* they never see the mask, so labels cannot be misaligned;
* all randomness must come from the ``rng`` argument (runs stay reproducible);
* the plugin file's SHA-256 is stored in every run that uses it.
"""

import numpy as np

from microseg_workbench.core.augment import gaussian_blur, register_photometric_op


def unsharp_mask(image: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Sharpen by adding back the difference to a blurred copy (random strength)."""
    amount = float(rng.uniform(0.0, params["max_amount"]))
    return image + amount * (image - gaussian_blur(image, params["sigma"]))


def vignette(image: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """Darken towards the corners, like uneven illumination in optical micrographs."""
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2
    strength = float(rng.uniform(0.0, params["max_strength"]))
    return image * (1.0 - strength * np.clip(r2 / 2, 0, 1))[:, :, None]


register_photometric_op("unsharp_mask", unsharp_mask, {"sigma": 1.5, "max_amount": 1.0},
                        "Random sharpening (unsharp mask)")
register_photometric_op("vignette", vignette, {"max_strength": 0.3}, "Uneven illumination towards the corners")
