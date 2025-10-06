import matplotlib.pyplot as plt
import numpy as np

# ---- Fill in your file paths here ----
flux1_file = "SII.npy"
flux2_file = "Halpha.npy"
flux3_file = "OIII.npy"

# ---- Options ----
per_channel = True  # If False, normalize globally
use_stretch = True  # Apply arcsinh stretch
Q = 3  # Strength of arcsinh stretch

# ---- Load fluxes ----
R = np.load(flux1_file)
G = np.load(flux2_file)
B = np.load(flux3_file)


# ---- Normalization ----
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


if per_channel:
    R_norm = normalize(R)
    G_norm = normalize(G)
    B_norm = normalize(B)
    rgb = np.stack([R_norm, G_norm, B_norm], axis=-1)
else:
    stacked = np.stack([R, G, B], axis=-1)
    vmin, vmax = stacked.min(), stacked.max()
    rgb = (stacked - vmin) / (vmax - vmin)

# ---- Optional stretch ----
if use_stretch:
    rgb = np.arcsinh(Q * rgb) / np.arcsinh(Q)

# ---- Clip and display ----
rgb = np.clip(rgb, 0, 1)


plt.figure(figsize=(8, 8))
plt.imshow(np.fliplr(rgb), origin="lower")
plt.axis("off")
plt.tight_layout()
plt.savefig("false_color_image.png", dpi=300, bbox_inches="tight")
plt.show()
