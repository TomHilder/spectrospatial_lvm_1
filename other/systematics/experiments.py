# Imports and config

import warnings
from glob import glob
from pathlib import Path

import astropy.constants as constants
import astropy.units as u
import cmasher as cmr
import jax
import matplotlib.pyplot as plt
import numpy as np
from loss import neg_ln_posterior
from lvm_lib import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from lvm_lib.fit_data.filtering import BAD_FLUX_THRESHOLD
from mask_plot import mask_near_points
from model_with_systematics import LVMModelNew as LVMModel
from modelling_lib import (
    ConstrainedParameter,
    Matern12,
    Matern32,
    OptimiserSchedule,
    Parameter,
    SpatialDataGeneric,
    SpatialDataLVM,
    build_model,
)
from numpy.random import default_rng
from plots_lib import COLORS
from systematics_schedule import all_phases

rng = default_rng(0)
jax.config.update("jax_enable_x64", True)
plt.style.use("plots_lib.custom")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


# LINE_CENTRE = 6562.817  # Halpha line centre in Angstroms
# LINE_CENTRE = 4658.5  # FeIII line centre in Angstroms
# LINE_CENTRE = 6584  # NII line centre in Angstroms
# LINE_CENTRE = 5007  # OIII line centre in Angstroms
LINE_CENTRE = 6716  # SII line centre in Angstroms
# LINE_CENTRE = 4471  # He recombination line centre in Angstroms
# LINE_CENTRE = 6678  # Other He recombination line centre in Angstroms
LINE_WINDOW_HALF = 8
λ_range = (LINE_CENTRE - LINE_WINDOW_HALF, LINE_CENTRE + LINE_WINDOW_HALF)

loc = Path("../../data/rosette/")
drp_files = glob(str(loc / "lvm*.fits"))
# drp_files = [
#     loc / "lvmSFrame-00030873.fits",
#     loc / "lvmSFrame-00030874.fits",
#     loc / "lvmSFrame-00030926.fits",
#     loc / "lvmSFrame-00030928.fits",
#     loc / "lvmSFrame-00030816.fits",
#     loc / "lvmSFrame-00030818.fits",
#     loc / "lvmSFrame-00030871.fits",
# ]
tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in drp_files])

# Hα
# config = DataConfig.from_tiles(
#     tiles,
#     λ_range,
#     normalise_F_scale=5e-12,
#     normalise_F_offset=0.0,
#     # normalise_F_offset=-1e-14,
#     F_range=(BAD_FLUX_THRESHOLD, 0.5e-13),
# )
# OIII
# config = DataConfig.from_tiles(
#     tiles,
#     λ_range,
#     normalise_F_scale=5e-12,
#     normalise_F_offset=0.0,
#     # normalise_F_offset=-1e-14,
#     F_range=(BAD_FLUX_THRESHOLD, 0.5e-13),
# )
# NII
# config = DataConfig.from_tiles(
#     tiles,
#     λ_range,
#     normalise_F_scale=1e-12,
#     normalise_F_offset=0.0,
#     # normalise_F_offset=-1e-14,
#     F_range=(BAD_FLUX_THRESHOLD, 0.5e-13),
# )
# SII
config = DataConfig.from_tiles(
    tiles,
    λ_range,
    normalise_F_scale=1e-12,
    normalise_F_offset=0.0,
    # normalise_F_offset=-1e-14,
    F_range=(BAD_FLUX_THRESHOLD, 0.5e-13),
)
# He
# config = DataConfig.from_tiles(
#     tiles,
#     λ_range,
#     normalise_F_scale=1e-15,
#     normalise_F_offset=0.0,
#     # normalise_F_offset=-1e-14,
#     # F_range=(BAD_FLUX_THRESHOLD, 1e-14),
# )
# Fe ????
# config = DataConfig.from_tiles(
#     tiles,
#     λ_range,
#     normalise_F_scale=1e-13,
#     normalise_F_offset=0.0,
#     # normalise_F_offset=-1e-14,
#     F_range=(BAD_FLUX_THRESHOLD, 1.2e-13),
# )
# Other He
# config = DataConfig.from_tiles(
#     tiles,
#     λ_range,
#     normalise_F_scale=5e-15,
#     normalise_F_offset=0.0,
#     # normalise_F_offset=-1e-14,
#     F_range=(BAD_FLUX_THRESHOLD, 1e-14),
# )
builder = FitDataBuilder(tiles, config)
fd = builder.build()


fig, ax = plt.subplots(figsize=[8, 8], layout="compressed")
ax.set_title(rf"Max flux per spaxel, $\lambda \in {λ_range}$")
cs = ax.scatter(
    fd.predict_α(fd.α), fd.predict_δ(fd.δ), c=np.nanmax(fd.flux, axis=0), s=5, vmax=0.3
)
plt.colorbar(cs, ax=ax, label=r"$F_{\rm max}$")
ax.set_aspect(1)
ax.set_xlabel(r"$\alpha$ [deg]")
ax.set_ylabel(r"$\delta$ [deg]")
ax.set_xlim(config.α_range[1], config.α_range[0])
ax.set_ylim(*config.δ_range)
plt.show()

fig, ax = plt.subplots(figsize=[8, 4], layout="compressed")
ax.set_title(rf"Average spectrum, $\lambda \in {λ_range}$")
ax.plot(fd.λ, np.nanmean(fd.predict_flux(fd.flux), axis=1))
ax.set_xlabel(r"$\lambda$ [${\rm \AA}$]")
plt.show()


mean_lsf_σ = np.nanmean(fd.lsf_σ[fd.lsf_σ.shape[0] // 2, :])

# Hyperparameters
# n_modes = (451, 451)
n_modes = (351, 351)
# n_modes = (151, 151)
# n_modes = (51, 51)
n_spaxels = len(fd.α)

# === Parameters ===

# Line centre
μ_line = Parameter(initial=LINE_CENTRE, fixed=True)

# Kernel parameters
length_kernel_params = dict(initial=3.0, lower=1e-2, upper=1e1, fixed=True)
var_kernel_params = dict(initial=1.0, lower=1e-5, upper=1e1, fixed=True)
shared_length_scale = ConstrainedParameter(**length_kernel_params)
A_variance = ConstrainedParameter(**var_kernel_params)
v_variance = ConstrainedParameter(**var_kernel_params)
σ_length_scale = ConstrainedParameter(**length_kernel_params)
σ_variance = ConstrainedParameter(**var_kernel_params)

# Nuisances
offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels), fixed=True)

# Measured values
v_bary = Parameter(initial=fd.v_bary, fixed=True)
initial_σ_lsf = 1.0 * fd.lsf_σ[fd.lsf_σ.shape[0] // 2, :]
σ_lsf = Parameter(initial=initial_σ_lsf, fixed=True)

# Systematics
v_syst = Parameter(initial=25.0, fixed=True)  # Systematic velocity

# === Kernels ===

A_kernel = Matern32(length_scale=shared_length_scale, variance=A_variance)
v_kernel = Matern32(length_scale=shared_length_scale, variance=v_variance)
σ_kernel = Matern32(length_scale=σ_length_scale, variance=σ_variance)


# Build the model
my_model = build_model(
    LVMModel,
    n_spaxels=n_spaxels,
    offsets=offsets,
    line_centre=μ_line,
    n_modes=n_modes,
    A_kernel=A_kernel,
    v_kernel=v_kernel,
    σ_kernel=σ_kernel,
    σ_lsf=σ_lsf,
    v_bary=v_bary,
    v_syst=v_syst,
)

# Save a "locked" copy of the initial model
init_model = my_model.get_locked_model()

# with plt.style.context("default"):
#     my_model.plot_model_graph()

schedule = OptimiserSchedule(
    model=my_model,
    loss_fn=neg_ln_posterior,
    phase_configs=all_phases,
)


schedule.run_all(
    λ=fd.λ,
    xy_data=fd.αδ_data,
    data=fd.flux,
    u_data=fd.u_flux,
    mask=fd.mask,
)

plt.figure()
plt.title("Loss history")
plt.plot(schedule.loss_history)
# plt.xscale("log")
plt.xlabel("Step")
plt.ylabel("neg. log posterior")
plt.show()

plt.figure()
plt.title("Final stage loss history")
plt.plot(-schedule.loss_histories[-1])
plt.xlabel("Step")
plt.ylabel("log posterior")
plt.yscale("log")
plt.tight_layout()
plt.show()

pred_model = schedule.model_history[-1].get_locked_model()


# ====== PLOTTING PREDICTIONS ======

# Plot velocity predictions on original points
fig, ax = plt.subplots(1, 2, figsize=[16, 8], layout="compressed", sharex=True, sharey=True)
ax[0].set_title("Radial velocity predictions")
ax[0].scatter(
    fd.predict_α(fd.α),
    fd.predict_δ(fd.δ),
    c=pred_model.line.v_obs(fd.αδ_data),
    s=0.8,
    cmap="RdBu_r",
    # vmin=-12,
    # vmax=12,
)
ax[1].set_title("Line width predictions")
ax[1].scatter(
    fd.predict_α(fd.α),
    fd.predict_δ(fd.δ),
    c=pred_model.line.σ2_obs(fd.αδ_data) ** 0.5,
    s=0.8,
    cmap="magma",
)
for a in ax:
    a.set_xlabel(r"$\alpha$ [deg]")
    a.set_ylabel(r"$\delta$ [deg]")
    a.set_xlim(fd.predict_α(fd.α).max(), fd.predict_α(fd.α).min())
    a.set_ylim(fd.predict_δ(fd.δ).min(), fd.predict_δ(fd.δ).max())
    a.set_aspect(1)

plt.colorbar(ax[0].collections[0], ax=ax[0], label=r"km s$^{-1}$")
plt.colorbar(ax[1].collections[0], ax=ax[1], label=r"$\AA$")
plt.show()


n_dense = 1600
α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)

α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)
αδ_dense = SpatialDataGeneric(
    α_dense.flatten(),
    δ_dense.flatten(),
    idx=np.arange(n_dense**2, dtype=int),
)

# Predict the model on a dense grid
A_pred = pred_model.line.A(αδ_dense).reshape((n_dense, n_dense))
v_pred = pred_model.line.v(αδ_dense).reshape((n_dense, n_dense))
μ_obs_pred = pred_model.line.μ_obs(fd.αδ_data)
μ_obs_mean = np.nanmedian(μ_obs_pred)
vσ_pred = np.abs(pred_model.line.vσ(αδ_dense)).reshape((n_dense, n_dense))
# vσ_pred = pred_model.line.vσ(αδ_dense).reshape((n_dense, n_dense))

# Convert things to original/useful units
α_plot = fd.predict_α(α_dense)
δ_plot = fd.predict_δ(δ_dense)
A_plot = fd.predict_flux(A_pred)

v_plot = np.array(v_pred)
σ_plot = vσ_pred  # already in km/s now


# Assemble a mask
mask = mask_near_points(
    α_dense_1D,
    δ_dense_1D,
    fd.α,
    fd.δ,
    threshold=0.1,  # 0.25 degrees
)


fig, ax = plt.subplots(1, 3, figsize=[14, 6], layout="compressed", dpi=200)

pcolormesh_kwargs = dict(shading="auto", rasterized=True, aa=True)

# Line flux
ax[0].set_title("Line flux")
c0 = ax[0].pcolormesh(
    α_plot, δ_plot, np.where(mask, A_plot / 1e-12, np.nan), cmap="viridis", **pcolormesh_kwargs
)
ax[1].set_title(r"Radial velocity")
vmed = np.nanmedian(v_plot)
c1 = ax[1].pcolormesh(
    α_plot,
    δ_plot,
    np.where(mask, v_plot, np.nan),
    cmap="RdBu_r",
    vmin=-13,
    vmax=+13,
    **pcolormesh_kwargs,
)
ax[2].set_title(r"Dispersion")
c2 = ax[2].pcolormesh(
    α_plot,
    δ_plot,
    np.where(mask, σ_plot, np.nan),
    cmap="magma",
    vmin=0,
    vmax=0.8 * σ_plot.max(),
    # vmin=σ_plot.min(),
    # vmax=0.9 * σ_plot.max(,)
    **pcolormesh_kwargs,
)

for i, a in enumerate(ax.flatten()):
    # a.set_aspect(1)
    # xspan = α_plot.max() - α_plot.min()
    # yspan = δ_plot.max() - δ_plot.min()
    # padax = 0.05 * max(xspan, yspan)
    # a.set_ylim(δ_plot.min() - padax, δ_plot.max() + padax)
    # a.set_xlim(α_plot.min() - padax, α_plot.max() + padax)
    a.set_xlim(α_plot.max(), α_plot.min())
    a.set_axisbelow(False)
    if i != 0:
        a.set_xticklabels([])
        a.set_yticklabels([])
    # Set major ticks every half degree using automatic locator
    a.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    a.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    # Set minor ticks every 0.25 degree
    a.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
    a.yaxis.set_minor_locator(plt.MultipleLocator(0.25))

ax[0].set_xlabel(r"$\alpha$ [deg]")
ax[0].set_ylabel(r"$\delta$ [deg]")

colorbar_kwargs = dict(location="bottom", pad=0.02)
plt.colorbar(c0, ax=ax[0], **colorbar_kwargs, label=r"$10^{-12}$ erg cm$^{-2}$ s$^{-1}$")
plt.colorbar(c1, ax=ax[1], **colorbar_kwargs, label=r"km s$^{-1}$")
plt.colorbar(c2, ax=ax[2], **colorbar_kwargs, label=r"km s$^{-1}$")
plt.savefig("rosette_Halpha.pdf", bbox_inches="tight")
plt.show()

print(f"A length_scale: {pred_model.line.A_raw.kernel.length_scale.val}")
print(f"A variance: {pred_model.line.A_raw.kernel.variance.val}")
print(f"v length_scale: {pred_model.line.v.kernel.length_scale.val}")
print(f"v variance: {pred_model.line.v.kernel.variance.val}")
# print(
#     f"λ0 offset: {(np.array(pred_model.line.λ0_all.val / LINE_CENTRE) * constants.c).to(u.km / u.s).value}"
# )
# print(f"λ0 length_scale: {pred_model.line.λ0.kernel.length_scale.val}")
# print(f"λ0 variance: {pred_model.line.λ0.kernel.variance.val}")
print(f"σ length_scale: {pred_model.line.vσ_raw.kernel.length_scale.val}")
print(f"σ variance: {pred_model.line.vσ_raw.kernel.variance.val}")

# Print the LSF initial guess and the final value before varying σ
# print(f"LSF initial guess: {initial_σ_lsf:.4f}")
# lsf_average_inferred = σ_pred.mean()
# print(f"LSF average inferred: {lsf_average_inferred:.4f}")

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=[12, 5])
imshow_kwargs = dict(cmap="RdBu", vmin=-3, vmax=3)
ax[0].imshow(pred_model.line.A_raw.coefficients.val.reshape(n_modes), **imshow_kwargs)
ax[1].imshow(pred_model.line.v.coefficients.val.reshape(n_modes), **imshow_kwargs)
ax[2].imshow(pred_model.line.vσ_raw.coefficients.val.reshape(n_modes), **imshow_kwargs)
plt.show()


λ_dense = np.linspace(fd.λ.min(), fd.λ.max(), n_dense)
plot_λ = λ_dense

# We need to use vmap to vectorise over the wavelengths
pred_flux = jax.vmap(pred_model, in_axes=(0, None))(λ_dense, fd.αδ_data)
pred_flux = fd.predict_flux(pred_flux)

fig, ax = plt.subplots(4, 3, figsize=[14, 9], layout="compressed", sharex=True, sharey=True)

ax_flat = ax.flatten()

for i, j in enumerate(rng.choice(n_spaxels, 12, replace=False)):
    # Make model predictions for the j-th spaxel

    norm = 1e-12
    ax_flat[i].errorbar(
        fd.λ,
        fd.predict_flux(fd._flux[:, j]) / norm,
        yerr=fd.predict_ivar(fd.i_var[:, j]) ** -0.5 / norm,
        zorder=1,
        label="Data",
        elinewidth=1,
        lw=0,
        marker=".",
        capsize=1,
    )
    ax_flat[i].plot(plot_λ, pred_flux[:, j] / norm, c=COLORS[1], zorder=0, label="Model")

ax[0, 0].legend(loc="upper left", frameon=False)

ax[-1, 0].set_xlabel(r"$\lambda$ [${\rm \AA}$]")
ax[-1, 0].set_ylabel(r"Scaled flux")

fig.suptitle("Model predictions for random spaxels")


plt.show()


print(μ_obs_mean, LINE_CENTRE)

print(schedule.loss_history[-1])
print(pred_model.line.v_syst.val)

# np.save("SII.npy", A_plot)
