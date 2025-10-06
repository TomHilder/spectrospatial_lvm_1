# Imports and config

import warnings
from pathlib import Path

import astropy.constants as constants
import astropy.units as u
import jax
import matplotlib.pyplot as plt
import numpy as np
from loss import neg_ln_posterior
from lvm_lib import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from lvm_lib.fit_data.filtering import BAD_FLUX_THRESHOLD
from model import LVMModel
from modelling_lib import (
    ConstrainedParameter,
    Matern12,
    Matern32,
    Matern52,
    OptimiserSchedule,
    Parameter,
    build_model,
)
from modelling_lib import (
    SpatialDataGeneric as SpatialData,
)
from numpy.random import default_rng
from plots_lib import COLORS
from simple_schedule import all_phases

rng = default_rng(0)
jax.config.update("jax_enable_x64", True)
plt.style.use("plots_lib.custom")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


LINE_CENTRE = 4471  # He recombination line centre in Angstroms
LINE_WINDOW_HALF = 8
λ_range = (LINE_CENTRE - LINE_WINDOW_HALF, LINE_CENTRE + LINE_WINDOW_HALF)


loc = Path("../../data/rosette/")
drp_files = [
    loc / "lvmSFrame-00030873.fits",
    loc / "lvmSFrame-00030874.fits",
    loc / "lvmSFrame-00030926.fits",
    loc / "lvmSFrame-00030928.fits",
    loc / "lvmSFrame-00030816.fits",
    loc / "lvmSFrame-00030818.fits",
    loc / "lvmSFrame-00030871.fits",
]
tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in drp_files])

config = DataConfig.from_tiles(
    tiles,
    λ_range,
    normalise_F_scale=5e-14,
    normalise_F_offset=0.0,
    # normalise_F_offset=-1e-14,
    F_range=(BAD_FLUX_THRESHOLD, 1e-14),
)
builder = FitDataBuilder(tiles, config)
fd = builder.build()


fig, ax = plt.subplots(figsize=[8, 8], layout="compressed")
ax.set_title(rf"Average flux per spaxel, $\lambda \in {λ_range}$")
cs = ax.scatter(fd.predict_α(fd.α), fd.predict_δ(fd.δ), c=np.nanmax(fd.flux, axis=0), s=5)
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
n_modes = (151, 151)
n_spaxels = len(fd.α)

# Line peak
shared_length_scale = ConstrainedParameter(initial=1.0, lower=1e-8, upper=1e8)
A_variance = ConstrainedParameter(initial=1.0, lower=1e-8, upper=1e8)
A_kernel = Matern12(length_scale=shared_length_scale, variance=A_variance)

# Redshift/blueshift
λ0_all = Parameter(dims=(1,))
# λ0_length_scale = ConstrainedParameter(initial=1.0, lower=1e-8, upper=1e8)
λ0_variance = ConstrainedParameter(initial=1.0, lower=1e-8, upper=1e8)
λ0_kernel = Matern12(length_scale=shared_length_scale, variance=λ0_variance)

# Line width
σ_length_scale = ConstrainedParameter(initial=1.0, lower=1e-8, upper=1e8)
σ_variance = ConstrainedParameter(initial=1.0, lower=1e-8, upper=1e8)
σ_kernel = Matern12(length_scale=σ_length_scale, variance=σ_variance)
initial_σ_lsf = 1.0 * mean_lsf_σ
σ_mean = ConstrainedParameter(
    initial=initial_σ_lsf, lower=0.8 * mean_lsf_σ, upper=1.2 * mean_lsf_σ
)

# Nuisance offsets
offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels))

# Build the model
my_model = build_model(
    LVMModel,
    n_spaxels=n_spaxels,
    n_modes=n_modes,
    A_kernel=A_kernel,
    λ0_kernel=λ0_kernel,
    λ0_all=λ0_all,
    σ_kernel=σ_kernel,
    σ_lsf=σ_mean,
    offsets=offsets,
)

# Finalise the initialisation by setting coefficients
my_model = my_model.set(
    [
        "line.A.coefficients",
        "line.λ0.coefficients",
        "line.σ.coefficients",
    ],
    [
        rng.standard_normal(n_modes),
        np.zeros(n_modes),
        np.zeros(n_modes),
    ],
)

schedule = OptimiserSchedule(
    model=my_model,
    loss_fn=neg_ln_posterior,
    phase_configs=all_phases,
)

schedule.run_all(
    λ=fd.λ - LINE_CENTRE,
    xy_data=fd.αδ_data,
    data=fd.flux,
    u_data=fd.u_flux,
    mask=fd.mask,
)

plt.figure()
plt.plot(schedule.loss_history)
# plt.xscale("log")
plt.show()

pred_model = schedule.model_history[-1].get_locked_model()


n_dense = 800
α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)

α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)
# SpatialData is just a convenience object for some of the jax stuff
αδ_dense = SpatialData(
    α_dense.flatten(),
    δ_dense.flatten(),
    idx=np.arange(n_dense**2, dtype=int),
)

# Predict the model on a dense grid ("positive" gives us the model subject to the positivity constraints on A and σ)
A_pred = pred_model.line.A_positive(αδ_dense).reshape((n_dense, n_dense))
λ0_pred = pred_model.line.λ0(αδ_dense).reshape((n_dense, n_dense))
σ_pred = pred_model.line.σ_positive(αδ_dense).reshape((n_dense, n_dense))
# σ_pred = pred_model.line.σ_positive(αδ_dense) * np.ones_like(A_pred)

# Convert things to original/useful units
α_plot = fd.predict_α(α_dense)
δ_plot = fd.predict_δ(δ_dense)
A_plot = fd.predict_flux(A_pred)

λ0_plot = (np.array(λ0_pred / LINE_CENTRE) * constants.c).to(u.km / u.s).value
σ_plot = (np.array(σ_pred / LINE_CENTRE) * constants.c).to(u.km / u.s).value

fig, ax = plt.subplots(1, 3, figsize=[14, 6], layout="compressed", dpi=100)

pcolormesh_kwargs = dict(shading="auto", rasterized=True, aa=True)

# Line flux
ax[0].set_title("Line flux")
c0 = ax[0].pcolormesh(
    α_plot, δ_plot, A_plot / 1e-12, vmax=0.08, cmap="viridis", **pcolormesh_kwargs
)
ax[1].set_title(r"Radial velocity")
c1 = ax[1].pcolormesh(
    α_plot,
    δ_plot,
    λ0_plot,
    cmap="RdBu_r",
    vmin=-15,
    vmax=15,
    **pcolormesh_kwargs,
)
ax[2].set_title(r"Line width")
c2 = ax[2].pcolormesh(
    α_plot,
    δ_plot,
    σ_plot,
    cmap="magma",
    vmin=σ_plot.min(),
    vmax=σ_plot.max(),
    **pcolormesh_kwargs,
)

for i, a in enumerate(ax.flatten()):
    a.set_aspect(1)
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

plt.show()

print(f"A length_scale: {pred_model.line.A.kernel.length_scale.val}")
print(f"A variance: {pred_model.line.A.kernel.variance.val}")
print(
    f"λ0 offset: {(np.array(pred_model.line.λ0_all.val / LINE_CENTRE) * constants.c).to(u.km / u.s).value}"
)
print(f"λ0 length_scale: {pred_model.line.λ0.kernel.length_scale.val}")
print(f"λ0 variance: {pred_model.line.λ0.kernel.variance.val}")
print(f"σ length_scale: {pred_model.line.σ.kernel.length_scale.val}")
print(f"σ variance: {pred_model.line.σ.kernel.variance.val}")

# Print the LSF initial guess and the final value before varying σ
print(f"LSF initial guess: {initial_σ_lsf:.4f}")
lsf_average_inferred = σ_pred.mean()
print(f"LSF average inferred: {lsf_average_inferred:.4f}")

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=[12, 5])
imshow_kwargs = dict(cmap="RdBu", vmin=-3, vmax=3)
ax[0].imshow(pred_model.line.A.coefficients.val.reshape(n_modes), **imshow_kwargs)
ax[1].imshow(pred_model.line.λ0.coefficients.val.reshape(n_modes), **imshow_kwargs)
ax[2].imshow(pred_model.line.σ.coefficients.val.reshape(n_modes), **imshow_kwargs)
plt.show()


λ_dense = np.linspace(fd.λ.min(), fd.λ.max(), n_dense) - LINE_CENTRE
plot_λ = λ_dense + LINE_CENTRE

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
