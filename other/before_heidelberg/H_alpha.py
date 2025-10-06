from glob import glob
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from lvm_lib.config.data_config import DataConfig
from lvm_lib.data.tile import LVMTile, LVMTileCollection
from lvm_lib.fit_data.builder import FitDataBuilder
from lvm_lib.fit_data.filtering import BAD_FLUX_THRESHOLD
from model import LVMModel, σ_LOWER
from modelling_lib import (
    ConstrainedParameter,
    Matern12,
    Matern32,
    Matern52,
    OptimiserFrame,
    Parameter,
    SpatialData,
    SquaredExponential,
    build_model,
)
from modelling_lib.model.parameter import l_bounded_inv
from numpy.random import default_rng
from optax import adam, lbfgs

rng = default_rng(0)

jax.config.update("jax_enable_x64", True)

plt.style.use("plots_lib.custom")

# ======= Reading the data =======

# drp_files = ["../../data/rosette/lvmSFrame-00030873.fits"]
# drp_files = glob("../../data/rosette/early_science/lvm*.fits")
drp_files = glob("../../data/rosette/lvm*.fits")
# drp_files = ["../../data/rosette/lvmSFrame-00029618.fits"]

tiles = LVMTileCollection.from_tiles(
    [LVMTile.from_file(Path(f)) for f in drp_files],
)


# LINE_CENTRE = 6583.45  # NII
LINE_CENTRE = 6562.85  # Hα


λ_range = (LINE_CENTRE - 8, LINE_CENTRE + 8)
config = DataConfig.from_tiles(
    tiles,
    λ_range,
    normalise_F_scale=1e-12,
    normalise_F_offset=-1e-14,
    F_range=(BAD_FLUX_THRESHOLD, 0.5e-13),
    # F_range=(BAD_FLUX_THRESHOLD, np.inf),
)


builder = FitDataBuilder(tiles, config)
fd = builder.build()


plt.figure(dpi=120, figsize=(8, 8))
plt.scatter(
    fd.predict_α(fd.α),
    fd.predict_δ(fd.δ),
    c=np.nanmean(fd.flux, axis=0),
    s=2,
    # s=56,
    cmap="viridis",
    vmin=0.0,
    vmax=0.1,
)
plt.colorbar()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis("scaled")
plt.show()

plt.figure(dpi=150)
plt.plot(fd.λ - LINE_CENTRE, np.nanmean(fd.flux, axis=1))

# ======= Instantiate and initialise the model =======

n_spaxels = len(fd.α)
n_modes = (351, 351)

# A_length_scale = ConstrainedParameter(initial=0.8, fixed=False, lower=0.01, upper=5)
# λ0_length_scale = ConstrainedParameter(initial=0.8, fixed=False, lower=0.01, upper=5)
# σ_length_scale = ConstrainedParameter(initial=0.8, fixed=False, lower=0.01, upper=5)
# A_variance = ConstrainedParameter(initial=0.01, fixed=False, lower=1e-5, upper=5)
# λ0_variance = ConstrainedParameter(initial=0.01, fixed=False, lower=1e-5, upper=5)
# σ_variance = ConstrainedParameter(initial=0.01, fixed=False, lower=1e-5, upper=5)

# A_length_scale = ConstrainedParameter(initial=1.0, fixed=False, lower=0.01, upper=5)
shared_length_scale = ConstrainedParameter(initial=1.0, fixed=False, lower=0.01, upper=5)
σ_length_scale = ConstrainedParameter(initial=0.6, fixed=True, lower=0.1, upper=5)
A_variance = ConstrainedParameter(initial=0.1, fixed=False, lower=1e-5, upper=5)
λ0_variance = ConstrainedParameter(initial=0.01, fixed=False, lower=1e-5, upper=5)
σ_variance = ConstrainedParameter(initial=0.01, fixed=True, lower=1e-5, upper=5)

chosen_kernel = Matern12
A_kernel = chosen_kernel(length_scale=shared_length_scale, variance=A_variance)
λ0_kernel = chosen_kernel(length_scale=shared_length_scale, variance=λ0_variance)
σ_kernel = Matern32(length_scale=σ_length_scale, variance=σ_variance)

lsf_i = np.nanmedian(fd.lsf_σ)


def gaussian(x, A, σ):
    return A * jnp.exp(-0.5 * (x / σ) ** 2)


plt.plot(fd.λ - LINE_CENTRE, gaussian(fd.λ - LINE_CENTRE, 0.2, lsf_i), label="LSF")
plt.show()


σ_mean = Parameter(initial=l_bounded_inv(lsf_i, σ_LOWER), fixed=False)

offsets = Parameter(dims=(n_spaxels,), fixed=False, initial=0.02 * np.ones(n_spaxels))

# Initialise the model
my_model = build_model(
    LVMModel,
    n_spaxels=n_spaxels,
    n_modes=n_modes,
    A_kernel=A_kernel,
    λ0_kernel=λ0_kernel,
    σ_kernel=σ_kernel,
    σ_mean=σ_mean,
    offsets=offsets,
)

# Fix the \sigma coefficients
my_model = eqx.tree_at(lambda m: m.line.σ.gp.coefficients.fix, my_model, True)

# with plt.style.context("default"):
#     my_model.plot_model_graph()

# Initialise coefficients randomly
my_model = my_model.set(
    ["line.A.coefficients", "line.λ0.coefficients", "line.σ.gp.coefficients"],
    [
        rng.standard_normal(n_modes),
        rng.standard_normal(n_modes),
        0.0 * rng.standard_normal(n_modes),
    ],
)

# Lock a copy of the initalised model
init_model_locked = my_model.get_locked_model()


nx_dense = 800
ny_dense = 800
x_dense_ = jnp.linspace(fd.α.min(), fd.α.max(), nx_dense)
y_dense_ = jnp.linspace(fd.δ.min(), fd.δ.max(), ny_dense)
x_dense, y_dense = jnp.meshgrid(x_dense_, y_dense_)
x_dense = x_dense.flatten()
y_dense = y_dense.flatten()
xy_dense = SpatialData(x=x_dense, y=y_dense, indices=jnp.arange(x_dense.size))

xy_shape = (nx_dense, ny_dense)

model_init = jax.vmap(init_model_locked, in_axes=(0, None))(fd.λ - LINE_CENTRE, xy_dense)

plt.figure(dpi=120, figsize=(8, 8))
plt.title("Initial guess's mean flux")
plt.scatter(
    fd.predict_α(x_dense),
    fd.predict_δ(y_dense),
    c=np.nanmean(model_init, axis=0),
    s=2,
    # s=56,
    cmap="viridis",
    vmin=0.0,
    vmax=0.1,
)
plt.colorbar()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.axis("scaled")
plt.show()


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    # Model predictions
    pred = jax.vmap(model, in_axes=(0, None))(λ, xy_data)
    # Likelihood
    ln_like = jnp.sum(
        jnp.where(
            mask,
            jax.scipy.stats.norm.logpdf(x=pred, loc=data, scale=u_data),
            0.0,
        )
    )
    ln_prior = (
        model.line.A.prior_logpdf().sum()
        + model.line.λ0.prior_logpdf().sum()
        + model.line.σ.gp.prior_logpdf().sum()
    )
    return -1 * (ln_like + ln_prior)


opt_frame1 = OptimiserFrame(
    model=my_model,
    loss_fn=neg_ln_posterior,
    optimiser=lbfgs(),
)


# ======= Optimise the model =======

n_steps = 200

# First round of optimisation with lbfgs
opt_model = opt_frame1.run(
    n_steps,
    λ=fd.λ - LINE_CENTRE,
    xy_data=fd.αδ_data,
    data=fd.flux,
    u_data=fd.u_flux,
    mask=fd.mask,
)

opt_model_intermed = opt_model.get_locked_model()
print("Intermediate model parameters:")
print(f"A variance: {opt_model_intermed.line.A.kernel.variance.val[0]:.3e}")
print(f"λ0 variance: {opt_model_intermed.line.λ0.kernel.variance.val[0]:.3e}")
print(f"σ variance: {opt_model_intermed.line.σ.gp.kernel.variance.val[0]:.3e}")
print(f"A length scale: {opt_model_intermed.line.A.kernel.length_scale.val[0]:.3e}")
print(f"λ0 length scale: {opt_model_intermed.line.λ0.kernel.length_scale.val[0]:.3e}")
print(f"σ length scale: {opt_model_intermed.line.σ.gp.kernel.length_scale.val[0]:.3e}")

# Fix the hyperparameters
opt_model = eqx.tree_at(lambda m: m.line.A.kernel.length_scale.fix, opt_model, True)
opt_model = eqx.tree_at(lambda m: m.line.λ0.kernel.length_scale.fix, opt_model, True)
opt_model = eqx.tree_at(lambda m: m.line.σ.gp.kernel.length_scale.fix, opt_model, True)
opt_model = eqx.tree_at(lambda m: m.line.A.kernel.variance.fix, opt_model, True)
opt_model = eqx.tree_at(lambda m: m.line.λ0.kernel.variance.fix, opt_model, True)
opt_model = eqx.tree_at(lambda m: m.line.σ.gp.kernel.variance.fix, opt_model, True)

# Unfix the \sigma coefficients
# opt_model = eqx.tree_at(lambda m: m.line.σ.gp.coefficients.fix, opt_model, False)

# Second round of optimisation with lbfgs
opt_frame2 = OptimiserFrame(
    model=opt_model,
    loss_fn=neg_ln_posterior,
    optimiser=adam(0.2),
)

# First round of optimisation with lbfgs
opt_model = opt_frame2.run(
    int(5 * n_steps),
    λ=fd.λ - LINE_CENTRE,
    xy_data=fd.αδ_data,
    data=fd.flux,
    u_data=fd.u_flux,
    mask=fd.mask,
)

opt_model_locked = opt_model.get_locked_model()

plt.figure(dpi=150)
plt.plot(opt_frame1.loss_history + opt_frame2.loss_history)
# plt.yscale("log")
# plt.xscale("log")
plt.show()


# ======= Plot some predictions =======

λ_dense = jnp.linspace(fd.λ.min(), fd.λ.max(), 100)
λ_dense_pred = λ_dense - LINE_CENTRE


# Plot some random spectra
pred = jax.vmap(opt_model_locked, in_axes=(0, None))(λ_dense_pred, fd.αδ_data)

i = 456
for i in rng.choice(a=n_spaxels, size=5):
    plt.figure()
    plt.plot(fd.λ, fd._flux[:, i])
    plt.plot(λ_dense, pred[:, i], alpha=0.8)
    plt.show()


A_pred = opt_model_locked.line.A_positive(xy_dense)
λ0_pred = opt_model_locked.line.λ0(xy_dense)
σ_pred = opt_model_locked.line.σ_positive(xy_dense)

fig, ax = plt.subplots(1, 3, figsize=[12, 4], layout="compressed", dpi=100)

ax[0].pcolormesh(x_dense_, y_dense_, A_pred.reshape((xy_shape)))
ax[1].pcolormesh(
    x_dense_,
    y_dense_,
    λ0_pred.reshape((xy_shape)),
    cmap="RdBu",
    vmin=np.median(λ0_pred) - 0.3,
    vmax=np.median(λ0_pred) + 0.3,
)
ax[2].pcolormesh(x_dense_, y_dense_, σ_pred.reshape((xy_shape)), cmap="plasma")

for axi in ax.flatten():
    # axi.scatter(fd.α, fd.δ, marker=".", c="k", alpha=0.01)
    axi.set_aspect(1)
    axi.set_xlim(fd.α.max(), fd.α.min())

plt.show()


plt.figure(dpi=150)
plt.scatter(
    fd.α,
    fd.δ,
    c=opt_model_locked.continuum.const.spaxel_values.val,
    s=2,
    cmap="RdBu",
    vmin=-0.1,
    vmax=0.1,
)
plt.axis("scaled")
plt.show()

# Plot the coefficients of the Fourier basis functions
fig, ax = plt.subplots(1, 3, figsize=[12, 4], layout="compressed")
A_coeffs = opt_model_locked.line.A.coefficients.val.reshape(n_modes).T
λ0_coeffs = opt_model_locked.line.λ0.coefficients.val.reshape(n_modes).T
σ_coeffs = opt_model_locked.line.σ.gp.coefficients.val.reshape(n_modes).T
ax[0].imshow(
    A_coeffs,
    vmin=-A_coeffs.max(),
    vmax=A_coeffs.max(),
    cmap="RdBu",
)
ax[0].set_title(r"$A$ coefficients")
ax[1].imshow(
    λ0_coeffs,
    vmin=-λ0_coeffs.max(),
    vmax=λ0_coeffs.max(),
    cmap="RdBu",
)
ax[1].set_title(r"$\lambda_0$ coefficients")
ax[2].imshow(
    σ_coeffs,
    vmin=-σ_coeffs.max(),
    vmax=σ_coeffs.max(),
    cmap="RdBu",
)
ax[2].set_title(r"$\sigma$ coefficients")
for axi in ax.flatten():
    axi.set_xticks([])
    axi.set_yticks([])
plt.show()

# Compare the coefficients to the initial model by plotting the difference
init_A_coeffs = init_model_locked.line.A.coefficients.val.reshape(n_modes).T
init_λ0_coeffs = init_model_locked.line.λ0.coefficients.val.reshape(n_modes).T
init_σ_coeffs = init_model_locked.line.σ.gp.coefficients.val.reshape(n_modes).T
fig, ax = plt.subplots(1, 3, figsize=[12, 4], layout="compressed")
ax[0].imshow(
    A_coeffs - init_A_coeffs,
    vmin=-A_coeffs.max(),
    vmax=A_coeffs.max(),
    cmap="RdBu",
)
ax[0].set_title(r"$A$ coefficients difference")
ax[1].imshow(
    λ0_coeffs - init_λ0_coeffs,
    vmin=-λ0_coeffs.max(),
    vmax=λ0_coeffs.max(),
    cmap="RdBu",
)
ax[1].set_title(r"$\lambda_0$ coefficients difference")
ax[2].imshow(
    σ_coeffs - init_σ_coeffs,
    vmin=-σ_coeffs.max(),
    vmax=σ_coeffs.max(),
    cmap="RdBu",
)
ax[2].set_title(r"$\sigma$ coefficients difference")
for axi in ax.flatten():
    axi.set_xticks([])
    axi.set_yticks([])
plt.show()


print("Optimised model parameters:")
print(f"A variance: {opt_model_locked.line.A.kernel.variance.val[0]:.3e}")
print(f"λ0 variance: {opt_model_locked.line.λ0.kernel.variance.val[0]:.3e}")
print(f"σ variance: {opt_model_locked.line.σ.gp.kernel.variance.val[0]:.3e}")
print(f"A length scale: {opt_model_locked.line.A.kernel.length_scale.val[0]:.3e}")
print(f"λ0 length scale: {opt_model_locked.line.λ0.kernel.length_scale.val[0]:.3e}")
print(f"σ length scale: {opt_model_locked.line.σ.gp.kernel.length_scale.val[0]:.3e}")
