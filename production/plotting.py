# plotting.py
# CLI for plotting diagnostic plots for the model fits
import argparse
from pathlib import Path
from typing import List, Tuple

import cmasher as cmr
import jax
import matplotlib.pyplot as plt
import mpl_drip.colormaps
import numpy as np
import seaborn as sns
from configs.data import DRP_FILES
from configs.line_data import LINE_CENTRES
from configs.lvm_cmap import vel_cmap
from lvm_tools import FitDataBuilder, LVMTile, LVMTileCollection
from lvm_tools.utils.mask import mask_near_points
from main import EXPERIMENTAL, collect_lines_and_configs, fits_path, get_model_name
from mpl_drip import COLORS
from spectracles import load_model
from spectracles.model.data import SpatialDataGeneric
from spectracles.model.share_module import ShareModule
from spectracles.model.spatial import get_freqs

plt.style.use("mpl_drip.custom")
jax.config.update("jax_enable_x64", True)
rng = np.random.default_rng(0)

plots_path = Path("plots")
if EXPERIMENTAL:
    plots_path = Path("plots_experimental")


# =============================================================================
# USAGE EXAMPLES

# # All plots for all lines, latest models only
# python plotting.py

# # Loss progress for all models of a specific line
# python plotting.py --line Fe_I --model-number all --plot-type loss_progress

# # Specific models across all lines
# python plotting.py --model-number 1,5,10 --plot-type residuals

# # Range of models for interactive viewing
# python plotting.py --line Fe_I --model-number 1-20 --show


# =============================================================================
# PLOT FUNCTIONS - Define these yourself


def plot_loss_curve(model, loss, fd):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(loss)
    ax[1].plot(np.log10(-loss))
    ax[0].set_ylabel("Loss (neg log posterior)")
    ax[1].set_ylabel("Negative Log10 Loss")
    ax[1].set_xlabel("Step")
    # Stop the y-axis limit from being too huge
    upper = min(np.nanmax(loss), 1e9)
    loss_range = upper - np.nanmin(loss)
    ax[0].set_ylim(
        bottom=np.nanmin(loss) - 0.05 * loss_range,
        top=upper + 0.05 * loss_range,
    )
    return fig


def plot_loss_progress(models_dict, losses_dict, fd):
    fig, ax = plt.subplots()
    model_numbers = sorted(models_dict.keys())
    loss = [losses_dict[mn] for mn in model_numbers]
    loss = np.concatenate(loss)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(loss)
    ax[1].plot(np.log10(-loss))
    ax[0].set_ylabel("Loss (neg log posterior)")
    ax[1].set_ylabel("Negative Log10 Loss")
    ax[1].set_xlabel("Step")
    # Stop the y-axis limit from being too huge
    upper = min(np.nanmax(loss), 1e9)
    loss_range = upper - np.nanmin(loss)
    ax[0].set_ylim(
        bottom=np.nanmin(loss) - 0.05 * loss_range,
        top=upper + 0.05 * loss_range,
    )
    return fig


def plot_hyperparameter_progress(models_dict, losses_dict, fd):
    fig, ax = plt.subplots()
    model_numbers = sorted(models_dict.keys())

    def get_hyperparams(model):
        A_len = model.line.A_raw.kernel.length_scale.val
        A_var = model.line.A_raw.kernel.variance.val
        v_len = model.line.v.kernel.length_scale.val
        v_var = model.line.v.kernel.variance.val
        σv_len = model.line.vσ_raw.kernel.length_scale.val
        σv_var = model.line.vσ_raw.kernel.variance.val
        return np.array([A_len, A_var, v_len, v_var, σv_len, σv_var])

    hyperparams = np.array([get_hyperparams(models_dict[mn]) for mn in model_numbers])

    fig, ax = plt.subplots()
    for i, param_name in enumerate(
        [
            r"$l_A$",
            r"$\sigma_A^2$",
            r"$l_v$",
            r"$\sigma_v^2$",
            r"$l_{\sigma v}$",
            r"$\sigma_{\sigma v}^2$",
        ]
    ):
        ax.plot(model_numbers, hyperparams[:, i], label=param_name, alpha=0.6)
    ax.set_xlabel("Model Number")
    ax.set_ylabel("Hyperparameter Value")
    ax.set_title("Hyperparameter Progress")
    # ax.set_yscale("log")
    ax.legend(loc="best")
    return fig


def plot_spectra(model, loss, fd):
    # Get model predictions on dense λ grid
    n_λ_dense = 1000
    λ_dense = np.linspace(fd.λ.min(), fd.λ.max(), n_λ_dense)
    pred_flux = jax.vmap(model, in_axes=(0, None))(λ_dense, fd.αδ_data)
    pred_flux = fd.predict_flux(pred_flux)
    fig, ax = plt.subplots(4, 3, figsize=[14, 9], sharex=True, sharey=True, layout="compressed")
    ax_flat = ax.flatten()
    norm = 1e-12
    # Reset the rng so both this plot and the residuals plot use the same random spaxels
    rng = np.random.default_rng(0)
    for i, j in enumerate(rng.choice(len(model.offs.const.spaxel_values.val), 12, replace=False)):
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
            c=COLORS[0],
        )
        ax_flat[i].plot(λ_dense, pred_flux[:, j] / norm, c=COLORS[1], zorder=0, label="Model")
    ax[0, 0].legend(loc="upper left", frameon=False)
    ax[-1, 0].set_xlabel(r"$\lambda$ [${\rm \AA}$]")
    ax[-1, 0].set_ylabel(r"Scaled flux")
    fig.suptitle("Model predictions for random spaxels")
    return fig


def plot_spectra_residuals(model, loss, fd):
    # Get model predictions on data λ grid
    pred_flux = jax.vmap(model, in_axes=(0, None))(fd.λ, fd.αδ_data)
    pred_flux = fd.predict_flux(pred_flux)
    fig, ax = plt.subplots(4, 3, figsize=[14, 9], sharex=True, sharey=True, layout="compressed")
    ax_flat = ax.flatten()
    norm = 1e-12
    # Reset the rng so both this plot and the spectra comparison plot use the same random spaxels
    rng = np.random.default_rng(0)
    for i, j in enumerate(rng.choice(len(model.offs.const.spaxel_values.val), 12, replace=False)):
        ax_flat[i].errorbar(
            fd.λ,
            (fd.predict_flux(fd._flux[:, j]) - pred_flux[:, j]) / norm,
            yerr=fd.predict_ivar(fd.i_var[:, j]) ** -0.5 / norm,
            zorder=1,
            elinewidth=1,
            lw=0,
            marker=".",
            capsize=1,
            c=COLORS[0],
        )
    ax[-1, 0].set_xlabel(r"$\lambda$ [${\rm \AA}$]")
    ax[-1, 0].set_ylabel(r"Scaled flux")
    fig.suptitle("Residuals for random spaxels")
    return fig


def plot_map_chi2(model, loss, fd):
    # Get model predictions on data λ grid
    pred_flux = jax.vmap(model, in_axes=(0, None))(fd.λ, fd.αδ_data)
    pred_flux = fd.predict_flux(pred_flux)

    # Calculate chi2
    chi2 = ((fd.predict_flux(fd._flux) - pred_flux) ** 2) * fd.predict_ivar(fd.i_var)
    chi2_map = np.nansum(chi2, axis=0)

    # Total number of parameters (This is so poorly motivated for NDOF that I should probably just remove it)
    # n_params = (
    #     model.line.A_raw.coefficients.val.size
    #     + model.line.v.coefficients.val.size
    #     + model.line.vσ_raw.coefficients.val.size
    #     + model.offs.const.spaxel_values.val.size
    #     + 2  # For the barycentric correction and systematic velocity
    # )
    n_dof_per_spaxel = fd.λ.size  # - n_params // fd.α.size

    red_chi2_map = chi2_map / n_dof_per_spaxel

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.scatter(
        fd.predict_α(fd.α),
        fd.predict_δ(fd.δ),
        c=red_chi2_map,
        cmap="viridis",
        s=5,
        edgecolor="none",
    )
    ax.set_title("Reduced Chi2 Map")
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$\delta$ [deg]")
    ax.set_xlim(fd.predict_α(fd.α).max(), fd.predict_α(fd.α).min())
    ax.set_ylim(fd.predict_δ(fd.δ).min(), fd.predict_δ(fd.δ).max())
    ax.set_aspect(1)
    fig.colorbar(cax, ax=ax, label="Value")

    return fig


def plot_map_residuals(model, loss, fd):
    # Get model predictions on data λ grid
    pred_flux = jax.vmap(model, in_axes=(0, None))(fd.λ, fd.αδ_data)
    pred_flux = fd.predict_flux(pred_flux)

    # Calculate residuals
    residuals = fd.predict_flux(fd._flux) - pred_flux
    summed_residuals = np.nansum(residuals, axis=0)

    vmax = np.nanmax(np.abs(summed_residuals))

    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.scatter(
        fd.predict_α(fd.α),
        fd.predict_δ(fd.δ),
        c=summed_residuals,
        cmap="RdBu",
        vmin=-vmax,
        vmax=vmax,
        s=5,
        edgecolor="none",
    )
    ax.set_title("Summed Residuals Map")
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$\delta$ [deg]")
    ax.set_xlim(fd.predict_α(fd.α).max(), fd.predict_α(fd.α).min())
    ax.set_ylim(fd.predict_δ(fd.δ).min(), fd.predict_δ(fd.δ).max())
    ax.set_aspect(1)
    fig.colorbar(cax, ax=ax, label="Value")

    return fig


def plot_spectrum_chi2(model, loss, fd):
    # Get model predictions on data λ grid
    pred_flux = jax.vmap(model, in_axes=(0, None))(fd.λ, fd.αδ_data)
    pred_flux = fd.predict_flux(pred_flux)

    # Calculate chi2
    chi2 = ((fd.predict_flux(fd._flux) - pred_flux) ** 2) * fd.predict_ivar(fd.i_var)
    chi2_spectrum = np.nansum(chi2, axis=1)

    n_dof_per_wavelength = fd.α.size
    red_chi2_spectrum = chi2_spectrum / n_dof_per_wavelength

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fd.λ, red_chi2_spectrum)
    ax.set_title("Reduced Chi2 Spectrum (summed over spaxels)")
    ax.set_xlabel(r"$\lambda$ [${\rm \AA}$]")
    ax.set_ylabel(r"$\chi_{\rm r}^2$")
    return fig


def plot_residuals_spectrum(model, loss, fd):
    # Get model predictions on data λ grid
    pred_flux = jax.vmap(model, in_axes=(0, None))(fd.λ, fd.αδ_data)
    pred_flux = fd.predict_flux(pred_flux)

    # Calculate residuals
    residuals = fd.predict_flux(fd._flux) - pred_flux
    summed_residuals = np.nansum(residuals, axis=1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fd.λ, summed_residuals)
    ax.set_title("Summed Residuals Spectrum (summed over spaxels)")
    ax.set_xlabel(r"$\lambda$ [${\rm \AA}$]")
    ax.set_ylabel(r"Residual Flux")
    return fig


def plot_observed_maps(model, loss, fd):
    α = fd.predict_α(fd.α)
    δ = fd.predict_δ(fd.δ)
    v = model.line.v_obs(fd.αδ_data)
    σ = model.line.σ2_obs(fd.αδ_data) ** 0.5
    c = model.offs.const.spaxel_values.val
    v_vmax = np.nanmax(np.abs(v))
    σ_vmax = np.nanmax(np.abs(σ))
    c_vmax = np.nanmax(np.abs(c))
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True, layout="compressed")
    marker_s = 0.5
    cax1 = ax[0].scatter(α, δ, c=v, cmap="RdBu", s=marker_s)
    cax2 = ax[1].scatter(α, δ, c=σ, cmap="cmr.ocean_r", s=marker_s, vmin=0, vmax=σ_vmax)
    cax3 = ax[2].scatter(α, δ, c=c, cmap="cmr.holly", s=marker_s, vmin=-c_vmax, vmax=c_vmax)
    ax[0].set_title("Centroid Velocity (observer frame)")
    ax[1].set_title("Total Line Width (observer frame)")
    ax[2].set_title("Offset Nuisances")
    ax[0].set_xlabel(r"$\alpha$ [deg]")
    ax[1].set_xlabel(r"$\alpha$ [deg]")
    for a in ax:
        a.set_xlim(α.max(), α.min())
        a.set_ylim(δ.min(), δ.max())
        a.set_aspect(1)
    fig.colorbar(cax1, ax=ax[0], label=r"${\rm [km/s]}$")
    fig.colorbar(cax2, ax=ax[1], label=r"[\AA]")
    fig.colorbar(cax3, ax=ax[2], label=r"[scaled flux]")
    return fig


def plot_vcal(model, loss, fd):
    α = fd.predict_α(fd.α)
    δ = fd.predict_δ(fd.δ)
    v = model.line.v_cal(fd.αδ_data)
    print(f"C_v_cal.val: {model.line.v_cal.C_v_cal.val}")
    # v_vmax = np.nanmax(np.abs(v))
    v_vmax = 25
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True, layout="compressed")
    marker_s = 0.5
    cax1 = ax.scatter(
        α,
        δ,
        c=v,
        cmap="RdBu",
        s=marker_s,
        vmin=0,
        vmax=-7,
    )  # , vmin=-v_vmax, vmax=v_vmax)
    ax.set_title("Calibration Velocity (observer frame)")
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$\delta$ [deg]")
    ax.set_xlim(α.max(), α.min())
    ax.set_ylim(δ.min(), δ.max())
    ax.set_aspect(1)
    fig.colorbar(cax1, ax=ax, label=r"${\rm [km/s]}$")
    return fig


def plot_fcal(model, loss, fd):
    α = fd.predict_α(fd.α)
    δ = fd.predict_δ(fd.δ)
    f_cal = model.line.f_cal(fd.αδ_data)
    f_cal_unconstrained = model.line.f_cal_raw.tile_values.val
    print(f"f_cal_unconstrained: {f_cal_unconstrained}")
    f_cal = 100 * (f_cal / np.nanmedian(f_cal) - 1)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True, layout="compressed")
    marker_s = 0.5
    cax1 = ax.scatter(α, δ, c=f_cal, cmap="PiYG", s=marker_s, vmin=-25, vmax=25)
    ax.set_title("Flux Calibration Factor")
    ax.set_xlabel(r"$\alpha$ [deg]")
    ax.set_ylabel(r"$\delta$ [deg]")
    ax.set_xlim(α.max(), α.min())
    ax.set_ylim(δ.min(), δ.max())
    ax.set_aspect(1)
    fig.colorbar(cax1, ax=ax, label=r"[$\%$ difference]")
    return fig


def plot_maps_singlet(model, loss, fd):
    n_dense = 2000
    dense_shape = (n_dense, n_dense)
    α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
    δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)
    α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)
    αδ_dense = SpatialDataGeneric(
        α_dense.flatten(),
        δ_dense.flatten(),
        idx=np.arange(n_dense * n_dense, dtype=int),
    )
    α_plot = fd.predict_α(α_dense)
    δ_plot = fd.predict_δ(δ_dense)
    mask = mask_near_points(
        α_dense_1D,
        δ_dense_1D,
        fd.α,
        fd.δ,
        threshold=0.1,
    )
    A_pred = np.where(mask, model.line.A(αδ_dense).reshape(dense_shape), np.nan)
    v_pred = np.where(mask, model.line.v(αδ_dense).reshape(dense_shape), np.nan)
    vσ_pred = np.where(mask, model.line.vσ(αδ_dense).reshape(dense_shape), np.nan)
    A_pred = fd.predict_flux(A_pred)

    n_pix_centre = 600
    l_i, u_i = n_dense // 2 - n_pix_centre, n_dense // 2 + n_pix_centre
    A_vmax = np.nanmax(A_pred[l_i:u_i, l_i:u_i])
    # v_median = np.nanmedian(v_pred)
    v_vmax = 12  # np.nanpercentile(np.abs(v_pred), 99)

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
        layout="compressed",
        dpi=500,
    )
    pcolormesh_kwargs = dict(shading="auto", rasterized=True, aa=True)
    c0 = ax[0].pcolormesh(
        α_plot,
        δ_plot,
        A_pred,
        # cmap=vel_cmap(),
        cmap="viridis",
        # cmap=sns.color_palette("rocket", as_cmap=True),
        vmin=0,
        vmax=A_vmax,
        **pcolormesh_kwargs,
    )
    c1 = ax[1].pcolormesh(
        α_plot,
        δ_plot,
        v_pred,
        # cmap="RdBu",
        cmap="red_white_blue_r",
        vmin=-v_vmax,
        vmax=v_vmax,
        **pcolormesh_kwargs,
    )
    c2 = ax[2].pcolormesh(
        α_plot,
        δ_plot,
        np.abs(vσ_pred),
        cmap="cmr.ocean",
        # cmap=sns.color_palette("mako", as_cmap=True),
        vmin=0,
        vmax=np.nanmax(np.abs(vσ_pred)),
        **pcolormesh_kwargs,
    )
    ax[0].set_title("Line flux")
    ax[1].set_title("Radial velocity")
    ax[2].set_title("Velocity dispersion")
    for i, a in enumerate(ax.flatten()):
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
        a.set_aspect(1)
    ax[0].set_xlabel(r"$\alpha$ [deg]")
    ax[0].set_ylabel(r"$\delta$ [deg]")
    colorbar_kwargs = dict(location="bottom", pad=0.02)
    fig.colorbar(c0, ax=ax[0], **colorbar_kwargs, label=r"$10^{-12}$ erg cm$^{-2}$ s$^{-1}$")
    fig.colorbar(c1, ax=ax[1], **colorbar_kwargs, label=r"km s$^{-1}$")
    fig.colorbar(c2, ax=ax[2], **colorbar_kwargs, label=r"km s$^{-1}$")
    return fig


def plot_maps_doublet(model, loss, fd):
    n_dense = 2000
    dense_shape = (n_dense, n_dense)
    α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
    δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)
    α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)
    αδ_dense = SpatialDataGeneric(
        α_dense.flatten(),
        δ_dense.flatten(),
        idx=np.arange(n_dense * n_dense, dtype=int),
    )
    α_plot = fd.predict_α(α_dense)
    δ_plot = fd.predict_δ(δ_dense)
    mask = mask_near_points(
        α_dense_1D,
        δ_dense_1D,
        fd.α,
        fd.δ,
        threshold=0.1,
    )
    A_pred_1 = np.where(mask, model.line.A_1(αδ_dense).reshape(dense_shape), np.nan)
    A_pred_2 = np.where(mask, model.line.A_2(αδ_dense).reshape(dense_shape), np.nan)
    v_pred = np.where(mask, model.line.v(αδ_dense).reshape(dense_shape), np.nan)
    vσ_pred = np.where(mask, model.line.vσ(αδ_dense).reshape(dense_shape), np.nan)
    A_pred_1 = fd.predict_flux(A_pred_1)
    A_pred_2 = fd.predict_flux(A_pred_2)
    A_pred = A_pred_1 + A_pred_2

    n_pix_centre = 600
    l_i, u_i = n_dense // 2 - n_pix_centre, n_dense // 2 + n_pix_centre
    A_vmax = np.nanmax(A_pred[l_i:u_i, l_i:u_i])
    # v_median = np.nanmedian(v_pred)
    v_vmax = 12  # np.nanpercentile(np.abs(v_pred), 99)

    fig, ax = plt.subplots(
        1,
        3,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
        layout="compressed",
        dpi=500,
    )
    pcolormesh_kwargs = dict(shading="auto", rasterized=True, aa=True)
    c0 = ax[0].pcolormesh(
        α_plot,
        δ_plot,
        A_pred,
        # cmap=vel_cmap(),
        cmap="viridis",
        # cmap=sns.color_palette("rocket", as_cmap=True),
        vmin=0,
        vmax=A_vmax,
        **pcolormesh_kwargs,
    )
    c1 = ax[1].pcolormesh(
        α_plot,
        δ_plot,
        v_pred,
        # cmap="RdBu",
        cmap="red_white_blue_r",
        vmin=-v_vmax,
        vmax=v_vmax,
        **pcolormesh_kwargs,
    )
    c2 = ax[2].pcolormesh(
        α_plot,
        δ_plot,
        np.abs(vσ_pred),
        cmap="cmr.ocean",
        # cmap=sns.color_palette("mako", as_cmap=True),
        vmin=0,
        vmax=np.nanmax(np.abs(vσ_pred)),
        **pcolormesh_kwargs,
    )
    ax[0].set_title("Line flux")
    ax[1].set_title("Radial velocity")
    ax[2].set_title("Velocity dispersion")
    for i, a in enumerate(ax.flatten()):
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
        a.set_aspect(1)
    ax[0].set_xlabel(r"$\alpha$ [deg]")
    ax[0].set_ylabel(r"$\delta$ [deg]")
    colorbar_kwargs = dict(location="bottom", pad=0.02)
    fig.colorbar(c0, ax=ax[0], **colorbar_kwargs, label=r"$10^{-12}$ erg cm$^{-2}$ s$^{-1}$")
    fig.colorbar(c1, ax=ax[1], **colorbar_kwargs, label=r"km s$^{-1}$")
    fig.colorbar(c2, ax=ax[2], **colorbar_kwargs, label=r"km s$^{-1}$")
    return fig


def plot_maps(model, loss, fd):
    try:
        _ = model.line.A_raw
        return plot_maps_singlet(model, loss, fd)
    except AttributeError:
        return plot_maps_doublet(model, loss, fd)


# def plot_maps(model, loss, fd):
#     n_dense = 2000
#     dense_shape = (n_dense, n_dense)
#     α_dense_1D = np.linspace(fd.α.min(), fd.α.max(), n_dense)
#     δ_dense_1D = np.linspace(fd.δ.min(), fd.δ.max(), n_dense)
#     α_dense, δ_dense = np.meshgrid(α_dense_1D, δ_dense_1D)
#     αδ_dense = SpatialDataGeneric(
#         α_dense.flatten(),
#         δ_dense.flatten(),
#         idx=np.arange(n_dense * n_dense, dtype=int),
#     )
#     mask = mask_near_points(
#         α_dense_1D,
#         δ_dense_1D,
#         fd.α,
#         fd.δ,
#         threshold=0.1,
#     )

#     A_pred = np.where(mask, model.line.A(αδ_dense).reshape(dense_shape), np.nan)
#     v_pred = np.where(mask, model.line.v(αδ_dense).reshape(dense_shape), np.nan)
#     vσ_pred = np.where(mask, model.line.vσ(αδ_dense).reshape(dense_shape), np.nan)
#     A_pred = fd.predict_flux(A_pred)

#     n_pix_centre = 300
#     l_i, u_i = n_dense // 2 - n_pix_centre, n_dense // 2 + n_pix_centre
#     A_vmax = 1.1 * np.nanmax(A_pred[l_i:u_i, l_i:u_i])
#     A_vmax = 1.5e-12
#     v_median = np.nanmedian(v_pred)
#     v_vmax = np.nanpercentile(np.abs(v_pred), 98)

#     fig, ax = plt.subplots(
#         1,
#         3,
#         figsize=(14, 6),
#         sharex=True,
#         sharey=True,
#         layout="compressed",
#         dpi=1000,
#     )

#     extent = [α_dense_1D.min(), α_dense_1D.max(), δ_dense_1D.min(), δ_dense_1D.max()]
#     imshow_kwargs = dict(origin="lower", extent=extent, aspect="auto", interpolation="bilinear")

#     c0 = ax[0].imshow(A_pred, cmap="viridis", vmin=0, vmax=A_vmax, **imshow_kwargs)
#     c1 = ax[1].imshow(
#         v_pred,
#         cmap="RdBu",
#         vmin=v_median - v_vmax,
#         vmax=v_median + v_vmax,
#         **imshow_kwargs,
#     )
#     c2 = ax[2].imshow(
#         np.abs(vσ_pred),
#         cmap="cmr.ocean_r",
#         vmin=0,
#         vmax=np.nanmax(np.abs(vσ_pred)),
#         **imshow_kwargs,
#     )

#     ax[0].set_title("Line flux")
#     ax[1].set_title("Radial velocity")
#     ax[2].set_title("Velocity dispersion")

#     for i, a in enumerate(ax.flatten()):
#         a.set_xlim(extent[1], extent[0])  # Reverse α axis if desired
#         a.set_axisbelow(False)
#         if i != 0:
#             a.set_xticklabels([])
#             a.set_yticklabels([])
#         a.xaxis.set_major_locator(plt.MultipleLocator(0.5))
#         a.yaxis.set_major_locator(plt.MultipleLocator(0.5))
#         a.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
#         a.yaxis.set_minor_locator(plt.MultipleLocator(0.25))
#         a.set_aspect(1)

#     ax[0].set_xlabel(r"$\alpha$ [deg]")
#     ax[0].set_ylabel(r"$\delta$ [deg]")

#     colorbar_kwargs = dict(location="bottom", pad=0.02)
#     fig.colorbar(c0, ax=ax[0], **colorbar_kwargs, label=r"$10^{-12}$ erg cm$^{-2}$ s$^{-1}$")
#     fig.colorbar(c1, ax=ax[1], **colorbar_kwargs, label=r"km s$^{-1}$")
#     fig.colorbar(c2, ax=ax[2], **colorbar_kwargs, label=r"km s$^{-1}$")

#     return fig


def plot_fourier_coefficients(model, loss, fd):
    """2D imshow plots of the Fourier coefficients for each component."""
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(10, 8),
        sharex=True,
        sharey=True,
        layout="compressed",
    )
    n_modes = model.line.A_raw.n_modes
    fx, fy = get_freqs(n_modes)
    extent = (fx.min(), fx.max(), fy.min(), fy.max())
    imshow_kwargs = dict(
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="cmr.viola",
        vmin=-3,
        vmax=3,
    )
    # Plot A coefficients
    try:
        c = ax[0].imshow(model.line.A_raw.coefficients.val, **imshow_kwargs)
    except AttributeError:
        c = ax[0].imshow(model.line.A_raw_1.coefficients.val, **imshow_kwargs)
    ax[0].set_title(r"$A$ Coefficients")
    # Plot v coefficients
    ax[1].imshow(model.line.v.coefficients.val, **imshow_kwargs)
    ax[1].set_title(r"$v$ Coefficients")
    # Plot vσ coefficients
    ax[2].imshow(model.line.vσ_raw.coefficients.val, **imshow_kwargs)
    ax[2].set_title(r"$v\sigma$ Coefficients")
    ax[0].set_xlabel(r"$\omega_x$")
    ax[0].set_ylabel(r"$\omega_y$")
    for a in ax:
        a.set_aspect(1)
    fig.colorbar(c, ax=ax[:], orientation="vertical", pad=0.02)
    return fig


# Map plot types to (type, function) pairs
# 'single' plots get called once per model
# 'multi' plots get called once per line with all models for that line
PLOT_FUNCTIONS = {
    "loss": ("single", plot_loss_curve),
    "loss_progress": ("multi", plot_loss_progress),
    "hyperparams": ("multi", plot_hyperparameter_progress),
    "spectra": ("single", plot_spectra),
    "residuals": ("single", plot_spectra_residuals),
    "chi2_map": ("single", plot_map_chi2),
    "residuals_map": ("single", plot_map_residuals),
    "chi2_spectrum": ("single", plot_spectrum_chi2),
    "residuals_spectrum": ("single", plot_residuals_spectrum),
    "observed_maps": ("single", plot_observed_maps),
    "maps": ("single", plot_maps),
    "fourier": ("single", plot_fourier_coefficients),
    "vcal": ("single", plot_vcal),
    "fcal": ("single", plot_fcal),
}

ALL_PLOT_TYPES = list(PLOT_FUNCTIONS.keys())


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def load_model_and_loss(line: str, model_number: int):
    """Load model and loss data."""
    model_name = get_model_name(line, model_number)
    model_path = fits_path / line / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} does not exist.")

    model: ShareModule = load_model(model_path)
    locked_model = model.get_locked_model()

    loss_path = model_path.with_name(model_path.name + ".loss.npy")
    if not loss_path.exists():
        raise FileNotFoundError(f"Loss file {loss_path} does not exist.")
    # Load loss data from the file
    loss = np.load(loss_path, allow_pickle=True)

    return locked_model, loss


def get_available_lines() -> List[str]:
    """Get available lines from fits directory."""
    if not fits_path.exists():
        return []
    return [d.name for d in fits_path.iterdir() if d.is_dir()]


def get_available_model_numbers(line: str) -> List[int]:
    """Get all available model numbers for a line."""
    line_path = fits_path / line
    model_files = list(line_path.glob("*.model"))
    model_numbers = [int(f.name.split(".")[1]) for f in model_files if f.is_file()]
    return model_numbers


def get_latest_model_number(line: str) -> int:
    """Get latest model number for a line."""
    available = get_available_model_numbers(line)
    if not available:
        raise ValueError(f"No models found for line {line}")
    return max(available)


def parse_model_numbers(model_arg: str) -> List[int]:
    """Parse model number argument into list of integers."""
    if model_arg == "all":
        return "all"  # Special case handled later

    model_numbers = []
    parts = model_arg.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like "1-5"
            start, end = part.split("-", 1)
            model_numbers.extend(range(int(start), int(end) + 1))
        else:
            # Handle single number
            model_numbers.append(int(part))

    return sorted(list(set(model_numbers)))  # Remove duplicates and sort


def get_model_numbers_for_line(line: str, requested_models) -> Tuple[List[int], List[int]]:
    """Get available model numbers for a line, with warnings for missing ones."""
    available = get_available_model_numbers(line)

    if requested_models == "all":
        return available, []

    # Check which requested models are available
    found = []
    missing = []

    for model_num in requested_models:
        if model_num in available:
            found.append(model_num)
        else:
            missing.append(model_num)

    return found, missing


def create_plots(
    lines: List[str],
    model_arg: str,
    plot_types: List[str],
    show: bool = False,
    skip_save: bool = False,
):
    """Create all requested plots."""

    print("Reading data...", end=" ", flush=True)
    tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])
    print("Done.")

    print("Collecting lines and configurations...", end=" ")
    configs = collect_lines_and_configs()
    print("Done.")

    # Parse model numbers
    if model_arg is None:
        # Default to latest for each line
        requested_models = None
    else:
        requested_models = parse_model_numbers(model_arg)

    for line in lines:
        print(f"Processing {line}")

        builder = FitDataBuilder(tiles, configs[line])
        fd = builder.build()

        # Get model numbers for this line
        if requested_models is None:
            # Use latest model only
            try:
                line_models = [get_latest_model_number(line)]
                missing = []
            except Exception as e:
                print(f"Could not get latest model for {line}: {e}")
                continue
        else:
            line_models, missing = get_model_numbers_for_line(line, requested_models)

            if missing:
                print(f"  Warning: Models {missing} not found for {line}")

            if not line_models:
                print(f"  Skipping {line}: no requested models available")
                continue

        # Load all models and losses for this line
        models_dict = {}
        losses_dict = {}

        for model_number in line_models:
            try:
                model, loss = load_model_and_loss(line, model_number)
                models_dict[model_number] = model
                losses_dict[model_number] = loss
            except FileNotFoundError as e:
                print(f"  Warning: Could not load model {model_number}: {e}")

        if not models_dict:
            print(f"  Skipping {line}: no models could be loaded")
            continue

        # Create plots
        for plot_type in plot_types:
            if plot_type not in PLOT_FUNCTIONS:
                print(f"Unknown plot type: {plot_type}")
                continue

            plot_category, plot_func = PLOT_FUNCTIONS[plot_type]

            if plot_category == "single":
                # Create one plot per model
                for model_number in sorted(models_dict.keys()):
                    print(f"  Creating {plot_type} plot for model {model_number}...")

                    model = models_dict[model_number]
                    loss = losses_dict[model_number]

                    try:
                        fig = plot_func(model, loss, fd)
                        # fig.suptitle(f"{line} - Model {model_number} - {plot_type}")
                    except Exception as e:
                        print(f"  Error creating {plot_type} plot for model {model_number}: {e}")
                        continue

                    # Save if not skipping
                    if not skip_save:
                        output_dir = plots_path / line
                        output_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{line}_model{model_number:04d}_{plot_type}.pdf"
                        filepath = output_dir / filename
                        fig.savefig(filepath, bbox_inches="tight")
                        print(f"    Saved to {filepath}")

                    # Show if requested
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)

            elif plot_category == "multi":
                # Create one plot for all models
                if len(models_dict) < 2:
                    print(f"  Skipping {plot_type} plot: need at least 2 models")
                    continue

                print(f"  Creating {plot_type} plot for models {sorted(models_dict.keys())}...")

                try:
                    fig = plot_func(models_dict, losses_dict, fd)
                except Exception as e:
                    print(
                        f"  Error creating {plot_type} plot for models {sorted(models_dict.keys())}: {e}"
                    )
                    continue
                model_range = f"{min(models_dict.keys())}-{max(models_dict.keys())}"
                # fig.suptitle(f"{line} - Models {model_range} - {plot_type}")

                # Save if not skipping
                if not skip_save:
                    output_dir = plots_path / line
                    output_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{line}_models{model_range}_{plot_type}.pdf"
                    filepath = output_dir / filename
                    fig.savefig(filepath, bbox_inches="tight")
                    print(f"    Saved to {filepath}")

                # Show if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create diagnostic plots for model fits")

    parser.add_argument("--line", type=str, help='Line to plot (or "all" for all lines)')
    parser.add_argument(
        "--model-number",
        type=str,
        help='Model number(s) to plot: single (5), range (1-10), list (1,3,5), or "all"',
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=ALL_PLOT_TYPES + ["all"],
        default="all",
        help="Type of plot to create",
    )
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--skip-save", action="store_true", help="Skip saving plots to files")

    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which lines to process
    if args.line == "all" or args.line is None:
        lines = get_available_lines()
        if not lines:
            print("No lines found in fits directory")
            return
    else:
        lines = [args.line]

    # Limit lines only to those that exist
    lines = [line for line in lines if line in LINE_CENTRES.keys()]

    # Determine plot types to create
    if args.plot_type == "all":
        plot_types = ALL_PLOT_TYPES
    else:
        plot_types = [args.plot_type]

    print(f"Creating plots for lines: {lines}")
    if args.model_number:
        print(f"Model numbers: {args.model_number}")
    else:
        print("Model numbers: latest for each line")
    print(f"Plot types: {plot_types}")

    create_plots(lines, args.model_number, plot_types, args.show, args.skip_save)


if __name__ == "__main__":
    main()
