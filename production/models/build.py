from pathlib import Path

import numpy as np
from lvm_tools.fit_data.fit_data import FitData
from spectracles import ConstrainedParameter, Kernel, Matern32, Parameter, build_model
from spectracles.model.share_module import ShareModule

from .doublet_model import LVMModelDoublet
from .fluxcal_model import LVMModelFluxCal
from .model import LVMModel
from .production_model import LVMModelProduction
from .wavecal_model import LVMModelWaveCal

DEFAULT_L_BOUNDS = (1e-2, 1e1)
DEFAULT_VAR_BOUNDS = (1e-5, 1e1)

DEFAULT_BOUNDS = {
    "A_l": DEFAULT_L_BOUNDS,
    "v_l": DEFAULT_L_BOUNDS,
    "σ_l": DEFAULT_L_BOUNDS,
    "A_var": DEFAULT_VAR_BOUNDS,
    "v_var": DEFAULT_VAR_BOUNDS,
    "vσ_var": DEFAULT_VAR_BOUNDS,
}
DEFAULT_INIT = {
    "A_l": 0.1,
    "v_l": 0.1,
    "σ_l": 0.1,
    "A_var": 1.0,
    "v_var": 1.0,
    "vσ_var": 9.0,
}

WAVECAL_FILE = Path(
    "/Users/tomhilder/Documents/PhD/research/lvm_project/lvm_spectral_spatial/paper_1/production/calibration_results/Hβλ4861_C_v_cal.npy"
).resolve()
FLUXCAL_FILE = Path(
    "/Users/tomhilder/Documents/PhD/research/lvm_project/lvm_spectral_spatial/paper_1/production/calibration_results/Hαλ6563_fcal_raw.npy"
).resolve()

BLUE_END = 5800


def get_λ0_ind(fd, μ):
    return np.argmin(np.abs(fd.λ - μ))


def model_builder(
    model: ShareModule,
    line_centre: float,
    fd: FitData,
    n_modes: tuple,
    bounds: dict,
    init: dict,
    kernel: Kernel = Matern32,
):
    if model == LVMModel:
        return build_LVMModel(model, line_centre, fd, n_modes, bounds, init, kernel)
    elif model == LVMModelWaveCal:
        return build_LVMModelWaveCal(model, line_centre, fd, n_modes, bounds, init, kernel)
    elif model == LVMModelFluxCal:
        return build_LVMModelFluxCal(model, line_centre, fd, n_modes, bounds, init, kernel)
    elif model == LVMModelProduction:
        return build_LVMModelProduction(model, line_centre, fd, n_modes, bounds, init, kernel)
    elif model == LVMModelDoublet:
        return build_LVMModelDoublet(model, line_centre, fd, n_modes, bounds, init, kernel)
    else:
        raise ValueError(f"Unknown model type: {model}")


def build_LVMModel(
    model: ShareModule,
    line_centre: float,
    fd: FitData,
    n_modes: tuple,
    bounds: dict,
    init: dict,
    kernel: Kernel = Matern32,
):
    n_spaxels = len(fd.α)

    # === Parameters ===

    # Line centre
    μ_line = Parameter(initial=line_centre, fixed=True)

    # Kernel parameters
    A_length_scale = ConstrainedParameter(
        initial=init["A_l"],
        lower=bounds["A_l"][0],
        upper=bounds["A_l"][1],
        fixed=True,
    )
    v_length_scale = ConstrainedParameter(
        initial=init["v_l"],
        lower=bounds["v_l"][0],
        upper=bounds["v_l"][1],
        fixed=True,
    )
    σ_length_scale = ConstrainedParameter(
        initial=init["σ_l"],
        lower=bounds["σ_l"][0],
        upper=bounds["σ_l"][1],
        fixed=True,
    )
    A_variance = ConstrainedParameter(
        initial=init["A_var"],
        lower=bounds["A_var"][0],
        upper=bounds["A_var"][1],
        fixed=True,
    )
    v_variance = ConstrainedParameter(
        initial=init["v_var"],
        lower=bounds["v_var"][0],
        upper=bounds["v_var"][1],
        fixed=True,
    )
    σ_variance = ConstrainedParameter(
        initial=init["vσ_var"],
        lower=bounds["vσ_var"][0],
        upper=bounds["vσ_var"][1],
        fixed=True,
    )

    # Nuisances
    offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels), fixed=True)

    # Measured values
    v_bary = Parameter(initial=fd.v_bary, fixed=True)
    initial_σ_lsf = 1.0 * fd.lsf_σ[fd.lsf_σ.shape[0] // 2, :]
    σ_lsf = Parameter(initial=initial_σ_lsf, fixed=True)

    # Systematics
    v_syst = Parameter(initial=0.0, fixed=True)  # Systematic velocity

    # === Kernels ===

    A_kernel = kernel(length_scale=A_length_scale, variance=A_variance)
    v_kernel = kernel(length_scale=v_length_scale, variance=v_variance)
    σ_kernel = kernel(length_scale=σ_length_scale, variance=σ_variance)

    # Build the model
    my_model = build_model(
        model,
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

    # Save a copy of the initial model
    init_model = my_model.copy()

    return my_model, init_model


def build_LVMModelWaveCal(
    model: ShareModule,
    line_centre: float,
    fd: FitData,
    n_modes: tuple,
    bounds: dict,
    init: dict,
    kernel: Kernel = Matern32,
):
    n_spaxels = len(fd.α)

    # Line centre
    μ_line = Parameter(initial=line_centre, fixed=True)

    # Kernel parameters
    A_length_scale = ConstrainedParameter(
        initial=init["A_l"],
        lower=bounds["A_l"][0],
        upper=bounds["A_l"][1],
        fixed=True,
    )
    v_length_scale = ConstrainedParameter(
        initial=init["v_l"],
        lower=bounds["v_l"][0],
        upper=bounds["v_l"][1],
        fixed=True,
    )
    σ_length_scale = ConstrainedParameter(
        initial=init["σ_l"],
        lower=bounds["σ_l"][0],
        upper=bounds["σ_l"][1],
        fixed=True,
    )
    A_variance = ConstrainedParameter(
        initial=init["A_var"],
        lower=bounds["A_var"][0],
        upper=bounds["A_var"][1],
        fixed=True,
    )
    v_variance = ConstrainedParameter(
        initial=init["v_var"],
        lower=bounds["v_var"][0],
        upper=bounds["v_var"][1],
        fixed=True,
    )
    σ_variance = ConstrainedParameter(
        initial=init["vσ_var"],
        lower=bounds["vσ_var"][0],
        upper=bounds["vσ_var"][1],
        fixed=True,
    )

    # Nuisances
    offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels), fixed=True)

    # Measured values
    v_bary = Parameter(initial=fd.v_bary, fixed=True)
    initial_σ_lsf = 1.0 * fd.lsf_σ[fd.lsf_σ.shape[0] // 2, :]
    σ_lsf = Parameter(initial=initial_σ_lsf, fixed=True)

    # Systematics
    v_syst = Parameter(initial=25.0, fixed=True)  # Systematic velocity
    C_v_cal = Parameter(
        initial=np.zeros((2)),
        fixed=True,
    )  # Per-IFU wavelength calibration offsets

    # === Kernels ===

    A_kernel = kernel(length_scale=A_length_scale, variance=A_variance)
    v_kernel = kernel(length_scale=v_length_scale, variance=v_variance)
    σ_kernel = kernel(length_scale=σ_length_scale, variance=σ_variance)

    # Build the model
    my_model = build_model(
        model,
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
        C_v_cal=C_v_cal,
    )

    # Save a copy of the initial model
    init_model = my_model.copy()

    return my_model, init_model


def build_LVMModelFluxCal(
    model: ShareModule,
    line_centre: float,
    fd: FitData,
    n_modes: tuple,
    bounds: dict,
    init: dict,
    kernel: Kernel = Matern32,
):
    n_spaxels = len(fd.α)
    n_tiles = len(np.unique(fd.tile_idx))
    raise Exception(
        "I fixed the n_tiles here (previously forgot the unique) but I haven't had a change to test it. It probably works."
    )

    # Line centre
    μ_line = Parameter(initial=line_centre, fixed=True)

    # Kernel parameters
    A_length_scale = ConstrainedParameter(
        initial=init["A_l"],
        lower=bounds["A_l"][0],
        upper=bounds["A_l"][1],
        fixed=True,
    )
    v_length_scale = ConstrainedParameter(
        initial=init["v_l"],
        lower=bounds["v_l"][0],
        upper=bounds["v_l"][1],
        fixed=True,
    )
    σ_length_scale = ConstrainedParameter(
        initial=init["σ_l"],
        lower=bounds["σ_l"][0],
        upper=bounds["σ_l"][1],
        fixed=True,
    )
    A_variance = ConstrainedParameter(
        initial=init["A_var"],
        lower=bounds["A_var"][0],
        upper=bounds["A_var"][1],
        fixed=True,
    )
    v_variance = ConstrainedParameter(
        initial=init["v_var"],
        lower=bounds["v_var"][0],
        upper=bounds["v_var"][1],
        fixed=True,
    )
    σ_variance = ConstrainedParameter(
        initial=init["vσ_var"],
        lower=bounds["vσ_var"][0],
        upper=bounds["vσ_var"][1],
        fixed=True,
    )

    # Nuisances
    offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels), fixed=True)

    # Measured values
    v_bary = Parameter(initial=fd.v_bary, fixed=True)
    initial_σ_lsf = 1.0 * fd.lsf_σ[fd.lsf_σ.shape[0] // 2, :]
    σ_lsf = Parameter(initial=initial_σ_lsf, fixed=True)

    # Systematics
    v_syst = Parameter(initial=25.0, fixed=True)  # Systematic velocity
    f_cal_unconstrained = Parameter(
        initial=np.zeros((n_tiles,)),
        fixed=True,
    )  # Per-IFU flux calibration multipliers

    # === Kernels ===

    A_kernel = kernel(length_scale=A_length_scale, variance=A_variance)
    v_kernel = kernel(length_scale=v_length_scale, variance=v_variance)
    σ_kernel = kernel(length_scale=σ_length_scale, variance=σ_variance)

    # Build the model
    my_model = build_model(
        model,
        n_tiles=n_tiles,
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
        f_cal_unconstrained=f_cal_unconstrained,
    )

    # Save a copy of the initial model
    init_model = my_model.copy()

    return my_model, init_model


def build_LVMModelProduction(
    model: ShareModule,
    line_centre: float,
    fd: FitData,
    n_modes: tuple,
    bounds: dict,
    init: dict,
    kernel: Kernel = Matern32,
):
    n_spaxels = len(fd.α)
    n_tiles = len(np.unique(fd.tile_idx))

    # Line centre
    μ_line = Parameter(initial=line_centre, fixed=True)

    # Kernel parameters
    A_length_scale = ConstrainedParameter(
        initial=init["A_l"],
        lower=bounds["A_l"][0],
        upper=bounds["A_l"][1],
        fixed=True,
    )
    v_length_scale = ConstrainedParameter(
        initial=init["v_l"],
        lower=bounds["v_l"][0],
        upper=bounds["v_l"][1],
        fixed=True,
    )
    σ_length_scale = ConstrainedParameter(
        initial=init["σ_l"],
        lower=bounds["σ_l"][0],
        upper=bounds["σ_l"][1],
        fixed=True,
    )
    A_variance = ConstrainedParameter(
        initial=init["A_var"],
        lower=bounds["A_var"][0],
        upper=bounds["A_var"][1],
        fixed=True,
    )
    v_variance = ConstrainedParameter(
        initial=init["v_var"],
        lower=bounds["v_var"][0],
        upper=bounds["v_var"][1],
        fixed=True,
    )
    σ_variance = ConstrainedParameter(
        initial=init["vσ_var"],
        lower=bounds["vσ_var"][0],
        upper=bounds["vσ_var"][1],
        fixed=True,
    )

    # Nuisances
    offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels), fixed=True)

    # Measured values
    v_bary = Parameter(initial=fd.v_bary, fixed=True)
    initial_σ_lsf = 1.0 * fd.lsf_σ[fd.lsf_σ.shape[0] // 2, :]
    σ_lsf = Parameter(initial=initial_σ_lsf, fixed=True)

    # Systematics (saved from previous calibration)
    if line_centre < BLUE_END:
        saved_C_v_cal = np.load(WAVECAL_FILE)
    else:
        saved_C_v_cal = np.zeros((2))
    saved_f_cal_raw = np.load(FLUXCAL_FILE)
    v_syst = Parameter(initial=25.0, fixed=True)  # Systematic velocity
    C_v_cal = Parameter(
        initial=saved_C_v_cal,
        fixed=True,
    )  # Per-IFU wavelength calibration offsets
    f_cal_unconstrained = Parameter(
        initial=saved_f_cal_raw,
        fixed=True,
    )  # Per-IFU flux calibration multipliers

    # === Kernels ===

    A_kernel = kernel(length_scale=A_length_scale, variance=A_variance)
    v_kernel = kernel(length_scale=v_length_scale, variance=v_variance)
    σ_kernel = kernel(length_scale=σ_length_scale, variance=σ_variance)

    # Build the model
    my_model = build_model(
        model,
        n_tiles=n_tiles,
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
        C_v_cal=C_v_cal,
        f_cal_unconstrained=f_cal_unconstrained,
    )

    # Save a copy of the initial model
    init_model = my_model.copy()

    return my_model, init_model


def build_LVMModelDoublet(
    model: ShareModule,
    line_centre: tuple[float, float],
    fd: FitData,
    n_modes: tuple,
    bounds: dict,
    init: dict,
    kernel: Kernel = Matern32,
):
    n_spaxels = len(fd.α)
    n_tiles = len(np.unique(fd.tile_idx))

    # Line centre
    μ_line_1 = Parameter(initial=line_centre[0], fixed=True)
    μ_line_2 = Parameter(initial=line_centre[1], fixed=True)

    # Kernel parameters
    A_length_scale_1 = ConstrainedParameter(
        initial=init["A_l"],
        lower=bounds["A_l"][0],
        upper=bounds["A_l"][1],
        fixed=True,
    )
    A_length_scale_2 = ConstrainedParameter(
        initial=init["A_l"],
        lower=bounds["A_l"][0],
        upper=bounds["A_l"][1],
        fixed=True,
    )
    v_length_scale = ConstrainedParameter(
        initial=init["v_l"],
        lower=bounds["v_l"][0],
        upper=bounds["v_l"][1],
        fixed=True,
    )
    σ_length_scale = ConstrainedParameter(
        initial=init["σ_l"],
        lower=bounds["σ_l"][0],
        upper=bounds["σ_l"][1],
        fixed=True,
    )
    A_variance_1 = ConstrainedParameter(
        initial=init["A_var"],
        lower=bounds["A_var"][0],
        upper=bounds["A_var"][1],
        fixed=True,
    )
    A_variance_2 = ConstrainedParameter(
        initial=init["A_var"],
        lower=bounds["A_var"][0],
        upper=bounds["A_var"][1],
        fixed=True,
    )
    v_variance = ConstrainedParameter(
        initial=init["v_var"],
        lower=bounds["v_var"][0],
        upper=bounds["v_var"][1],
        fixed=True,
    )
    σ_variance = ConstrainedParameter(
        initial=init["vσ_var"],
        lower=bounds["vσ_var"][0],
        upper=bounds["vσ_var"][1],
        fixed=True,
    )

    # Nuisances
    offsets = Parameter(dims=(n_spaxels,), initial=0.0 * np.ones(n_spaxels), fixed=True)

    # Measured values
    v_bary = Parameter(initial=fd.v_bary, fixed=True)

    i_λ0_1 = get_λ0_ind(fd, line_centre[0])
    i_λ0_2 = get_λ0_ind(fd, line_centre[1])
    σ_lsf_1 = Parameter(initial=fd.lsf_σ[i_λ0_1, :], fixed=True)
    σ_lsf_2 = Parameter(initial=fd.lsf_σ[i_λ0_2, :], fixed=True)

    # Systematics (saved from previous calibration)
    if line_centre[1] < BLUE_END:
        saved_C_v_cal = np.load(WAVECAL_FILE)
    else:
        saved_C_v_cal = np.zeros((2))
    saved_f_cal_raw = np.load(FLUXCAL_FILE)
    v_syst = Parameter(initial=25.0, fixed=True)  # Systematic velocity
    C_v_cal = Parameter(
        initial=saved_C_v_cal,
        fixed=True,
    )  # Per-IFU wavelength calibration offsets
    f_cal_unconstrained = Parameter(
        initial=saved_f_cal_raw,
        fixed=True,
    )  # Per-IFU flux calibration multipliers

    # === Kernels ===

    A_kernel_1 = kernel(length_scale=A_length_scale_1, variance=A_variance_1)
    A_kernel_2 = kernel(length_scale=A_length_scale_2, variance=A_variance_2)
    v_kernel = kernel(length_scale=v_length_scale, variance=v_variance)
    σ_kernel = kernel(length_scale=σ_length_scale, variance=σ_variance)

    # Build the model
    my_model = build_model(
        model,
        n_tiles=n_tiles,
        n_spaxels=n_spaxels,
        offsets=offsets,
        line_centre_1=μ_line_1,
        line_centre_2=μ_line_2,
        n_modes=n_modes,
        A_kernel_1=A_kernel_1,
        A_kernel_2=A_kernel_2,
        v_kernel=v_kernel,
        σ_kernel=σ_kernel,
        σ_lsf_1=σ_lsf_1,
        σ_lsf_2=σ_lsf_2,
        v_bary=v_bary,
        v_syst=v_syst,
        C_v_cal=C_v_cal,
        f_cal_unconstrained=f_cal_unconstrained,
    )

    # Save a copy of the initial model
    init_model = my_model.copy()

    return my_model, init_model
