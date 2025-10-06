import jax
import jax.numpy as jnp
from numpy.random import default_rng
from optax import adam, lbfgs
from spectracles.optimise.opt_schedule import PhaseConfig

N_STEPS_DEFAULT = 2000
ΔLOSS_DEFAULT = None
OPT_DEFAULT = adam(1e-2)

rng = default_rng(0)
jax.config.update("jax_enable_x64", True)

# All model parameter path aliases for conciseness
paths = {
    "A_X": "line.A_raw.coefficients",
    "A_l": "line.A_raw.kernel.length_scale",
    "A_var": "line.A_raw.kernel.variance",
    "v_X": "line.v.coefficients",
    "v_l": "line.v.kernel.length_scale",
    "v_var": "line.v.kernel.variance",
    "vσ_X": "line.vσ_raw.coefficients",
    "vσ_l": "line.vσ_raw.kernel.length_scale",
    "vσ_var": "line.vσ_raw.kernel.variance",
    "v_syst": "line.v_syst",
    "v_cal": "line.v_cal.C_v_cal",
    "offs": "offs.const.spaxel_values",
    "f_cal": "line.f_cal_raw.tile_values",
}

paths_doublets = {
    "A_X_1": "line.A_raw_1.coefficients",
    "A_X_2": "line.A_raw_2.coefficients",
    "A_l_1": "line.A_raw_1.kernel.length_scale",
    "A_l_2": "line.A_raw_2.kernel.length_scale",
    "A_var_1": "line.A_raw_1.kernel.variance",
    "A_var_2": "line.A_raw_2.kernel.variance",
    "v_X": "line.v.coefficients",
    "v_l": "line.v.kernel.length_scale",
    "v_var": "line.v.kernel.variance",
    "vσ_X": "line.vσ_raw.coefficients",
    "vσ_l": "line.vσ_raw.kernel.length_scale",
    "vσ_var": "line.vσ_raw.kernel.variance",
    "v_syst": "line.v_syst",
    "v_cal": "line.v_cal.C_v_cal",
    "offs": "offs.const.spaxel_values",
    "f_cal": "line.f_cal_raw.tile_values",
}


def get_dict_of_fixed_status(path_aliases_for_varying, v_cal=False, f_cal=False, paths=paths):
    aliases = paths.keys()
    params_that_dont_exist = []
    if not v_cal:
        params_that_dont_exist.append("v_cal")
    if not f_cal:
        params_that_dont_exist.append("f_cal")
    aliases = [alias for alias in aliases if alias not in params_that_dont_exist]
    return {paths[alias]: (alias not in path_aliases_for_varying) for alias in aliases}


phase_config_defaults = {
    "n_steps": N_STEPS_DEFAULT,
    "optimiser": OPT_DEFAULT,
    "Δloss_criterion": ΔLOSS_DEFAULT,
}


def get_coeffs(
    param="A_X",
    v_syst=False,
    v_cal=False,
    offs=False,
    f_cal=False,
    n_modes=None,
    init=False,
    doublet=False,
):
    if not doublet:
        paths_used = paths
    else:
        paths_used = paths_doublets
    if init:
        val_updates = {paths_used[param]: jnp.array(rng.standard_normal(n_modes))}
    else:
        val_updates = {}
    params = [param]
    if v_syst:
        params.append("v_syst")
    if v_cal:
        params.append("v_cal")
    if offs:
        params.append("offs")
    if f_cal:
        params.append("f_cal")
    return PhaseConfig(
        fix_status_updates=get_dict_of_fixed_status(
            params,
            v_cal=v_cal,
            f_cal=f_cal,
            paths=paths_used,
        ),
        param_val_updates=val_updates,
        **phase_config_defaults,
    )


def get_hypercoeffs(param="A_X", length=True, variance=True):
    if param == "A_X":
        params = ["A_X"]
        if length:
            params.append("A_l")
        if variance:
            params.append("A_var")
    elif param == "v_X":
        params = ["v_X"]
        if length:
            params.append("v_l")
        if variance:
            params.append("v_var")
    elif param == "vσ_X":
        params = ["vσ_X"]
        if length:
            params.append("vσ_l")
        if variance:
            params.append("vσ_var")
    return PhaseConfig(
        fix_status_updates=get_dict_of_fixed_status(params),
        param_val_updates={},
        **phase_config_defaults,
    )


def get_all_coeffs(v_syst=False, offs=False):
    params = ["A_X", "v_X", "vσ_X"]
    if v_syst:
        params.append("v_syst")
    if offs:
        params.append("offs")
    return PhaseConfig(
        fix_status_updates=get_dict_of_fixed_status(params),
        param_val_updates={},
        **phase_config_defaults,
    )


def get_everything():
    params = [alias for alias in paths.keys()]
    return PhaseConfig(
        fix_status_updates=get_dict_of_fixed_status(params),
        param_val_updates={},
        **phase_config_defaults,
    )
