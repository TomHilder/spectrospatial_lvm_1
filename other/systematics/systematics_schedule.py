import jax
import jax.numpy as jnp
from numpy.random import default_rng
from optax import lbfgs
from spectracles.optimise.opt_schedule import PhaseConfig

rng = default_rng(0)
jax.config.update("jax_enable_x64", True)


# N_STEPS_MULTIPLIER: int = 50
N_STEPS_MULTIPLIER: int = 5000

# N_MODES = (451, 451)  # We shouldn't have this in both places but fine for now
N_MODES = (351, 351)  # We shouldn't have this in both places but fine for now
# N_MODES = (151, 151)  # We shouldn't have this in both places but fine for now
# N_MODES = (51, 51)  # We shouldn't have this in both places but fine for now

opt = lbfgs()

Δloss_rough = 1e2
Δloss_final = 1e0

# Phase for varying the line strength and nuisance offsets but nothing else
A_coeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 10,
    optimiser=opt,
    Δloss_criterion=Δloss_rough,
    fix_status_updates={
        # Allowed to vary:
        "line.A_raw.coefficients": False,
        "line.v_syst": False,
        "offs.const.spaxel_values": False,
        # Fixed:
        "line.v.coefficients": True,
        "line.vσ_raw.coefficients": True,
    },
    param_val_updates={
        "line.A_raw.coefficients": jnp.array(rng.standard_normal(N_MODES)),
    },
)
A_hypercoeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 2,
    optimiser=opt,
    Δloss_criterion=Δloss_rough,
    fix_status_updates={
        # Allowed to vary:
        "line.A_raw.coefficients": False,
        "line.A_raw.kernel.length_scale": False,
        "line.A_raw.kernel.variance": False,  # TODO: change back
        # Fixed:
        "offs.const.spaxel_values": True,
        "line.v_syst": True,
    },
)
v_coeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 10,
    optimiser=opt,
    Δloss_criterion=Δloss_rough,
    fix_status_updates={
        # Allowed to vary:
        "line.v.coefficients": False,
        "line.v_syst": True,
        # Fixed:
        "line.A_raw.coefficients": True,
        "line.A_raw.kernel.length_scale": True,
        "line.A_raw.kernel.variance": True,
    },
    param_val_updates={
        "line.v.coefficients": jnp.array(rng.standard_normal(N_MODES)),
    },
)
v_hypercoeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 2,
    optimiser=opt,
    Δloss_criterion=Δloss_rough,
    fix_status_updates={
        # Allowed to vary:
        "line.v.coefficients": False,
        "line.v.kernel.length_scale": False,
        "line.v.kernel.variance": False,  # TODO: change back
        # Fixed:
        "line.v_syst": True,
    },
)
σ_coeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 10,
    optimiser=opt,
    Δloss_criterion=Δloss_rough,
    fix_status_updates={
        # Allowed to vary:
        "line.vσ_raw.coefficients": False,
        # Fixed:
        "line.v.coefficients": True,
        "line.v.kernel.length_scale": True,
        "line.v.kernel.variance": True,
    },
    param_val_updates={
        "line.vσ_raw.coefficients": jnp.array(rng.standard_normal(N_MODES)),
    },
)
σ_hypercoeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 2,
    optimiser=opt,
    Δloss_criterion=Δloss_rough,
    fix_status_updates={
        # Allowed to vary:
        "line.vσ_raw.coefficients": False,
        "line.vσ_raw.kernel.length_scale": False,
        "line.vσ_raw.kernel.variance": False,  # TODO: change back
    },
)
all_so_far_old = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 40,
    optimiser=opt,
    Δloss_criterion=Δloss_final,
    fix_status_updates={
        # Line peak
        "line.A_raw.coefficients": False,
        "line.A_raw.kernel.length_scale": True,
        "line.A_raw.kernel.variance": True,  # TODO: change back
        # Radial velocity
        "line.v_syst": True,  # TODO: change back (probably???)
        "line.v.coefficients": False,
        "line.v.kernel.variance": True,  # TODO: change back
        "line.v.kernel.length_scale": True,  # TODO: change back
        # Offsets
        "offs.const.spaxel_values": False,
    },
)
all_so_far = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 40 * 3,
    optimiser=opt,
    Δloss_criterion=Δloss_final,
    fix_status_updates={
        # Line peak
        "line.A_raw.coefficients": False,
        "line.A_raw.kernel.length_scale": False,
        "line.A_raw.kernel.variance": False,
        # Radial velocity
        "line.v_syst": False,
        "line.v.coefficients": False,
        "line.v.kernel.variance": False,
        # Line width
        "line.vσ_raw.coefficients": False,
        "line.vσ_raw.kernel.length_scale": False,
        "line.vσ_raw.kernel.variance": False,
        # Offsets
        "offs.const.spaxel_values": False,
    },
)


all_phases: list[PhaseConfig] = [
    A_coeffs,
    A_hypercoeffs,
    # v_coeffs,
    # v_hypercoeffs,
    # σ_coeffs,
    # σ_hypercoeffs,
    # all_so_far_old,
    # all_so_far,
]
