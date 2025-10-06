import jax
import jax.numpy as jnp
from modelling_lib.optimise.opt_schedule import PhaseConfig
from numpy.random import default_rng
from optax import adam, lbfgs

rng = default_rng(0)
jax.config.update("jax_enable_x64", True)


N_STEPS_MULTIPLIER: int = 100

N_MODES = (151, 151)  # We shouldn't have this in both places but fine for now

# Phase for varying the line strength and nuisance offsets but nothing else
A_coeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 20,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.A.coefficients": False,
        "continuum.const.spaxel_values": False,
        # Fixed:
        "line.A.kernel.length_scale": True,
        "line.A.kernel.variance": True,
        "line.λ0.coefficients": True,
        # "line.λ0.kernel.length_scale": True,
        "line.λ0.kernel.variance": True,
        "line.λ0_all": True,
        "line.σ_lsf": True,
        "line.σ.coefficients": True,
        "line.σ.kernel.length_scale": True,
        "line.σ.kernel.variance": True,
    },
)
A_hypercoeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 1,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.A.coefficients": False,
        "line.A.kernel.length_scale": False,
        "line.A.kernel.variance": False,
        # Fixed:
        "continuum.const.spaxel_values": True,
        # "line.λ0.kernel.length_scale": True,
        "line.λ0.kernel.variance": True,
        "line.λ0.coefficients": True,
        "line.λ0_all": True,
        "line.σ_lsf": True,
        "line.σ.coefficients": True,
        "line.σ.kernel.length_scale": True,
        "line.σ.kernel.variance": True,
    },
)
λ0_system = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 1,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.λ0_all": False,
        # Fixed:
        "continuum.const.spaxel_values": True,
        "line.A.coefficients": True,
        "line.A.kernel.length_scale": True,
        "line.A.kernel.variance": True,
        "line.λ0.coefficients": True,
        # "line.λ0.kernel.length_scale": True,
        "line.λ0.kernel.variance": True,
        "line.σ_lsf": True,
        "line.σ.coefficients": True,
        "line.σ.kernel.length_scale": True,
        "line.σ.kernel.variance": True,
    },
)
λ0_coeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 20,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.λ0.coefficients": False,
        # Fixed:
        "continuum.const.spaxel_values": True,
        "line.A.coefficients": True,
        "line.A.kernel.length_scale": True,
        "line.A.kernel.variance": True,
        "line.λ0_all": True,
        # "line.λ0.kernel.length_scale": True,
        "line.λ0.kernel.variance": True,
        "line.σ_lsf": True,
        "line.σ.coefficients": True,
        "line.σ.kernel.length_scale": True,
        "line.σ.kernel.variance": True,
    },
    param_val_updates={"line.λ0.coefficients": jnp.array(rng.standard_normal(N_MODES))},
)
λ0_hypercoeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 1,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.λ0.coefficients": False,
        # "line.λ0.kernel.length_scale": False,
        "line.λ0.kernel.variance": False,
        # Fixed:
        "continuum.const.spaxel_values": True,
        "line.A.coefficients": True,
        "line.A.kernel.length_scale": True,
        "line.A.kernel.variance": True,
        "line.λ0_all": True,
        "line.σ_lsf": True,
        "line.σ.coefficients": True,
        "line.σ.kernel.length_scale": True,
        "line.σ.kernel.variance": True,
    },
)
Aλ0_all = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 1,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.A.coefficients": False,
        "line.A.kernel.length_scale": False,
        "line.A.kernel.variance": False,
        "line.λ0.coefficients": False,
        # "line.λ0.kernel.length_scale": False,
        "line.λ0.kernel.variance": False,
        "line.λ0_all": False,
        # Fixed:
        "line.σ_lsf": True,
        "continuum.const.spaxel_values": True,
        "line.σ.coefficients": True,
        "line.σ.kernel.length_scale": True,
        "line.σ.kernel.variance": True,
    },
)
# Aλ0_all = PhaseConfig(
#     n_steps=N_STEPS_MULTIPLIER * 100,
#     optimiser=lbfgs(),
#     fix_status_updates={
#         # Allowed to vary:
#         "line.A.coefficients": False,
#         "line.A.kernel.length_scale": False,
#         "line.A.kernel.variance": False,
#         "line.λ0.coefficients": False,
#         # "line.λ0.kernel.length_scale": False,
#         "line.λ0.kernel.variance": False,
#         "line.λ0_all": False,
#         "line.σ_lsf": False,
#         # Fixed:
#         "continuum.const.spaxel_values": False,
#         "line.σ.coefficients": False,
#         "line.σ.kernel.length_scale": False,
#         "line.σ.kernel.variance": False,
#     },
# )
σ_coeffs = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 20,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.σ.coefficients": False,
        "line.σ.kernel.length_scale": False,
        "line.σ.kernel.variance": False,
        "line.σ_lsf": False,
        # Fixed:
        "continuum.const.spaxel_values": True,
        "line.A.coefficients": True,
        "line.A.kernel.length_scale": True,
        "line.A.kernel.variance": True,
        "line.λ0.coefficients": True,
        "line.λ0_all": True,
        # "line.λ0.kernel.length_scale": True,
        "line.λ0.kernel.variance": True,
    },
    param_val_updates={"line.σ.coefficients": jnp.array(rng.standard_normal(N_MODES))},
)
everything = PhaseConfig(
    n_steps=N_STEPS_MULTIPLIER * 20,
    optimiser=lbfgs(),
    fix_status_updates={
        # Allowed to vary:
        "line.σ.coefficients": False,
        "line.σ.kernel.length_scale": False,
        "line.σ.kernel.variance": False,
        "line.σ_lsf": False,
        "continuum.const.spaxel_values": False,
        "line.A.coefficients": False,
        "line.A.kernel.length_scale": False,
        "line.A.kernel.variance": False,
        "line.λ0.coefficients": False,
        "line.λ0_all": False,
        # "line.λ0.kernel.length_scale": False,
        "line.λ0.kernel.variance": False,
    },
)


all_phases: list[PhaseConfig] = [
    A_coeffs,
    A_hypercoeffs,
    λ0_system,
    λ0_coeffs,
    λ0_hypercoeffs,
    Aλ0_all,
    # σ_coeffs,
    # everything,
]
