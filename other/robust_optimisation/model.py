from typing import Callable

import jax.numpy as jnp
from jaxtyping import Array
from modelling_lib import (
    Constant,
    FourierGP,
    Kernel,
    Parameter,
    PerSpaxel,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)
from modelling_lib import (
    SpatialDataGeneric as SpatialData,
)
from modelling_lib.model.parameter import l_bounded_inv

A_LOWER = 1e-6
σ_LOWER = 1e-12


# class FourierGPWithMean(SpatialModel):
#     # Model components
#     gp: FourierGP
#     # Model parameters
#     mean: Parameter

#     def __call__(self, data: SpatialData) -> Array:
#         return self.gp(data) + self.mean.val


class GaussianLineModel(SpectralSpatialModel):
    # Model parameters
    A: SpatialModel
    λ0: SpatialModel
    σ: SpatialModel
    σ_lsf: Parameter
    σ_lsf_uncon: float
    λ0_all: Parameter

    def __call__(self, λ: Array, spatial_data: SpatialData):
        A_positive = self.A_positive(spatial_data)
        σ_positive = self.σ_positive(spatial_data)
        A_norm = A_positive / (σ_positive * jnp.sqrt(2 * jnp.pi))
        return A_norm * jnp.exp(
            -0.5 * ((λ - self.λ0(spatial_data) - self.λ0_all.val) / σ_positive) ** 2
        )

    def A_positive(self, s) -> Callable:
        return l_bounded(self.A(s), lower=A_LOWER)

    def σ_positive(self, s) -> Callable:
        return self.σ(s) ** 2 + self.σ_lsf.val
        # return lambda s: l_bounded(self.σ(s) + self.σ_lsf_uncon, lower=σ_LOWER) + self.σ_lsf.val
        # return lambda s: self.σ_lsf.val


class LVMModel(SpectralSpatialModel):
    # Model components
    continuum: Constant
    line: GaussianLineModel

    def __init__(
        self,
        n_spaxels: int,
        n_modes: tuple[int, int],
        A_kernel: Kernel,
        λ0_kernel: Kernel,
        λ0_all: Parameter,
        σ_kernel: Kernel,
        σ_lsf: Parameter,
        offsets: Parameter,
    ):
        self.continuum = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = GaussianLineModel(
            A=FourierGP(n_modes, A_kernel),
            λ0=FourierGP(n_modes, λ0_kernel),
            σ=FourierGP(n_modes, σ_kernel),
            σ_lsf=σ_lsf,
            σ_lsf_uncon=l_bounded_inv(1e-5, lower=σ_LOWER),
            λ0_all=λ0_all,
        )

    def __call__(self, λ, spatial_data):
        return self.continuum(λ, spatial_data) + self.line(λ, spatial_data)
