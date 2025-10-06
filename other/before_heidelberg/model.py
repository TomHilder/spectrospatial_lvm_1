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
from modelling_lib import SpatialDataGeneric as SpatialData

A_LOWER = 1e-2
σ_LOWER = 1e-2


class FourierGPWithMean(SpatialModel):
    # Model components
    gp: FourierGP
    # Model parameters
    mean: Parameter

    def __call__(self, data: SpatialData) -> Array:
        return self.gp(data) + self.mean.val


class ConstrainedGaussian(SpectralSpatialModel):
    # Model parameters
    A: SpatialModel
    λ0: SpatialModel
    σ: SpatialModel

    def __init__(self, A: SpatialModel, λ0: SpatialModel, σ: SpatialModel):
        self.A = A
        self.λ0 = λ0
        self.σ = σ

    def __call__(self, λ: Array, spatial_data: SpatialData):
        A_positive = self.A_positive(spatial_data)
        σ_positive = self.σ_positive(spatial_data)
        A_norm = A_positive / (σ_positive * jnp.sqrt(2 * jnp.pi))
        return A_norm * jnp.exp(-0.5 * ((λ - self.λ0(spatial_data)) / σ_positive) ** 2)

    @property
    def A_positive(self) -> Callable:
        return lambda s: l_bounded(self.A(s), lower=A_LOWER)

    @property
    def σ_positive(self) -> Callable:
        return lambda s: l_bounded(self.σ(s), lower=σ_LOWER)


class LVMModel(SpectralSpatialModel):
    # Model components
    continuum: Constant
    line: ConstrainedGaussian

    def __init__(
        self,
        n_spaxels: int,
        n_modes: tuple[int, int],
        A_kernel: Kernel,
        λ0_kernel: Kernel,
        σ_kernel: Kernel,
        σ_mean: Parameter,
        offsets: Parameter,
    ):
        self.continuum = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = ConstrainedGaussian(
            A=FourierGP(n_modes, A_kernel),
            λ0=FourierGP(n_modes, λ0_kernel),
            σ=FourierGPWithMean(FourierGP(n_modes, σ_kernel), σ_mean),
        )

    def __call__(self, λ, spatial_data):
        return self.continuum(λ, spatial_data) + self.line(λ, spatial_data)
