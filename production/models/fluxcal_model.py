import astropy.constants as const
import jax.numpy as jnp
from jaxtyping import Array
from spectracles import (
    Constant,
    FourierGP,
    Kernel,
    Parameter,
    PerSpaxel,
    SpatialDataLVM,
    SpatialModel,
    SpectralSpatialModel,
    l_bounded,
)
from spectracles.model.spatial import PerTile

A_LOWER = 0.0

C_KMS = const.c.to_value("km/s")


class EmissionLineFluxCal(SpectralSpatialModel):
    # Line centre in Angstroms
    μ: Parameter
    # Model components / line quantities
    A_raw: SpatialModel  # Unconstrained line flux
    v: SpatialModel  # Radial velocity in rest frame in km/s
    vσ_raw: SpatialModel  # Broadening velocity in km/s before constraint
    # Measured line quantities
    σ_lsf: SpatialModel  # LSF width (std dev) in Angstroms
    v_bary: SpatialModel  # Barycentric velocity CORRECTION in km/s
    # Systematics
    v_syst: Parameter  # Systematic velocity offset in km/s
    f_cal_raw: SpatialModel  # Flux calibration factor per tile

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        f_cal = self.f_cal(spatial_data)
        return f_cal * peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s), lower=0.0)

    def v_obs(self, s) -> Array:
        return self.v(s) + self.v_syst.val - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs(s) / C_KMS) ** 2 + self.σ_lsf(s) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class LVMModelFluxCal(SpectralSpatialModel):
    # Model components
    line: EmissionLineFluxCal  # Emission line model
    offs: PerSpaxel  # Nuisance offsets per spaxel

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        offsets: Parameter,
        line_centre: Parameter,
        n_modes: tuple[int, int],
        A_kernel: Kernel,
        v_kernel: Kernel,
        σ_kernel: Kernel,
        σ_lsf: Parameter,
        v_bary: Parameter,
        v_syst: Parameter,
        f_cal_unconstrained: Parameter,
    ):
        self.offs = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = EmissionLineFluxCal(
            μ=line_centre,
            A_raw=FourierGP(n_modes=n_modes, kernel=A_kernel),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel),
            vσ_raw=FourierGP(n_modes=n_modes, kernel=σ_kernel),
            σ_lsf=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf),
            v_bary=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary),
            v_syst=v_syst,
            f_cal_raw=PerTile(n_tiles=n_tiles, tile_values=f_cal_unconstrained),
        )

    def __call__(self, λ, spatial_data):
        return self.offs(λ, spatial_data) + self.line(λ, spatial_data)
