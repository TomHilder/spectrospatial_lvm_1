import astropy.constants as const
import jax
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
from spectracles.model.data import SpatialData
from spectracles.model.spatial import PerIFU, PerIFUAndTile

A_LOWER = 0.0

C_KMS = const.c.to_value("km/s")


class WaveCalVelocity(SpatialModel):
    # Model parameters
    C_v_cal: Parameter  # 2 value model parameter
    # Constants
    μ: Parameter  # line centre in Angstroms

    def __call__(self, data: SpatialData) -> Array:
        v0 = 0.0  # effectively pinned to v_syst
        v1 = v0 - C_KMS * (self.C_v_cal.val[0] / self.μ.val[0])
        v2 = v1 - C_KMS * (self.C_v_cal.val[1] / self.μ.val[0])
        v_ifu_vals = jnp.array([v0, v1, v2])
        return v_ifu_vals[data.ifu_idx]

    # def __call__(self, data: SpatialData) -> Array:
    #     jax.debug.print("C_v_cal.val shape: {}", self.C_v_cal.val.shape)
    #     jax.debug.print("μ.val shape: {}", self.μ.val.shape)

    #     c0 = self.C_v_cal.val[0]
    #     c1 = self.C_v_cal.val[1]
    #     jax.debug.print("c0 shape: {}", c0.shape)
    #     jax.debug.print("c1 shape: {}", c1.shape)

    #     div0 = c0 / self.μ.val
    #     div1 = c1 / self.μ.val
    #     jax.debug.print("div0 shape: {}", div0.shape)
    #     jax.debug.print("div1 shape: {}", div1.shape)

    #     v0 = 0.0
    #     v1 = v0 - div0
    #     v2 = v1 - div1
    #     jax.debug.print("v1 shape: {}", v1.shape)
    #     jax.debug.print("v2 shape: {}", v2.shape)

    #     v_ifu_vals = jnp.array([v0, v1, v2])
    #     return v_ifu_vals[data.ifu_idx]


class EmissionLineWaveCal(SpectralSpatialModel):
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
    v_cal: WaveCalVelocity  # Per-IFU Velocity calibration offset in km/s

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs = self.μ_obs(spatial_data)
        σ2_obs = self.σ2_obs(spatial_data)
        peak = self.A(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs)
        return peak * jnp.exp(-0.5 * (λ - μ_obs) ** 2 / σ2_obs)

    def A(self, s) -> Array:
        return l_bounded(self.A_raw(s), lower=A_LOWER)

    def vσ(self, s) -> Array:
        # return self.vσ_raw(s)
        return l_bounded(self.vσ_raw(s), lower=0.0)

    def v_obs(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal(s) - self.v_bary(s)

    def μ_obs(self, s) -> Array:
        return self.μ.val * (1 + self.v_obs(s) / C_KMS)

    def σ2_obs(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs(s) / C_KMS) ** 2 + self.σ_lsf(s) ** 2


class LVMModelWaveCal(SpectralSpatialModel):
    # Model components
    line: EmissionLineWaveCal  # Emission line model
    offs: PerSpaxel  # Nuisance offsets per spaxel

    def __init__(
        self,
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
        C_v_cal: Parameter,  # MUST be 2 values i.e. shape is (2,)
    ):
        self.offs = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = EmissionLineWaveCal(
            μ=line_centre,
            A_raw=FourierGP(n_modes=n_modes, kernel=A_kernel),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel),
            vσ_raw=FourierGP(n_modes=n_modes, kernel=σ_kernel),
            σ_lsf=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf),
            v_bary=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary),
            v_syst=v_syst,
            v_cal=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre),
        )

    def __call__(self, λ, spatial_data):
        return self.offs(λ, spatial_data) + self.line(λ, spatial_data)
