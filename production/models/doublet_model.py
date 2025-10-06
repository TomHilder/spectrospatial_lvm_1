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
from spectracles.model.data import SpatialData
from spectracles.model.spatial import PerTile

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


class EmissionLineDoublet(SpectralSpatialModel):
    # Line centre in Angstroms
    μ_1: Parameter
    μ_2: Parameter
    # Model components / line quantities
    A_raw_1: SpatialModel
    A_raw_2: SpatialModel
    v: SpatialModel
    vσ_raw: SpatialModel
    # Measured line quantities
    σ_lsf_1: SpatialModel
    σ_lsf_2: SpatialModel
    v_bary: SpatialModel
    # Systematics
    v_syst: Parameter
    v_cal_1: WaveCalVelocity
    v_cal_2: WaveCalVelocity
    f_cal_raw: SpatialModel

    def __call__(self, λ: Array, spatial_data: SpatialDataLVM) -> Array:
        μ_obs_1 = self.μ_obs_1(spatial_data)
        μ_obs_2 = self.μ_obs_2(spatial_data)
        σ2_obs_1 = self.σ2_obs_1(spatial_data)
        σ2_obs_2 = self.σ2_obs_2(spatial_data)
        peak_1 = self.A_1(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs_1)
        peak_2 = self.A_2(spatial_data) / jnp.sqrt(2 * jnp.pi * σ2_obs_2)
        f_cal = self.f_cal(spatial_data)
        line_1 = peak_1 * jnp.exp(-0.5 * (λ - μ_obs_1) ** 2 / σ2_obs_1)
        line_2 = peak_2 * jnp.exp(-0.5 * (λ - μ_obs_2) ** 2 / σ2_obs_2)
        return f_cal * (line_1 + line_2)

    def A_1(self, s) -> Array:
        return l_bounded(self.A_raw_1(s), lower=A_LOWER)

    def A_2(self, s) -> Array:
        return l_bounded(self.A_raw_2(s), lower=A_LOWER)

    def vσ(self, s) -> Array:
        return l_bounded(self.vσ_raw(s), lower=0.0)

    def v_obs_1(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal_1(s) - self.v_bary(s)

    def v_obs_2(self, s) -> Array:
        return self.v(s) + self.v_syst.val + self.v_cal_2(s) - self.v_bary(s)

    def μ_obs_1(self, s) -> Array:
        return self.μ_1.val * (1 + self.v_obs_1(s) / C_KMS)

    def μ_obs_2(self, s) -> Array:
        return self.μ_2.val * (1 + self.v_obs_2(s) / C_KMS)

    def σ2_obs_1(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs_1(s) / C_KMS) ** 2 + self.σ_lsf_1(s) ** 2

    def σ2_obs_2(self, s) -> Array:
        return (self.vσ(s) * self.μ_obs_2(s) / C_KMS) ** 2 + self.σ_lsf_2(s) ** 2

    def f_cal(self, s) -> Array:
        return l_bounded(self.f_cal_raw(s), lower=0.0) / l_bounded(0, lower=0.0)


class LVMModelDoublet(SpectralSpatialModel):
    # Model components
    line: EmissionLineDoublet  # Emission line model
    offs: PerSpaxel  # Nuisance offsets per spaxel

    def __init__(
        self,
        n_tiles: int,
        n_spaxels: int,
        offsets: Parameter,
        line_centre_1: Parameter,
        line_centre_2: Parameter,
        n_modes: tuple[int, int],
        A_kernel_1: Kernel,
        A_kernel_2: Kernel,
        v_kernel: Kernel,
        σ_kernel: Kernel,
        σ_lsf_1: Parameter,
        σ_lsf_2: Parameter,
        v_bary: Parameter,
        v_syst: Parameter,
        C_v_cal: Parameter,  # MUST be 2 values i.e. shape is (2,)
        f_cal_unconstrained: Parameter,
    ):
        self.offs = Constant(const=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=offsets))
        self.line = EmissionLineDoublet(
            μ_1=line_centre_1,
            μ_2=line_centre_2,
            A_raw_1=FourierGP(n_modes=n_modes, kernel=A_kernel_1),
            A_raw_2=FourierGP(n_modes=n_modes, kernel=A_kernel_2),
            v=FourierGP(n_modes=n_modes, kernel=v_kernel),
            vσ_raw=FourierGP(n_modes=n_modes, kernel=σ_kernel),
            σ_lsf_1=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_1),
            σ_lsf_2=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=σ_lsf_2),
            v_bary=PerSpaxel(n_spaxels=n_spaxels, spaxel_values=v_bary),
            v_syst=v_syst,
            v_cal_1=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre_1),
            v_cal_2=WaveCalVelocity(C_v_cal=C_v_cal, μ=line_centre_2),
            f_cal_raw=PerTile(n_tiles=n_tiles, tile_values=f_cal_unconstrained),
        )

    def __call__(self, λ, spatial_data):
        return self.offs(λ, spatial_data) + self.line(λ, spatial_data)
