from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from models.build import DEFAULT_BOUNDS, DEFAULT_INIT
from models.doublet_model import LVMModelDoublet
from models.fluxcal_model import LVMModelFluxCal
from models.model import LVMModel
from models.production_model import LVMModelProduction
from models.wavecal_model import LVMModelWaveCal
from schedules.schedules import (
    double_pass_no_hyper_no_all,
    double_pass_no_hyper_no_all_doublet,
    get_first_pass_no_hyper,
    get_refine_A,
    get_refine_coeffs_only_individually,
    get_refine_v,
    get_refine_vσ,
    get_subsequent_pass_phases,
)
from spectracles import Kernel, Matern12, Matern32
from spectracles.model.share_module import ShareModule


@dataclass(frozen=True)
class Dish:
    # This is the only one you're allowed to change after starting the fit
    schedule: Callable

    # Constant after model is built/initialised
    model: ShareModule
    kernel: Kernel
    modes: tuple[int, int]
    bounds: dict = field(default_factory=lambda: DEFAULT_BOUNDS)
    init: dict = field(default_factory=lambda: DEFAULT_INIT)


MENU = {
    "[OII]λ3726": Dish(
        model=LVMModelDoublet,
        kernel=Matern32,
        modes=(401, 401),
        schedule=lambda n_modes: double_pass_no_hyper_no_all_doublet(n_modes, v_syst=True),
    ),
    # "[OII]λ3727": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ3820": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CaIIλ3934": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ4026": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "Hδλ4102": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ4121": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "Hγλ4340": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CIIλ4267": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[OIII]λ4363": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ4471": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "OIIλ4650": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "OIIλ4660": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIIλ4686": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "Hβλ4861": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[OIII]λ5007": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ5016": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "NIIλ5680": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[NII]λ5755": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ5876": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "NaIλ5890": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "NaIλ5896": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[SIII]λ6312": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[NII]λ6548": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "Hαλ6563": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(451, 451),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CIIλ6578": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[NII]λ6583": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ6678": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[SII]λ6716": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[SII]λ6731": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "FeIλ6855": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "HeIλ7065": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CIIλ7236": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     # schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    #     schedule=lambda n_modes: get_refine_A(n_modes, v_syst=True, f_cal=True),
    # ),
    # "HeIλ7281": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[OII]λ7319": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "OIλ7774": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "OIλ8446": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CaIIλ8498": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[SIII]λ9069": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[SIII]λ9531": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(401, 401),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
}

EXPERIMENTAL_MENU = {
    # "Hβλ4861": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(351, 351),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CIIλ6578": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(251, 251),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "[OIII]λ4363": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(251, 251),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "CIIλ4267": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(251, 251),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
    # "NIIλ5680": Dish(
    #     model=LVMModelProduction,
    #     kernel=Matern32,
    #     modes=(251, 251),
    #     schedule=lambda n_modes: double_pass_no_hyper_no_all(n_modes, v_syst=True),
    # ),
}


def dish_to_str(dish: Dish):
    return (
        f"{dish.model.__name__}, {dish.kernel.__name__}, {dish.modes}"
        f"bounds={dish.bounds}, init={dish.init}"
    )


def save_dish(dish: Dish, filepath: Path = Path("default.dish")):
    """Save dish to txt file coverting everything to strings using dish_to_str."""
    with filepath.open("w") as f:
        f.write(dish_to_str(dish))


def load_dish_str(filepath: Path = Path("default.dish")) -> str:
    """Load dish from txt file."""
    with filepath.open("r") as f:
        return f.read().strip()
