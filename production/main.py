# main.py
# This is the script that we run in terminal to do all the requested fits


import warnings
from pathlib import Path

import jax
import numpy as np
from configs.config_io import load_config, load_hash, save_hash
from configs.data import DRP_FILES
from configs.line_data import LINE_CENTRES
from configs.loss import neg_ln_posterior, neg_ln_posterior_doublet
from lvm_tools import DataConfig, FitDataBuilder, LVMTile, LVMTileCollection
from models.build import model_builder
from spectracles.model.io import load_model, save_model
from spectracles.optimise.opt_schedule import OptimiserSchedule
from whats_on_the_menu import EXPERIMENTAL_MENU, MENU, dish_to_str, load_dish_str, save_dish

# TODO: I think my current schedules do not actually make the hyperparameters vary that I want because I am setting the same one on and off in the same part of the schedule due to the sharing of the same hyperparameter across the kernels for each component. NOTE: We are fixing the hyperparameters now anyway so this is redundant, but in principle it's still an issue for the case where they vary during the fit.

# Configuration for running
# Mostly you don't need to manually skip just take it off the menu
LINES_TO_SKIP = [
    # "Hαλ6563",
    # "[OIII]λ5007",
    # "[NII]λ6584",
    # "[SII]λ6716",
    # "HeIλ4471",
]
OVERWRITE = (
    False  # If True, unmatched config hashes will result in re-running the fits from scratch
)
EXPERIMENTAL = False  # If True, will use experimental fits path

# Handle doublets
DOUBLETS = {
    "[OII]λ3726": "[OII]λ3729",
}

# Don't change this stuff
config_path = Path("configs")
fits_path = Path("fits")
experimental_path = Path("fits_experimental")
if EXPERIMENTAL:
    fits_path = experimental_path
    MENU = EXPERIMENTAL_MENU

# JAX/other stuff
jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")


def collect_lines_and_configs():
    lines = [line for line in LINE_CENTRES.keys() if line not in LINES_TO_SKIP]
    data_configs = {}
    for line in lines:
        config_file = config_path / f"{line}.json"
        if not config_file.exists():
            print(f"Configuration file for {line} does not exist. Skipping.")
            continue
        config = load_config(config_file)
        data_configs[line] = DataConfig.from_dict(config)
    return data_configs


def fits_path_verify(line):
    line_path = fits_path / line
    if not line_path.exists():
        line_path.mkdir(parents=True)
    return line_path


def hash_verify(fit_path, hash_string):
    hash_file = fit_path / "hash.txt"
    # If the hash file doesn't exist, we write one
    if not hash_file.exists():
        save_hash(hash_string, hash_file)
        return False
    else:
        existing_hash = load_hash(hash_file)
        if existing_hash != hash_string:
            if not OVERWRITE:
                raise Exception(
                    f"Configuration hash mismatch for {fit_path}. "
                    "Set OVERWRITE to True to re-run fits."
                )
            print(f"Configuration has changed for {fit_path}. Re-running fit from scratch.")
            save_hash(hash_string, hash_file)
            return False
        else:
            print(f"Config hash matches, resuming from existing fits for {fit_path}.")
            return True


def get_model_name(line, model_number):
    return f"{line}.{model_number:04d}.model"


def main():
    print("Reading data...")
    tiles = LVMTileCollection.from_tiles([LVMTile.from_file(Path(f)) for f in DRP_FILES])

    print("Collecting lines and configurations...")
    configs = collect_lines_and_configs()

    resuming = {}
    for line, config in configs.items():
        # Preprocessing
        print("\n===============")

        # Check that the line is on the menu
        if line not in MENU.keys():
            print(f"{line} is not on the menu. Skipping.")
            continue

        print(f"Processing {line}...")
        fit_path = fits_path_verify(line)
        builder = FitDataBuilder(tiles, config)
        try:
            resuming[line] = hash_verify(fit_path, builder.hash())
        except Exception:
            print(f"Configuration hash mismatch with OVERWRITE=False. Skipping {line}.")
            continue

        # If we we are not resuming, we have to delete the existing fits
        if not resuming[line] and OVERWRITE:
            for existing_fit in fit_path.glob("*.model"):
                existing_fit.unlink()
            for existing_loss in fit_path.glob("*.loss"):
                existing_loss.unlink()
            for existing_fit in fit_path.glob("*.dish"):
                existing_fit.unlink()
            print(f"Deleted existing fits for {line}.")

        # Fit preparation
        fd = builder.build()

        model = None
        model_number = 0
        # Resume
        if resuming[line]:
            print(f"Resuming from previous fit for {line}...")
            # Collect all existing models
            existing_models = sorted(fit_path.glob("*.model"))
            if existing_models:
                last_model = existing_models[-1]
                model = load_model(last_model)
                model_number = int(last_model.name.split(".")[1])
                print(
                    f"Last model found: {last_model.name} (detected model number: {model_number})"
                )
            else:
                print(f"No previous model found for {line}. Starting fresh.")

        # Read the dish/menu file and check stuff hasn't changed (except the schedule)
        dish_file = fit_path / f"{line}.dish"
        if dish_file.exists():
            if dish_to_str(MENU[line]) != load_dish_str(dish_file):
                raise Exception(
                    "Dish configuration has changed. Please either restore the menu, or manually delete the current fit. Automation here isn't supported."
                )
            else:
                print("Dish confirmed scrumptious. Continuing with the fit.")
        else:
            print(f"No dish file found for {line}. Saving current dish to {dish_file}.")
            save_dish(MENU[line], dish_file)

        # Not resuming
        if model is None:
            print(f"Building new model for {line}...")

            if line in DOUBLETS.keys():
                line_centre = (
                    LINE_CENTRES[line],
                    LINE_CENTRES[DOUBLETS[line]],
                )
            else:
                line_centre = LINE_CENTRES[line]

            model, init_model = model_builder(
                model=MENU[line].model,
                line_centre=line_centre,
                fd=fd,
                n_modes=MENU[line].modes,
                bounds=MENU[line].bounds,
                init=MENU[line].init,
                kernel=MENU[line].kernel,
            )
            if model_number != 0:
                raise ValueError(
                    f"Model number {model_number} should be 0 when building a new model. Something went wrong."
                )
            save_model(init_model, fit_path / get_model_name(line, 0))

        # Figure out which loss to use
        if line in DOUBLETS.keys():
            log_p = neg_ln_posterior_doublet
        else:
            log_p = neg_ln_posterior
        # Running the fit
        print(f"Running fit for {line}...")
        schedule = OptimiserSchedule(
            model=model,
            loss_fn=log_p,
            phase_configs=MENU[line].schedule(MENU[line].modes),
        )
        schedule.run_all(
            λ=fd.λ,
            xy_data=fd.αδ_data,
            data=fd.flux,
            u_data=fd.u_flux,
            mask=fd.mask,
        )
        print(f"Saving all models and losses for {line}...")
        for intermediate_mode, intermediate_loss in zip(
            schedule.model_history[1:], schedule.loss_histories
        ):
            model_number += 1
            model_name = get_model_name(line, model_number)
            save_model(intermediate_mode, fit_path / model_name)
            np.save(fit_path / f"{model_name}.loss", intermediate_loss)
            print(f"Saved {model_name} and its loss.")

        print("===============\n")

    print("All fits processed.")


if __name__ == "__main__":
    main()
