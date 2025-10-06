from .phases import get_all_coeffs, get_coeffs, get_everything, get_hypercoeffs

# These all have to have the same call signature because I'm stupid


# def get_first_pass_phases(n_modes):
#     return [
#         get_coeffs("A_X", v_syst=True, offs=True, n_modes=n_modes, init=True),
#         get_hypercoeffs("A_X"),
#         get_coeffs("v_X", n_modes=n_modes, init=True),
#         get_hypercoeffs("v_X"),
#         get_coeffs("vσ_X", n_modes=n_modes, init=True),
#         get_hypercoeffs("vσ_X"),
#         get_all_coeffs(),
#     ]


# def get_first_pass_phases_l_only(n_modes):
#     return [
#         get_coeffs("A_X", v_syst=True, offs=True, n_modes=n_modes, init=True),
#         get_hypercoeffs("A_X", length=True, variance=False),
#         get_coeffs("v_X", n_modes=n_modes, init=True),
#         get_hypercoeffs("v_X", length=True, variance=False),
#         get_coeffs("vσ_X", n_modes=n_modes, init=True),
#         get_hypercoeffs("vσ_X", length=True, variance=False),
#         get_all_coeffs(),
#     ]


def get_first_pass_no_hyper(n_modes):
    return [
        get_coeffs("A_X", v_syst=True, offs=True, n_modes=n_modes, init=True),
        get_coeffs("v_X", n_modes=n_modes, init=True),
        get_coeffs("vσ_X", n_modes=n_modes, init=True),
        get_all_coeffs(),
    ]


def double_pass_no_hyper_no_all(n_modes, v_syst=False, v_cal=False, f_cal=False):
    return [
        get_coeffs(
            "A_X",
            v_syst=v_syst,
            v_cal=v_cal,
            offs=True,
            f_cal=f_cal,
            n_modes=n_modes,
            init=True,
        ),
        get_coeffs("v_X", n_modes=n_modes, v_cal=v_cal, init=True),
        get_coeffs("vσ_X", n_modes=n_modes, init=True),
        get_coeffs("A_X", v_syst=v_syst, v_cal=v_cal, offs=True, f_cal=f_cal, n_modes=n_modes),
        get_coeffs("v_X", n_modes=n_modes, v_cal=v_cal),
        get_coeffs("vσ_X", n_modes=n_modes),
    ]


def double_pass_no_hyper_no_all_doublet(n_modes, v_syst=False, v_cal=False, f_cal=False):
    return [
        get_coeffs(
            "A_X_1",
            v_syst=v_syst,
            v_cal=v_cal,
            offs=True,
            f_cal=f_cal,
            n_modes=n_modes,
            init=True,
            doublet=True,
        ),
        get_coeffs(
            "A_X_2",
            v_syst=v_syst,
            v_cal=v_cal,
            offs=True,
            f_cal=f_cal,
            n_modes=n_modes,
            init=True,
            doublet=True,
        ),
        get_coeffs(
            "v_X",
            n_modes=n_modes,
            v_cal=v_cal,
            init=True,
            doublet=True,
        ),
        get_coeffs(
            "vσ_X",
            n_modes=n_modes,
            init=True,
            doublet=True,
        ),
        get_coeffs(
            "A_X_1",
            v_syst=v_syst,
            v_cal=v_cal,
            offs=True,
            f_cal=f_cal,
            n_modes=n_modes,
            doublet=True,
        ),
        get_coeffs(
            "A_X_2",
            v_syst=v_syst,
            v_cal=v_cal,
            offs=True,
            f_cal=f_cal,
            n_modes=n_modes,
            doublet=True,
        ),
        get_coeffs(
            "v_X",
            n_modes=n_modes,
            v_cal=v_cal,
            doublet=True,
        ),
        get_coeffs(
            "vσ_X",
            n_modes=n_modes,
            doublet=True,
        ),
    ]


def get_subsequent_pass_phases(n_modes, v_cal=False, f_cal=False):
    return [
        get_coeffs("A_X", v_syst=True, v_cal=v_cal, offs=True, f_cal=f_cal),
        # get_hypercoeffs("A_X"),
        get_coeffs("v_X", v_cal=v_cal),
        # get_hypercoeffs("v_X"),
        get_coeffs("vσ_X", v_cal=v_cal),
        # get_hypercoeffs("vσ_X"),
        get_all_coeffs(),
    ]


def get_refine_coeffs_only_individually(n_modes, v_cal=False, f_cal=False):
    return [
        get_coeffs("A_X", v_syst=True, v_cal=v_cal, f_cal=f_cal),
        get_coeffs("v_X", v_cal=v_cal),
        get_coeffs("vσ_X", v_cal=v_cal),
    ]


def get_refine_A(n_modes, v_syst=False, v_cal=False, f_cal=False):
    return [
        get_coeffs("A_X", v_syst=v_syst, v_cal=v_cal, f_cal=f_cal),
    ]


def get_refine_v(n_modes):
    return [
        get_coeffs("v_X"),
    ]


def get_refine_vσ(n_modes):
    return [
        get_coeffs("vσ_X"),
    ]


# def get_all_params(n_modes):
#     return [
#         get_everything(),
#     ]
