import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binned_statistic
from spectracles.model.spatial import get_freqs

plt.style.use("mpl_drip.custom")

rng = np.random.default_rng(0)

N_MODES = 401
TRUE_S = 13
TRUE_VAR = 2e5**2


def kernel(freqs, length, nu=1.5, n=2):
    return (1 + (freqs * length) ** 2) ** (-0.5 * (nu + n / 2))


# def kernel(freqs, length):
#     exp_arg = -0.25 * np.abs(freqs * length) ** 2
#     clamped_exp_arg = np.clip(exp_arg, -745.0, 0.0)
#     return np.exp(clamped_exp_arg)


def get_alpha_hat(freqs, s, X):
    return np.sum(np.abs(X) ** 2 / kernel(freqs, s) ** 2) / len(X)


def estimate_s(
    k_func,
    freqs,
    samples,
    bounds=None,
    log_prior=None,
    tol=1e-10,
    var=1,
):
    freqs = np.asarray(freqs, float)
    X = np.asarray(samples, float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != freqs.size:
        raise ValueError("samples second dimension must match freqs length")

    S2 = np.sum(X * X, axis=0)
    N = X.shape[0]

    def neg_logpost(s):
        psd = k_func(freqs, float(s))
        kv = var * psd
        ll = -0.5 * (N * np.sum(np.log(kv)) + np.sum(S2 / kv))
        return -ll

    if bounds is not None:
        res = minimize_scalar(
            neg_logpost, bounds=tuple(map(float, bounds)), method="bounded", options={"xatol": tol}
        )
    else:
        res = minimize_scalar(neg_logpost, method="brent", options={"xtol": tol})

    s_hat = float(res.x)

    return (
        s_hat,
        {
            "fun": float(res.fun),
            "success": bool(res.success),
            "nit": int(res.nfev),
            "message": res.message,
        },
    )


def estimate_s_nu(
    k_func,
    freqs,
    samples,
    bounds=None,  # should be [(s_low, s_high), (nu_low, nu_high)] if used
    log_prior=None,  # function taking (s, nu) -> log_prior
    tol=1e-10,
    var=1,
):
    freqs = np.asarray(freqs, float)
    X = np.asarray(samples, float)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != freqs.size:
        raise ValueError("samples second dimension must match freqs length")

    S2 = np.sum(X * X, axis=0)
    N = X.shape[0]

    def neg_logpost(params):
        s, nu = map(float, params)

        psd = k_func(freqs, s, nu)
        kv = var * psd

        # psd_shape = k_func(freqs, s, nu)  # returns PSD (i.e., feature_weights**2)
        # phi = np.asarray(psd_shape, float)
        # S2 = np.sum(np.real(X * np.conj(X)), axis=0)  # per-frequency sum over samples
        # # Closed-form amplitude given theta:
        # A_hat = (np.sum(S2 / phi)) / (N * phi.size)
        # # Profiled negative log-likelihood:
        # kv = A_hat * phi

        ll = -0.5 * (N * np.sum(np.log(kv)) + np.sum(S2 / kv))
        if log_prior is not None:
            ll += log_prior(s, nu)
        return -ll

    res = minimize(
        neg_logpost,
        x0=[0.1, 1.5],  # starting guesses; adjust as needed
        bounds=bounds,
        tol=tol,
    )

    s_hat, nu_hat = map(float, res.x)

    return (
        (s_hat, nu_hat),
        {
            "fun": float(res.fun),
            "success": bool(res.success),
            "nit": int(res.nit),
            "message": res.message,
        },
    )


if __name__ == "__main__":
    fx, fy = get_freqs((N_MODES, N_MODES))
    freqs = np.sqrt(fx**2 + fy**2)

    freqs_plot = np.linspace(freqs.min(), freqs.max(), 200)

    fw_prior = np.sqrt(TRUE_VAR) * kernel(freqs, length=TRUE_S)
    fw_prior_plot = np.sqrt(TRUE_VAR) * kernel(freqs_plot, length=TRUE_S)
    X_prior = fw_prior * rng.standard_normal(size=freqs.shape)

    # plt.imshow(X_prior, cmap="RdBu", vmin=-2, vmax=2)
    # plt.show()

    s_hat, var_hat, info = estimate_s(
        lambda f, s: kernel(f, s) ** 2,
        freqs.flatten(),
        X_prior.flatten(),
        bounds=(1e-2, 1e2),
    )

    print(f"MLE s: {s_hat:.3f}, True s: {TRUE_S:.3f}")
    print(f"MLE var: {var_hat:.3f}, True var: {TRUE_VAR:.3f}")

    fw_prior_inferred = np.sqrt(var_hat) * kernel(freqs, length=s_hat)
    fw_prior_inferred_plot = np.sqrt(var_hat) * kernel(freqs_plot, length=s_hat)

    # binned
    range_X = [0, 50]
    n_bins = 30
    X_std, f_bin_edges, _ = binned_statistic(
        x=freqs.flatten(),
        values=X_prior.flatten(),
        range=range_X,
        statistic="std",
        bins=n_bins,
    )
    f_bin, _, _ = binned_statistic(
        x=freqs.flatten(),
        values=freqs.flatten(),
        range=range_X,
        statistic="mean",
        bins=n_bins,
    )

    fig, ax = plt.subplots(1, 1, figsize=[8, 6], dpi=150, layout="compressed")
    ax.scatter(freqs, X_prior, s=25, linewidths=0.25, edgecolors="white")
    ax.plot(f_bin, X_std, c="C4", label="Inferred PSD (binning)")
    ax.plot(freqs_plot, fw_prior_plot, c="C1", label="True PSD")
    ax.plot(
        freqs_plot, fw_prior_inferred_plot, c="C2", label="Inferred PSD (MLE)", ls="--", alpha=0.7
    )
    ax.set_xlim(-3, 153)
    ax.set_ylim(-1.4, 1.4)
    ax.set_ylabel("Fourier weight")
    ax.set_xlabel("Frequency")
    ax.legend(loc="upper right")
    plt.show()
