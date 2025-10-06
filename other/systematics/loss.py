import jax
import jax.numpy as jnp

SCALE = 1


def neg_ln_posterior(model, λ, xy_data, data, u_data, mask):
    # Model predictions
    pred = jax.vmap(model, in_axes=(0, None))(λ, xy_data)
    # Likelihood
    ln_like = jnp.sum(
        jnp.where(
            mask,
            jax.scipy.stats.norm.logpdf(x=pred, loc=data, scale=u_data),
            0.0,
        )
    )
    ln_prior = (
        model.line.A_raw.prior_logpdf()
        + model.line.v.prior_logpdf()
        + model.line.vσ_raw.prior_logpdf()
        # + model.line.σ_raw.prior_logpdf().sum()
    )
    ln_hyperprior = (
        jax.scipy.stats.expon.logpdf(x=model.line.A_raw.kernel.variance.val, scale=SCALE).sum()
        + jax.scipy.stats.expon.logpdf(x=model.line.v.kernel.variance.val, scale=SCALE).sum()
        # + jax.scipy.stats.expon.logpdf(x=model.line.σ_raw.kernel.variance.val, scale=SCALE).sum()
        + jax.scipy.stats.expon.logpdf(x=model.line.vσ_raw.kernel.variance.val, scale=SCALE).sum()
    )
    return -1 * (ln_like + ln_prior)
    # return -1 * (ln_like + ln_prior + ln_hyperprior)
