import jax
import jax.numpy as jnp


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
        model.line.A.prior_logpdf().sum()
        + model.line.λ0.prior_logpdf().sum()
        + model.line.σ.prior_logpdf().sum()
    )
    ln_hyperprior = (
        jax.scipy.stats.expon.logpdf(x=model.line.A.kernel.variance.val, scale=0.0005).sum()
        + jax.scipy.stats.expon.logpdf(x=model.line.λ0.kernel.variance.val, scale=0.0005).sum()
        + jax.scipy.stats.expon.logpdf(x=model.line.σ.kernel.variance.val, scale=0.0005).sum()
        # + jax.scipy.stats.norm.logpdf(
        #     x=jnp.log(model.line.λ0.kernel.length_scale.val),
        #     loc=jnp.log(1.0),
        #     scale=0.001,
        # ).sum()
    )
    return -1 * (ln_like + ln_prior + ln_hyperprior)
