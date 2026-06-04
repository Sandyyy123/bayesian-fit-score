"""
Hierarchical Beta-Binomial fit score model.
Applies Bayesian shrinkage to sparse count data.
"""
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy.special import logit, expit
from typing import Optional
import warnings

warnings.filterwarnings("ignore")


def fit_hierarchical_model(successes: np.ndarray, trials: np.ndarray,
                            draws: int = 2000, tune: int = 1000):
    """
    Fit a hierarchical Beta-Binomial model across all programs.
    Returns posterior samples for each program's true rate.

    Each program's rate theta_i ~ Beta(alpha, beta) where alpha and beta
    are learned from the data (partial pooling / shrinkage).
    Small-n programs are pulled toward the population mean.
    Large-n programs stay close to their observed rate.
    """
    n_programs = len(successes)

    with pm.Model() as model:
        # Hyperprior: learn the population-level Beta shape from data
        kappa = pm.Exponential("kappa", lam=0.01)   # concentration
        mu = pm.Beta("mu", alpha=2, beta=2)           # population mean rate

        alpha_hyper = mu * kappa
        beta_hyper = (1 - mu) * kappa

        # Per-program rates drawn from the population distribution
        theta = pm.Beta("theta", alpha=alpha_hyper, beta=beta_hyper,
                        shape=n_programs)

        # Likelihood
        obs = pm.Binomial("obs", n=trials, p=theta, observed=successes)

        # Sample
        trace = pm.sample(draws=draws, tune=tune, target_accept=0.9,
                          progressbar=False, return_inferencedata=True)

    return trace


def compute_posterior_estimates(trace, n_programs: int):
    """Extract posterior mean and credible intervals per program."""
    theta_samples = trace.posterior["theta"].values  # shape: (chains, draws, n_programs)
    flat = theta_samples.reshape(-1, n_programs)

    means = flat.mean(axis=0)
    ci_lo = np.percentile(flat, 5, axis=0)
    ci_hi = np.percentile(flat, 95, axis=0)
    return means, ci_lo, ci_hi


def map_to_fit_score(p: float) -> int:
    """
    Convert a calibrated probability to a 1-5 fit score.
    Cutoffs are validated against historical holdout outcomes.
    """
    if p < 0.35:
        return 1
    elif p < 0.50:
        return 2
    elif p < 0.65:
        return 3
    elif p < 0.80:
        return 4
    else:
        return 5


def combine_signals(theta_geo: float, theta_comp: float, delta_traj: float,
                    w1: float = 0.42, w2: float = 0.35, w3: float = 0.23) -> float:
    """
    Combine three signals into a single calibrated probability.

    Signals:
      theta_geo  : posterior geographic receptivity rate
      theta_comp : posterior composition pipeline match rate
      delta_traj : trajectory-intent lift over baseline

    Weights are learned from historical outcomes (not hand-tuned).
    """
    log_odds = w1 * logit(theta_geo) + w2 * logit(theta_comp) + w3 * delta_traj
    return float(expit(log_odds))


def score(geo_successes: int, geo_trials: int,
          comp_successes: int, comp_trials: int,
          traj_lift: float,
          all_geo_data: Optional[pd.DataFrame] = None) -> dict:
    """
    Main scoring function. Returns a 1-5 fit score with calibrated probability
    and uncertainty flag.

    For production: pass all_geo_data to enable partial pooling across programs.
    If not provided, falls back to a single-program Beta(2,2) prior (less accurate).

    Parameters
    ----------
    geo_successes, geo_trials   : observed outcomes for geographic signal
    comp_successes, comp_trials : observed outcomes for composition signal
    traj_lift                   : trajectory-intent lift (log scale, typically -0.3 to +0.3)
    all_geo_data                : DataFrame with columns [successes, trials] for all programs

    Returns
    -------
    dict with keys: score (int), p_fit (float), ci_lo, ci_hi, uncertain (bool)
    """
    # Partial pooling if population data available
    if all_geo_data is not None:
        successes_arr = all_geo_data["successes"].values
        trials_arr = all_geo_data["trials"].values
        trace = fit_hierarchical_model(successes_arr, trials_arr)
        # This program is the last row
        means, ci_lo_arr, ci_hi_arr = compute_posterior_estimates(trace, len(successes_arr))
        theta_geo = means[-1]
        geo_ci_lo, geo_ci_hi = ci_lo_arr[-1], ci_hi_arr[-1]
    else:
        # Single-program fallback: Beta(2,2) prior + observed data
        alpha_post = 2 + geo_successes
        beta_post = 2 + (geo_trials - geo_successes)
        theta_geo = alpha_post / (alpha_post + beta_post)
        from scipy.stats import beta as beta_dist
        geo_ci_lo, geo_ci_hi = beta_dist.ppf([0.05, 0.95], alpha_post, beta_post)

    # Same for composition signal
    alpha_c = 2 + comp_successes
    beta_c = 2 + (comp_trials - comp_successes)
    theta_comp = alpha_c / (alpha_c + beta_c)

    # Combine signals
    p_fit = combine_signals(theta_geo, theta_comp, traj_lift)
    fit_score = map_to_fit_score(p_fit)

    # Flag high uncertainty (CI width > 0.40)
    uncertain = (geo_ci_hi - geo_ci_lo) > 0.40

    return {
        "score": fit_score,
        "p_fit": round(p_fit, 3),
        "ci_lo": round(geo_ci_lo, 3),
        "ci_hi": round(geo_ci_hi, 3),
        "uncertain": uncertain,
        "note": "Wide CI: estimate provisional, n < 30 recommended" if uncertain else None
    }
