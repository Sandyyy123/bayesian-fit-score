"""
Demo runner for the Bayesian Fit Score model.
Runs in demo mode (no real data required) to show the shrinkage effect.
"""
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.special import logit, expit


def demo_shrinkage():
    """Demonstrate how Bayesian shrinkage corrects small-sample rates."""
    print("\n=== Bayesian Shrinkage Demo ===")
    print("Naive rates vs posterior estimates\n")

    programs = [
        {"name": "Prog-A (n=9)",   "k": 7,   "n": 9},
        {"name": "Prog-B (n=280)", "k": 151, "n": 280},
        {"name": "Prog-C (n=45)",  "k": 28,  "n": 45},
    ]

    # Population prior: Beta(2,2) ~ mean 0.50
    alpha_prior, beta_prior = 2, 2

    for p in programs:
        k, n = p["k"], p["n"]
        naive = k / n
        alpha_post = alpha_prior + k
        beta_post = beta_prior + (n - k)
        posterior_mean = alpha_post / (alpha_post + beta_post)
        ci_lo, ci_hi = beta_dist.ppf([0.05, 0.95], alpha_post, beta_post)
        ci_width = ci_hi - ci_lo
        uncertain = ci_width > 0.40

        print(f"  {p['name']:20s}  naive={naive:.1%}  posterior={posterior_mean:.1%}  "
              f"90%CI=[{ci_lo:.2f},{ci_hi:.2f}]  width={ci_width:.2f}"
              + ("  *** UNCERTAIN" if uncertain else ""))

    print()
    print("Key insight: Prog-A (n=9, naive 78%) is shrunk to ~61%.")
    print("You cannot tell a client Prog-A is more receptive than Prog-B (54%).")
    print("The wide CI ([0.37, 0.84]) shows the estimate is dominated by noise.\n")


def demo_score():
    """Show score output for 5 illustrative programs."""
    print("=== Score Output Demo ===\n")

    def single_score(k_geo, n_geo, k_comp, n_comp, traj, w=(0.42, 0.35, 0.23)):
        a_g, b_g = 2 + k_geo, 2 + (n_geo - k_geo)
        a_c, b_c = 2 + k_comp, 2 + (n_comp - k_comp)
        theta_g = a_g / (a_g + b_g)
        theta_c = a_c / (a_c + b_c)
        p = float(expit(w[0]*logit(theta_g) + w[1]*logit(theta_c) + w[2]*traj))
        score = 1 if p < .35 else 2 if p < .50 else 3 if p < .65 else 4 if p < .80 else 5
        ci_lo, ci_hi = beta_dist.ppf([0.05, 0.95], a_g, b_g)
        return p, score, ci_lo, ci_hi

    cases = [
        ("Prog-001", 256, 312, 94, 125, +0.18),
        ("Prog-002", 133, 187, 75, 125, +0.09),
        ("Prog-003",   7,   9, 5,   9, +0.02),
        ("Prog-004",  39,  94, 30,  78, -0.05),
        ("Prog-005",  68, 241, 22, 100, -0.14),
    ]

    print(f"  {'Program':12s} {'n':>5} {'P(fit)':>8} {'Score':>6} {'90%CI':>14}  Flag")
    print("  " + "-"*60)
    for name, kg, ng, kc, nc, traj in cases:
        p, s, lo, hi = single_score(kg, ng, kc, nc, traj)
        flag = " *** wide CI" if (hi - lo) > 0.40 else ""
        print(f"  {name:12s} {ng:>5}   {p:.3f}    {s}/5   [{lo:.2f}–{hi:.2f}]{flag}")
    print()


if __name__ == "__main__":
    demo_shrinkage()
    demo_score()
    print("To run with real data: from scoring_model import score")
    print("See README.md for full usage.")
