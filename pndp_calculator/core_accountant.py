import math
from typing import Tuple
import numpy as np
from scipy import special, integrate
from scipy.optimize import root_scalar
from scipy.stats import norm


def calculate_optimal_sigma_rdp(
    epsilon: float,
    delta: float,
    effective_variance_multiplier: float,
) -> Tuple[float, float]:
    def rdp_at_alpha(alpha: float, noise_multiplier: float) -> float:
        return alpha / (2.0 * (noise_multiplier ** 2))

    def eps_from_noise_multiplier(noise_multiplier: float) -> Tuple[float, float]:
        alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        best_eps = float("inf")
        best_alpha = float("nan")
        for alpha in alphas:
            if alpha <= 1.0:
                continue
            rdp = rdp_at_alpha(alpha, noise_multiplier)
            eps = (
                rdp
                - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
                + math.log((alpha - 1.0) / alpha)
            )
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha
        return best_eps, best_alpha

    nm_low = 0.0
    nm_high = 1.0
    eps_high, _ = eps_from_noise_multiplier(nm_high)
    max_nm = 1e6
    while eps_high > epsilon:
        nm_high *= 2.0
        if nm_high > max_nm:
            raise ValueError("Target privacy budget is too strict.")
        eps_high, _ = eps_from_noise_multiplier(nm_high)

    tol = 1e-4
    best_alpha = float("nan")
    while epsilon - eps_high > tol:
        nm_mid = (nm_low + nm_high) / 2.0
        eps_mid, alpha_mid = eps_from_noise_multiplier(nm_mid)
        if eps_mid <= epsilon:
            nm_high = nm_mid
            eps_high = eps_mid
            best_alpha = alpha_mid
        else:
            nm_low = nm_mid

    if math.isnan(best_alpha):
        _, best_alpha = eps_from_noise_multiplier(nm_high)

    noise_multiplier_opacus = nm_high * math.sqrt(effective_variance_multiplier)
    return float(noise_multiplier_opacus), float(best_alpha)


def calculate_optimal_sigma_gdp(
    epsilon: float,
    delta: float,
    effective_variance_multiplier: float,
) -> Tuple[float, float]:
    def mu_from_noise_multiplier(noise_multiplier: float) -> float:
        return 1.0 / noise_multiplier

    def eps_from_noise_multiplier(noise_multiplier: float) -> Tuple[float, float]:
        mu = mu_from_noise_multiplier(noise_multiplier)

        def f(eps: float) -> float:
            term1 = norm.cdf(-eps / mu + mu / 2.0)
            term2 = math.exp(eps) * norm.cdf(-eps / mu - mu / 2.0)
            return term1 - term2 - delta

        root_res = root_scalar(f, bracket=[0.0, 500.0], method="brentq")
        return float(root_res.root), mu

    nm_low = 0.0
    nm_high = 1.0
    eps_high, _ = eps_from_noise_multiplier(nm_high)
    max_nm = 1e6
    while eps_high > epsilon:
        nm_high *= 2.0
        if nm_high > max_nm:
            raise ValueError("Target privacy budget is too strict.")
        eps_high, _ = eps_from_noise_multiplier(nm_high)

    tol = 1e-4
    best_mu = float("nan")
    while epsilon - eps_high > tol:
        nm_mid = (nm_low + nm_high) / 2.0
        eps_mid, mu_mid = eps_from_noise_multiplier(nm_mid)
        if eps_mid <= epsilon:
            nm_high = nm_mid
            eps_high = eps_mid
            best_mu = mu_mid
        else:
            nm_low = nm_mid

    if math.isnan(best_mu):
        _, best_mu = eps_from_noise_multiplier(nm_high)

    noise_multiplier_opacus = nm_high * math.sqrt(effective_variance_multiplier)
    return float(noise_multiplier_opacus), float(best_mu)


def compute_exact_rdp_step(q: float, sigma_norm: float, alpha: float) -> float:
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return alpha / (2.0 * sigma_norm**2)
        
    if float(alpha).is_integer():
        alpha_int = int(alpha)
        log_a = -np.inf
        for i in range(alpha_int + 1):
            log_coef = (
                math.log(special.binom(alpha_int, i)) 
                + i * math.log(q) 
                + (alpha_int - i) * math.log(1.0 - q)
            )
            s = log_coef + (i * i - i) / (2.0 * sigma_norm**2)
            log_a = np.logaddexp(log_a, s)
        return float(log_a / (alpha - 1.0))
    else:
        def integrand(z):
            p_z = np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi)
            ratio = np.exp(z / sigma_norm - 0.5 / sigma_norm**2)
            term = (1.0 - q + q * ratio)**alpha
            return p_z * term
        a_alpha, _ = integrate.quad(integrand, -15, 15, epsabs=1e-6, epsrel=1e-6)
        if a_alpha <= 0:
            return float("inf")
        return float(math.log(a_alpha) / (alpha - 1.0))


def calculate_optimal_sigma_rdp_sampled(
    epsilon: float,
    delta: float,
    delta_sens: float,
    total_steps: int,
    q: float,
) -> Tuple[float, float]:
    def _compute_rdp_step(sigma_norm: float, alpha: float) -> float:
        if q == 0.0:
            return 0.0
        if q == 1.0:
            return alpha / (2.0 * sigma_norm**2)
            
        if float(alpha).is_integer():
            alpha_int = int(alpha)
            log_a = -np.inf
            for i in range(alpha_int + 1):
                log_coef = (
                    math.log(special.binom(alpha_int, i)) 
                    + i * math.log(q) 
                    + (alpha_int - i) * math.log(1.0 - q)
                )
                s = log_coef + (i * i - i) / (2.0 * sigma_norm**2)
                log_a = np.logaddexp(log_a, s)
            return float(log_a / (alpha - 1.0))
        else:
            def integrand(z):
                p_z = np.exp(-z**2 / 2.0) / np.sqrt(2.0 * np.pi)
                ratio = np.exp(z / sigma_norm - 0.5 / sigma_norm**2)
                term = (1.0 - q + q * ratio)**alpha
                return p_z * term
            a_alpha, _ = integrate.quad(integrand, -15, 15, epsabs=1e-6, epsrel=1e-6)
            if a_alpha <= 0:
                return float("inf")
            return float(math.log(a_alpha) / (alpha - 1.0))

    def eps_from_noise_multiplier(noise_multiplier: float) -> Tuple[float, float]:
        alphas = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        best_eps = float("inf")
        best_alpha = float("nan")
        
        for alpha in alphas:
            if alpha <= 1.0:
                continue
            rdp_total = total_steps * _compute_rdp_step(noise_multiplier, alpha)
            eps = (
                rdp_total
                - (math.log(delta) + math.log(alpha)) / (alpha - 1.0)
                + math.log((alpha - 1.0) / alpha)
            )
            if eps < best_eps:
                best_eps = eps
                best_alpha = alpha
        return best_eps, best_alpha

    nm_low = 0.001
    nm_high = 1.0
    eps_high, _ = eps_from_noise_multiplier(nm_high)
    max_nm = 1e6
    while eps_high > epsilon:
        nm_high *= 2.0
        if nm_high > max_nm:
            raise ValueError("Target privacy budget is too strict.")
        eps_high, _ = eps_from_noise_multiplier(nm_high)

    tol = 1e-4
    best_alpha = float("nan")
    while nm_high - nm_low > tol:
        nm_mid = (nm_low + nm_high) / 2.0
        eps_mid, alpha_mid = eps_from_noise_multiplier(nm_mid)
        if eps_mid <= epsilon:
            nm_high = nm_mid
            eps_high = eps_mid
            best_alpha = alpha_mid
        else:
            nm_low = nm_mid

    if math.isnan(best_alpha):
        _, best_alpha = eps_from_noise_multiplier(nm_high)

    sigma = nm_high * delta_sens
    return float(sigma), float(best_alpha)
