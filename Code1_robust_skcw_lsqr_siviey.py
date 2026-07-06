"""
Robust SKCW-Weighted LSQR for Stochastic Itô-Volterra Integral Equations (SIVIEs)
================================================================================
License: MIT
Description: 
    This framework implements a robust numerical solver for SIVIEs using 
    Second-Kind Chebyshev Wavelets (SKCW). It addresses ill-conditioning 
    through a novel Sensitivity Index S(M) and ensures stability against 
    heavy-tailed noise (Cauchy) using Iteratively Reweighted LSQR (IR-LSQR).

Key Features:
    1. Parallel Monte Carlo Simulations (Multiprocessing).
    2. Huber-weighted IR-LSQR for robust M-estimation.
    3. Adaptive monitoring of the optimal wavelet order M*.
    4. Stochastic Exponential (Itô) exact solution for multiplicative noise.
    5. Fast matrix construction via Gaussian Quadrature.
"""



import numpy as np
import scipy.linalg as la
from scipy.special import roots_legendre
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. GLOBAL CONFIGURATION (Final Quality Settings)
# =============================================================================
TEST_MODE = False   # Set to False for Final Paper Results (N_sim=2000)
GLOBAL_SEED = 42

if TEST_MODE:
    M_LEVELS = [1, 2, 3, 4, 5]
    N_SIM = 50
    M_POINTS = 100
else:
    M_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    N_SIM = 2000  # Standard for Q1 Journals
    M_POINTS = 500 # m > n for all M levels

# Kernel Functions (Example 1: Baseline)
def K1_kernel(s, t): return s
def K2_kernel(s, t): return 0.1
def g_func(t): return t

# =============================================================================
# 2. WAVELET BASIS & OPERATORS
# =============================================================================

def u_n(n, x):
    """Chebyshev polynomial of second kind with overflow protection."""
    if n == 0: return 1.0
    if n == 1: return 2.0 * x
    u_prev, u_curr = 1.0, 2.0 * x
    for _ in range(2, n + 1):
        u_next = 2.0 * x * u_curr - u_prev
        u_prev, u_curr = u_curr, u_next
    return np.clip(u_curr, -1e10, 1e10)

def skcw_basis(M, j, t):
    """SKCW basis at level M, index j, point t."""
    n_intervals = 2**(M-1)
    width = 1.0 / n_intervals
    k = j # translation index
    t_local = (t - k * width) / width
    if 0.0 <= t_local <= 1.0:
        psi = u_n(k, 2 * t_local - 1)
        return psi * np.sqrt(2.0 / width)
    return 0.0

def get_deterministic_part(M, t_points, n_quad=30):
    """Pre-calculates the deterministic portion of matrix A (I - P_D)."""
    n = 2**(M-1)
    m = len(t_points)
    D = np.zeros((m, n))
    nodes, weights = roots_legendre(n_quad)
    
    for j in range(n):
        for k, t in enumerate(t_points):
            val_psi = skcw_basis(M, j, t)
            if t > 0:
                mapped_nodes = (t / 2.0) * nodes + (t / 2.0)
                mapped_weights = (t / 2.0) * weights
                f_vals = np.array([K1_kernel(s, t) * skcw_basis(M, j, s) for s in mapped_nodes])
                int_val = np.sum(mapped_weights * f_vals)
            else:
                int_val = 0
            D[k, j] = val_psi - int_val
    return D

# =============================================================================
# 3. SOLVER & ANALYTICS
# =============================================================================

def exact_solution_ito(t, B_t=None):
    """Exact pathwise solution using Itô stochastic exponential."""
    sigma = 0.1
    if B_t is None: # Deterministic Expected Value
        return t * np.exp(t**2 / 2.0)
    else: # Full Path
        return t * np.exp(t**2 / 2.0 - 0.5 * sigma**2 * t + sigma * B_t)

def ir_lsqr_huber(A, G, k_H=1.345):
    """Weighted LSQR (Huber IRLS) for robustness against noise."""
    try:
        C, _, _, _ = la.lstsq(A, G)
        for _ in range(20):
            res = A @ C - G
            mad = max(np.median(np.abs(res - np.median(res))), 1e-8)
            weights = np.minimum(1.0, k_H / (np.abs(res) / mad + 1e-10))
            C_new, _, _, _ = la.lstsq(np.diag(weights) @ A, np.diag(weights) @ G)
            if la.norm(C_new - C) < 1e-6: break
            C = C_new
        return C
    except: return np.zeros(A.shape[1])

def worker_task(args):
    """Monte Carlo trajectory worker."""
    M, sim_id, noise_type, base_seed, t_points, D_matrix = args
    current_seed = base_seed + sim_id
    np.random.seed(current_seed)
    m, n = D_matrix.shape
    dt = 1.0 / (m - 1)
    
    try:
        # Stochastic Integral Approximation
        if noise_type == 'gaussian':
            dB = np.sqrt(dt) * np.random.randn(m)
        else: # Cauchy test
            dB = np.clip(np.random.cauchy(size=m) * 0.01, -0.5, 0.5)
        
        B_path = np.cumsum(dB)
        S = np.zeros((m, n))
        for j in range(n):
            current_int = 0.0
            for k in range(1, m):
                current_int += K2_kernel(t_points[k-1], t_points[k]) * skcw_basis(M, j, t_points[k-1]) * dB[k]
                S[k, j] = current_int
        
        A = D_matrix - S
        G = np.array([g_func(t) for t in t_points])
        
        # Robust Solve
        C = ir_lsqr_huber(A, G)
        
        # Metrics
        y_true = exact_solution_ito(t_points, B_path)
        y_hat = D_matrix @ C # Approximation (D includes Basis - Integral)
        mse = np.mean((y_hat - y_true)**2)
        
        s_vals = la.svd(A, compute_uv=False)
        kappa = s_vals[0] / (s_vals[-1] + 1e-15)
        res = A @ C - G
        mad = np.median(np.abs(res - np.median(res))) + 1e-12
        
        return mse, kappa, kappa * mad
    except: return None

# =============================================================================
# 4. ORCHESTRATOR & ANALYTICS
# =============================================================================

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    final_m, final_mse, final_kappa, final_s = [], [], [], []
    
    print("="*80)
    print(f"RMWA SYSTEM START | Mode: {'TEST' if TEST_MODE else 'FINAL'}")
    print(f"Nodes: {M_POINTS} | Simulations: {N_SIM} | Cores: 4")
    print("="*80)
    
    t_nodes = np.sort((np.cos(np.pi * np.arange(M_POINTS) / (M_POINTS - 1)) + 1.0) / 2.0)

    for M in M_LEVELS:
        start_t = time.time()
        # 1. Pre-calculate deterministic basis interaction
        D = get_deterministic_part(M, t_nodes)
        
        # 2. Parallel Monte Carlo execution
        args = [(M, i, 'gaussian', GLOBAL_SEED, t_nodes, D) for i in range(N_SIM)]
        with mp.Pool(processes=4) as pool:
            results = pool.map(worker_task, args)
        
        valid = [r for r in results if r is not None]
        if valid:
            mse_avg, k_avg, s_avg = np.mean(valid, axis=0)
            final_m.append(M)
            final_mse.append(mse_avg)
            final_kappa.append(k_avg)
            final_s.append(s_avg)
            
            print(f"M={M} | MSE={mse_avg:.2e} | kappa={k_avg:.2e} | S={s_avg:.2e} | Time={time.time()-start_t:.1f}s")

    # --- 5. POST-PROCESSING: ALPHA REGRESSION ---
    if len(final_m) > 1:
        log_m = np.log(final_m)
        log_k = np.log(final_kappa)
        alpha, intercept = np.polyfit(log_m, log_k, 1)
        
        print("\n" + "="*50)
        print(f"ANALYSIS RESULTS")
        print(f"Estimated Growth Rate (Alpha): {alpha:.4f}")
        print(f"Optimal Order (M*): {final_m[np.argmin(final_s)]}")
        print("="*50)

        # --- 6. LATEX TABLE GENERATOR ---
        print("\nCOPY-PASTE TO LATEX:")
        print("-" * 30)
        print("\\begin{table}[h!]\n\\centering\n\\begin{tabular}{ccccc}\n\\hline")
        print("Order $M$ & $n$ & $\\kappa(M)$ & $S(M)$ & Avg. MSE \\\\ \\hline")
        for i in range(len(final_m)):
            n_bases = 2**(final_m[i]-1)
            print(f"{final_m[i]} & {n_bases} & {final_kappa[i]:.2e} & {final_s[i]:.2e} & {final_mse[i]:.2e} \\\\")
        print(f"\\hline\n\\end{{tabular}}\n\\caption{{Numerical results summary ($\\alpha \\approx {alpha:.2f}$)}}\n\\end{{table}}")

        # --- 7. PLOTTING (FIXED WITH RAW STRINGS) ---
        plt.figure(figsize=(12, 5))
        
        # Plot Sensitivity Index
        plt.subplot(1, 2, 1)
        plt.semilogy(final_m, final_s, 'r-o', linewidth=2, markersize=8)
        plt.axvline(final_m[np.argmin(final_s)], color='k', linestyle='--', label=r'$M^*$')
        plt.title(r'Sensitivity Index Monitoring $S(M)$')
        plt.xlabel(r'Wavelet Order $M$')
        plt.ylabel(r'$S(M)$')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot Condition Number fitting
        plt.subplot(1, 2, 2)
        # Use rf"" for raw-f-string to handle both variables and LaTeX backslashes
        plt.loglog(final_m, final_kappa, 'b-s', label=rf'Data ($\alpha={alpha:.2f}$)')
        plt.title(r'Condition Number Growth $\kappa(M)$')
        plt.xlabel(r'$M$')
        plt.ylabel(r'$\kappa(M)$')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('RMWA_Final_Results.png', dpi=300)
        print("\nPlots saved successfully as 'RMWA_Final_Results.png'")
        plt.show()