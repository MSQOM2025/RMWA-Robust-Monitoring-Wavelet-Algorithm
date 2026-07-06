"""
Robust Monitoring Wavelet Algorithm (RMWA) - Publication Benchmarking
====================================================================
Solvers included: RMWA (Proposed), Standard QR, Tikhonov, Truncated SVD.
Optimized for: Multi-solver comparison with full reproducibility.
"""

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.special import roots_legendre
import multiprocessing as mp
import time
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 1. FINAL SETTINGS & TEST SWITCH
# =============================================================================
TEST_MODE =False     # برای اجرای 2000 تایی نهایی، این را False کنید
GLOBAL_SEED = 42

if TEST_MODE:
    M_LEVELS = [1, 2, 3, 4]
    N_SIM = 20        # تست سریع
    M_POINTS = 100
else:
    M_LEVELS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    N_SIM = 2000      # مقدار مورد نیاز برای مقاله Q1
    M_POINTS = 500    # تراکم بالای نقاط

# Benchmark Kernels
def K1_kernel(s, t): return s
def K2_kernel(s, t): return 0.1
def g_func(t): return t

# =============================================================================
# 2. COMPETITOR SOLVERS
# =============================================================================

def solve_tikhonov(A, G, lmbda=0.01):
    m, n = A.shape
    return la.solve(A.T @ A + lmbda**2 * np.eye(n), A.T @ G)

def solve_tsvd(A, G, threshold=0.01):
    u, s, vh = la.svd(A, full_matrices=False)
    s_inv = np.zeros_like(s)
    mask = s > (threshold * np.max(s))
    s_inv[mask] = 1.0 / s[mask]
    return vh.T @ (s_inv * (u.T @ G))

def ir_lsqr_huber(A, G, k_H=1.345):
    try:
        C, _, _, _ = la.lstsq(A, G)
        for _ in range(15):
            res = A @ C - G
            mad = max(np.median(np.abs(res - np.median(res))), 1e-8)
            W_vec = np.minimum(1.0, k_H / (np.abs(res)/mad + 1e-10))
            C_new, _, _, _ = la.lstsq(np.diag(W_vec) @ A, np.diag(W_vec) @ G)
            if la.norm(C_new - C) < 1e-6: break
            C = C_new
        return C
    except: return np.zeros(A.shape[1])

# =============================================================================
# 3. MATHEMATICAL BACKBONE (Pre-calculated)
# =============================================================================

def skcw_basis(M, j, t):
    n = 2**(M-1); width = 1.0 / n
    if (j * width) <= t <= ((j + 1) * width):
        return np.sqrt(n)
    return 0.0

def precalculate_D(M, t_pts, n_quad=40):
    n_basis = 2**(M-1); m = len(t_pts)
    D = np.zeros((m, n_basis))
    nodes, weights = roots_legendre(n_quad)
    for j in range(n_basis):
        w = 1.0/n_basis
        for k, t in enumerate(t_pts):
            val_psi = skcw_basis(M, j, t)
            s_i, e_i = j*w, min(t, (j+1)*w)
            if e_i > s_i:
                t_sub = e_i - s_i
                m_nodes = (t_sub/2)*nodes + (s_int+e_int)/2 if 's_int' in locals() else (t_sub/2)*nodes + (s_i + e_i)/2
                m_weights = (t_sub/2)*weights
                int_v = np.sum(m_weights * (m_nodes * np.sqrt(n_basis))) # Kernel K1=s
                D[k, j] = val_psi - int_v
            else: D[k, j] = val_psi
    return D

# =============================================================================
# 4. MASTER WORKER
# =============================================================================

def master_worker(args):
    M, path_id, t_pts, D_M = args
    local_seed = GLOBAL_SEED + (M * 10000) + path_id
    np.random.seed(local_seed)
    m, n = D_M.shape; dt = 1.0 / (m - 1)
    
    try:
        # Standard Brownian paths for benchmarking
        dB = np.sqrt(dt) * np.random.randn(m); B_p = np.cumsum(dB)
        S = np.zeros((m, n))
        for j in range(n):
            mask = (t_pts >= j*(1/n)) & (t_pts <= (j+1)*(1/n))
            S[:, j] = np.where(mask, np.cumsum(0.1 * np.sqrt(n) * dB), 0)

        A, G = D_M - S, t_pts
        y_true = t_pts * np.exp(t_pts**2/2 - 0.005*t_pts + 0.1*B_p)

        # Solving using 4 methods
        c_rmwa = ir_lsqr_huber(A, G)
        c_qr   = la.lstsq(A, G)[0]
        c_tikh = solve_tikhonov(A, G)
        c_tsvd = solve_tsvd(A, G)

        def get_mse(C): return np.mean(((D_M @ C) - y_true)**2)
        
        # Stability index based on RMWA (Proposed)
        s_vals = la.svd(A, compute_uv=False)
        kappa = s_vals[0]/(s_vals[-1] + 1e-15)
        
        return get_mse(c_rmwa), get_mse(c_qr), get_mse(c_tikh), get_mse(c_tsvd), kappa
    except: return None

# =============================================================================
# 5. EXECUTION & AUTO-REPORTING
# =============================================================================

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    t_nodes = np.sort((np.cos(np.pi * np.arange(M_POINTS) / (M_POINTS - 1)) + 1.0) / 2.0)
    
    print("="*90)
    print(f"RMWA CONSOLIDATED ENGINE | MODE: {'TEST' if TEST_MODE else 'FINAL'}")
    print(f"Total Trajectories: {len(M_LEVELS) * N_SIM}")
    print("="*90)

    final_results = []
    for M in M_LEVELS:
        t_start = time.time()
        D_M = precalculate_D(M, t_nodes)
        
        task_args = [(M, i, t_nodes, D_M) for i in range(N_SIM)]
        with mp.Pool(4) as pool:
            raw_res = pool.map(master_worker, task_args)
        
        valid = [r for r in raw_res if r is not None]
        if valid:
            m_mse_rmwa, m_mse_qr, m_mse_tikh, m_mse_tsvd, m_kappa = np.mean(valid, axis=0)
            s_index = m_kappa * np.sqrt(m_mse_rmwa) # Adjusted Sensitivity for Multi-Solver table
            
            final_results.append([M, 2**(M-1), m_kappa, m_mse_rmwa, m_mse_qr, m_mse_tikh, m_mse_tsvd])
            print(f"Order M={M} done. RMWA MSE: {m_mse_rmwa:.2e} | Time: {time.time()-t_start:.1f}s")

    # DATA PRESENTATION
    cols = ['M', 'n', 'Kappa', 'MSE_RMWA', 'MSE_QR', 'MSE_Tikhonov', 'MSE_TSVD']
    df = pd.DataFrame(final_results, columns=cols)
    
    print("\n" + "="*95 + "\nCOMPARATIVE PERFORMANCE REPORT\n" + "="*95)
    print(df.to_string(index=False, justify='center', float_format="%.2e"))
    df.to_csv('Comprehensive_Comparison_Results.csv', index=False)
    
    # LATEX GENERATOR
    print("\n" + "-"*30 + "\nLaTeX Code for Final Paper:\n" + "-"*30)
    print(df.to_latex(index=False, float_format="%.2e", caption="Comparative results across different solvers"))