# RMWA: Robust Monitoring Wavelet Algorithm for SIVIEs

This repository contains the official Python implementation of the Robust Monitoring Wavelet Algorithm (RMWA) for solving Stochastic Itô-Volterra Integral Equations (SIVIEs) as described in the paper: 
> A Robust Wavelet–Krylov Method with Adaptive Stability Monitoring for Stochastic Itô–Volterra Integral Equations"

## 🚀 Overview
Numerical solutions of SIVIEs are often challenged by ill-conditioning at high resolutions and the presence of non-Gaussian (heavy-tailed) noise. RMWA addresses these issues through:
- **SKCW Discretization:** Second-Kind Chebyshev Wavelets with compact support.
- **Robust Solver:** Iteratively Reweighted LSQR (IR-LSQR) with Huber M-estimation.
- **Adaptive Monitoring:** A sensitivity index $S(M)$ that identifies the optimal truncation order $M^*$.

## 📁 Repository Structure
The repository is organized into core solvers, benchmarking suites, and application case studies:

| File | Description |
| :--- | :--- |
| `Code1_robust_skcw_lsqr_siviey.py` | The base framework for SKCW generation and the IR-LSQR robust solver. |
| `Updated_Code1_robust_skcw_lsqr_siviey.py` | **Primary Execution Script:** Implements Section 4 metrics (Conditioning regression, Sensitivity Index, and Monte Carlo Distribution). |
| `Comparision_Methods.py` | Benchmark suite comparing RMWA against Standard QR, Tikhonov, and Truncated SVD regularizations. |
| `Biological_Population_with_Cauchy_Environmental_Shocks.py` | Ecological case study involving memory effects and Cauchy-distributed environmental shocks. |
| `RMWA_Master_Results.csv` | Summarized statistical metrics ($\kappa, S(M), MSE$) for all levels. |
| `RMWA_Detailed_MSE_Dist.csv` | Raw 2000-path realization data used for error distribution rigor (Boxplots). |
| `Comprehensive_Comparison_Results.csv` | Detailed benchmarking data vs competitive solvers. |

## 🛠 Prerequisites
- **Language:** Python 3.12+
- **Key Libraries:** `NumPy`, `SciPy`, `Pandas`, `Matplotlib`, `multiprocessing`.
- **System Tested:** 12th Gen Intel Core i5-1235U / 16GB RAM.

## 🏃 How to Run
1. **Primary Analytics:** Execute `Updated_Code1_robust_skcw_lsqr_siviey.py` first to generate the statistical landscape of the resolution levels ($M=1$ to $9$). This will identify the optimal $M^*$ level for your problem.
2. **Comparative Benchmarking:** Run `Comparision_Methods.py` to evaluate the 16% accuracy gain of RMWA over Tikhonov and TSVD under stochastic outliers.
3. **Application Study:** After establishing the optimal $M^*$, run `Biological_Population_with_Cauchy_Environmental_Shocks.py` to see the algorithm's performance in a realistic biological restoration model.

## 📈 Performance Summary
- **Reproducibility:** Locked GLOBAL_SEED = 42 ensured across all 18,000 realization solves.
- **Complexity:** Exploits operator separation to solve 2000 paths in ~677s (at M=9) on modern i5 hardware.

## License
MIT License. Feel free to use and cite our work.
