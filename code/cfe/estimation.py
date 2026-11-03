"""Two-stage least squares with absorbed fixed effects, and cluster-robust standard errors from influence values.

The point estimates and standard errors match pyfixest (feols with CRV1 clustering, default small-sample
corrections) to the reported precision.
"""
import numpy as np
import pandas as pd


def demean(V: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Subtract group means from each column of V; grp holds integer group codes 0..G-1."""
    G = grp.max() + 1
    cnt = np.bincount(grp, minlength=G)
    M = np.stack([np.bincount(grp, weights=V[:, j], minlength=G) for j in range(V.shape[1])], 1) / cnt[:, None]
    return V - M[grp]


def tsls(Y: np.ndarray, dvar: np.ndarray, z: np.ndarray, fe, W: np.ndarray | None = None
         ) -> tuple[np.ndarray, np.ndarray]:
    """2SLS of each column of Y on dvar, instrumented by z, absorbing fixed effects fe and partialling out controls W
    (OLS when z is dvar).

    Returns the coefficients and each observation's influence value; summing influence values within a cluster
    gives that cluster's contribution to the sampling error. The influence values include the small-sample factor
    (N − 1) / (N − K), where K counts the absorbed cells, as pyfixest does (singleton cells are dropped from N and K).
    """
    m = Y.shape[1]
    grp = pd.factorize(fe)[0]
    parts = [Y, dvar, z] + ([W] if W is not None and W.shape[1] else [])
    V = demean(np.column_stack(parts).astype(float), grp)
    A, Wt = V[:, : m + 2], V[:, m + 2:]
    k_w = 0
    if Wt.shape[1]:
        A = A - Wt @ np.linalg.lstsq(Wt, A, rcond=None)[0]
        k_w = np.linalg.matrix_rank(Wt)
    Yt, dt, zt = A[:, :m], A[:, m], A[:, m + 1]
    den = zt @ dt
    b = zt @ Yt / den
    cnt = np.bincount(grp)
    n_eff, k = (cnt[grp] > 1).sum(), (cnt > 1).sum() + k_w + 1
    return b, zt[:, None] * (Yt - dt[:, None] * b) / den * np.sqrt((n_eff - 1) / (n_eff - k))


def cluster_se(psi: np.ndarray, cl) -> np.ndarray:
    """Cluster-robust standard errors from influence values psi (n x m), with the G / (G − 1) correction."""
    cl = pd.factorize(cl)[0]
    G = cl.max() + 1
    S = np.stack([np.bincount(cl, weights=psi[:, j], minlength=G) for j in range(psi.shape[1])], 1)
    return np.sqrt(G / (G - 1) * (S ** 2).sum(0))
