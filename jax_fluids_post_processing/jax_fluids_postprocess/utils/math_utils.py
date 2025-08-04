"""
Mathematical utilities for fluid flow analysis.
"""

from typing import Tuple
import numpy as np


def compute_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute vorticity components from velocity field.
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Vorticity components (omega_x, omega_y, omega_z)
    """
    dx, dy, dz = spacing
    
    # Compute velocity gradients
    du_dy = np.gradient(u, dy, axis=1)
    du_dz = np.gradient(u, dz, axis=2)
    
    dv_dx = np.gradient(v, dx, axis=0)
    dv_dz = np.gradient(v, dz, axis=2)
    
    dw_dx = np.gradient(w, dx, axis=0)
    dw_dy = np.gradient(w, dy, axis=1)
    
    # Compute vorticity components
    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    
    return omega_x, omega_y, omega_z


def compute_vorticity_magnitude(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute vorticity magnitude.
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Vorticity magnitude
    """
    omega_x, omega_y, omega_z = compute_vorticity(u, v, w, spacing)
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)


def compute_q_criterion(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute Q-criterion for vortex identification.
    
    Q = 0.5 * (||Ω||² - ||S||²)
    where Ω is the vorticity tensor and S is the strain rate tensor.
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Q-criterion field
    """
    dx, dy, dz = spacing
    
    # Compute velocity gradients
    du_dx = np.gradient(u, dx, axis=0)
    du_dy = np.gradient(u, dy, axis=1)
    du_dz = np.gradient(u, dz, axis=2)
    
    dv_dx = np.gradient(v, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dv_dz = np.gradient(v, dz, axis=2)
    
    dw_dx = np.gradient(w, dx, axis=0)
    dw_dy = np.gradient(w, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)
    
    # Strain rate tensor (symmetric part)
    S11 = du_dx
    S22 = dv_dy
    S33 = dw_dz
    S12 = S21 = 0.5 * (du_dy + dv_dx)
    S13 = S31 = 0.5 * (du_dz + dw_dx)
    S23 = S32 = 0.5 * (dv_dz + dw_dy)
    
    # Vorticity tensor (antisymmetric part)
    O12 = O21 = 0.5 * (du_dy - dv_dx)
    O13 = O31 = 0.5 * (du_dz - dw_dx)
    O23 = O32 = 0.5 * (dv_dz - dw_dy)
    
    # Frobenius norms
    S_norm_sq = 2 * (S11**2 + S22**2 + S33**2 + 2 * (S12**2 + S13**2 + S23**2))
    O_norm_sq = 2 * (O12**2 + O13**2 + O23**2)
    
    # Q-criterion
    Q = 0.5 * (O_norm_sq - S_norm_sq)
    
    return Q


def compute_lambda2_criterion(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute Lambda2 criterion for vortex identification.
    
    Lambda2 is the second eigenvalue of S² + Ω²
    where S is strain rate tensor and Ω is vorticity tensor.
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Lambda2 field
    """
    dx, dy, dz = spacing
    
    # Compute velocity gradients
    du_dx = np.gradient(u, dx, axis=0)
    du_dy = np.gradient(u, dy, axis=1)
    du_dz = np.gradient(u, dz, axis=2)
    
    dv_dx = np.gradient(v, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dv_dz = np.gradient(v, dz, axis=2)
    
    dw_dx = np.gradient(w, dx, axis=0)
    dw_dy = np.gradient(w, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)
    
    # Strain rate tensor S
    S11 = du_dx
    S22 = dv_dy
    S33 = dw_dz
    S12 = S21 = 0.5 * (du_dy + dv_dx)
    S13 = S31 = 0.5 * (du_dz + dw_dx)
    S23 = S32 = 0.5 * (dv_dz + dw_dy)
    
    # Vorticity tensor Ω
    O11 = O22 = O33 = 0.0  # Diagonal elements are zero
    O12 = 0.5 * (du_dy - dv_dx)
    O21 = -O12
    O13 = 0.5 * (du_dz - dw_dx)
    O31 = -O13
    O23 = 0.5 * (dv_dz - dw_dy)
    O32 = -O23
    
    # Compute S² + Ω² for each point
    lambda2 = np.zeros_like(u)
    
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                # S matrix at point (i,j,k)
                S = np.array([
                    [S11[i,j,k], S12[i,j,k], S13[i,j,k]],
                    [S21[i,j,k], S22[i,j,k], S23[i,j,k]],
                    [S31[i,j,k], S32[i,j,k], S33[i,j,k]]
                ])
                
                # Ω matrix at point (i,j,k)
                O = np.array([
                    [0, O12[i,j,k], O13[i,j,k]],
                    [-O12[i,j,k], 0, O23[i,j,k]],
                    [-O13[i,j,k], -O23[i,j,k], 0]
                ])
                
                # Compute S² + Ω²
                S2_O2 = np.dot(S, S) + np.dot(O, O)
                
                # Get eigenvalues and sort
                eigenvals = np.linalg.eigvals(S2_O2)
                eigenvals_sorted = np.sort(eigenvals)
                
                # Lambda2 is the second (middle) eigenvalue
                lambda2[i,j,k] = eigenvals_sorted[1]
    
    return lambda2


def compute_enstrophy(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute enstrophy (0.5 * ||ω||²).
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Enstrophy field
    """
    vorticity_mag = compute_vorticity_magnitude(u, v, w, spacing)
    return 0.5 * vorticity_mag**2


def compute_helicity(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    spacing: Tuple[float, float, float]
) -> np.ndarray:
    """
    Compute helicity (v · ω).
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        spacing: Grid spacing (dx, dy, dz)
        
    Returns:
        Helicity field
    """
    omega_x, omega_y, omega_z = compute_vorticity(u, v, w, spacing)
    return u * omega_x + v * omega_y + w * omega_z


def compute_reynolds_stress(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Reynolds stress components (turbulent fluctuations).
    
    Args:
        u, v, w: Velocity components (nx, ny, nz)
        
    Returns:
        Reynolds stress components (u'u', v'v', w'w', u'v', u'w', v'w')
    """
    # Compute mean velocities
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    w_mean = np.mean(w)
    
    # Compute fluctuations
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean
    
    # Compute Reynolds stress components
    uu = u_prime * u_prime
    vv = v_prime * v_prime
    ww = w_prime * w_prime
    uv = u_prime * v_prime
    uw = u_prime * w_prime
    vw = v_prime * w_prime
    
    return uu, vv, ww, uv, uw, vw