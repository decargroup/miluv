import numpy as np
from pymlg import SO3, SE23

gravity = np.array([0, 0, -9.81])

def N_mat(phi_vec: np.ndarray) -> np.ndarray:
    if np.linalg.norm(phi_vec) < SO3._small_angle_tol:
        return np.identity(3)
    else:
        phi = np.linalg.norm(phi_vec)
        a = (phi_vec / phi).reshape((-1, 1))
        a_wedge = SO3.wedge(a)
        c = (1 - np.cos(phi)) / phi**2
        s = (phi - np.sin(phi)) / phi**2
        return 2 * c * np.identity(3) + (1 - 2 * c) * (a @ a.T) + 2 * s * a_wedge

def G_mat(dt: float) -> np.ndarray:
    G = np.identity(5)
    G[:3, 3] = dt * gravity
    G[:3, 4] = -0.5 * dt**2 * gravity
    G[3, 4] = -dt
    return G

def U_mat(bias: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    phi = (u[:3] - bias[:3]) * dt
    a = (u[3:] - bias[3:]).reshape((-1, 1))
    U = np.identity(5)
    U[:3, :3] = SO3.Exp(phi)
    U[:3, 3] = (dt * SO3.left_jacobian(phi) @ a).ravel()
    U[:3, 4] = (dt**2 / 2 * N_mat(phi) @ a).ravel()
    U[3, 4] = dt
    return U

def U_inv_mat(bias: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    U = U_mat(bias, u, dt)
    
    R = U[:3, :3]
    c = U[3, 4]
    a = U[:3, 3].reshape((-1, 1))
    b = U[:3, 4].reshape((-1, 1))
    U_inv = np.identity(5)
    U_inv[:3, :3] = R.T
    U_inv[:3, 3] = np.ravel(-R.T @ a)
    U_inv[:3, 4] = np.ravel(R.T @ (c * a - b))
    U_inv[3, 4] = np.ravel(-c)
    return U_inv

def U_adjoint_mat(U: np.ndarray) -> np.ndarray:
    R = U[:3, :3]
    c = U[3, 4]
    a = U[:3, 3].reshape((-1, 1))
    b = U[:3, 4].reshape((-1, 1))
    Ad = np.zeros((9, 9))
    Ad[:3, :3] = R
    Ad[3:6, :3] = SO3.wedge(a) @ R
    Ad[3:6, 3:6] = R

    Ad[6:9, :3] = -SO3.wedge(c * a - b) @ R
    Ad[6:9, 3:6] = -c * R
    Ad[6:9, 6:9] = R
    return Ad
    
def L_mat(bias: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    a = (u[3:] - bias[3:]).reshape((-1, 1))
    omdt = (u[:3] - bias[:3]) * dt
    J_att_inv_times_N = SO3.left_jacobian_inv(omdt) @ N_mat(omdt)
    xi = np.zeros((9,))
    xi[:3] = omdt
    xi[3:6] = (dt * a).ravel()
    xi[6:9] = ((dt**2 / 2) * J_att_inv_times_N @ a).ravel()
    J = SE23.left_jacobian(-xi)
    
    Om = SO3.wedge(omdt)
    OmOm = Om @ Om
    Up = dt * np.eye(9, 6)
    W = OmOm @ SO3.wedge(a) + Om @ SO3.wedge(Om @ a) + SO3.wedge(OmOm @ a)
    Up[6:9, 0:3] = dt**3 * (1 / 12 * SO3.wedge(a) - 1 / 720 * dt**2 * W)
    Up[6:9, 3:6] = (dt**2 / 2) * J_att_inv_times_N

    L = J @ Up
    return L