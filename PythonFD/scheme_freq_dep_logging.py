#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, inv
from Vector_Fitting_for_python.vectfit3 import vectfit, opts
from tqdm import tqdm  # For the loading bar
import matplotlib.patches as patches  # For plotting room layout

# =============================================================================
# Simulation Classes
# =============================================================================
class ivp(object):
    def __init__(self, c=343, h=0.001, Lx=0.8, Ly=0.6, dt=1e-6, bnd='Neumann',
                 impedance=None, U0_func=None, V0_func=None, rhs=None, dir_func=None,
                 t_end=0.01, mic_x=0.4, mic_y=0.3, animation=False, error=False):
        self.c = c
        self.h = h
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.bnd = bnd
        self.impedance = impedance
        self.U0_func = U0_func
        self.V0_func = V0_func
        self.rhs = rhs
        self.dir_func = dir_func
        self.t_end = t_end
        self.mic_x = mic_x
        self.mic_y = mic_y
        self.animation = animation
        self.error = error
        self.recompute_grid()

    def recompute_grid(self):
        x = np.arange(0, self.Lx + self.h, self.h)
        y = np.arange(0, self.Ly + self.h, self.h)
        self.Nx = len(x)
        self.Ny = len(y)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        if self.U0_func is not None:
            self.U0 = self.U0_func(self.X, self.Y, 0).astype(complex)
        else:
            self.U0 = np.zeros_like(self.X, dtype=complex)
        if self.V0_func is not None:
            self.V0 = self.V0_func(self.X, self.Y, 0).astype(complex)
        else:
            self.V0 = np.zeros_like(self.X, dtype=complex)
        x_coords = np.arange(0, self.Lx + self.h, self.h)
        y_coords = np.arange(0, self.Ly + self.h, self.h)
        self.mic_i = np.argmin(np.abs(x_coords - self.mic_x))
        self.mic_j = np.argmin(np.abs(y_coords - self.mic_y))

class impedance_boundary(object):
    def __init__(self, left=None, right=None, bottom=None, top=None):
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top

# =============================================================================
# Boundary and Update Functions (unchanged from your code)
# =============================================================================
def update_boundary(ivp, u_next, u, u_prev, t):
    Nx = ivp.Nx - 1
    Ny = ivp.Ny - 1
    k_val = ivp.c * ivp.dt / ivp.h

    if ivp.bnd == 'Neumann':
        u_next[1:Nx, 0] = k_val**2 * (-4*u[1:Nx, 0] + u[2:, 0] + u[0:Nx-1, 0] + 2*u[1:Nx, 1]) + 2*u[1:Nx, 0] - u_prev[1:Nx, 0]
        u_next[1:Nx, Ny] = k_val**2 * (-4*u[1:Nx, Ny] + u[2:, Ny] + u[0:Nx-1, Ny] + 2*u[1:Nx, Ny-1]) + 2*u[1:Nx, Ny] - u_prev[1:Nx, Ny]
        u_next[0, 1:Ny] = k_val**2 * (-4*u[0, 1:Ny] + 2*u[1, 1:Ny] + u[0, 0:Ny-1] + u[0, 2:]) + 2*u[0, 1:Ny] - u_prev[0, 1:Ny]
        u_next[Nx, 1:Ny] = k_val**2 * (-4*u[Nx, 1:Ny] + 2*u[Nx-1, 1:Ny] + u[Nx, 0:Ny-1] + u[Nx, 2:]) + 2*u[Nx, 1:Ny] - u_prev[Nx, 1:Ny]
        u_next[0, 0] = k_val**2 * (-4*u[0, 0] + 2*u[1, 0] + 2*u[0, 1]) + 2*u[0, 0] - u_prev[0, 0]
        u_next[Nx, 0] = k_val**2 * (2*u[Nx-1, 0] - 4*u[Nx, 0] + 2*u[Nx, 1]) + 2*u[Nx, 0] - u_prev[Nx, 0]
        u_next[0, Ny] = k_val**2 * (2*u[0, Ny-1] - 4*u[0, Ny] + 2*u[1, Ny]) + 2*u[0, Ny] - u_prev[0, Ny]
        u_next[Nx, Ny] = k_val**2 * (-4*u[Nx, Ny] + 2*u[Nx-1, Ny] + 2*u[Nx, Ny-1]) + 2*u[Nx, Ny] - u_prev[Nx, Ny]
    elif ivp.bnd == 'Dirichlet':
        g = ivp.dir_func(ivp.X, ivp.Y, t+ivp.dt)
        u_next[0, :] = g[0, :]
        u_next[-1, :] = g[-1, :]
        u_next[:, 0] = g[:, 0]
        u_next[:, -1] = g[:, -1]
    else:
        raise Exception("Boundary condition provided not an option!")

    if ivp.impedance is not None:
        u_next = impedance_update(ivp, u_next, u, u_prev, t)
    return u_next

def update_impedance_plate(ivp, u_next, u, u_prev, t, wall, plate_params):
    dt = ivp.dt
    h = ivp.h
    scale_factor = -dt
    SER = plate_params['SER']
    psi_array = plate_params['psi']
    Phi = plate_params['Phi']
    Gamma = plate_params['Gamma']
    start_idx, end_idx = plate_params['indices']

    for i, idx in enumerate(range(start_idx, end_idx)):
        if wall == 'left':
            v_current = (u[0, idx] - u[1, idx]) / h
            pos = (0, idx)
        elif wall == 'right':
            v_current = (u[-1, idx] - u[-2, idx]) / h
            pos = (-1, idx)
        elif wall == 'bottom':
            v_current = (u[idx, 0] - u[idx, 1]) / h
            pos = (idx, 0)
        elif wall == 'top':
            v_current = (u[idx, -1] - u[idx, -2]) / h
            pos = (idx, -1)
        else:
            raise ValueError("Invalid wall specification.")

        psi_old = psi_array[:, i]
        psi_new = Phi @ psi_old + Gamma.flatten() * v_current
        psi_array[:, i] = psi_new

        ser_output = np.squeeze(np.dot(SER['C'], psi_new)) + SER['D'] * v_current
        correction = scale_factor * ser_output

        u_next[pos] = u[pos] + correction

        plate_params['v_prev'][i] = v_current

    return u_next

def impedance_update(ivp, u_next, u, u_prev, t):
    if ivp.impedance.left is not None:
        if isinstance(ivp.impedance.left, list):
            for plate in ivp.impedance.left:
                u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'left', plate)
        else:
            u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'left', ivp.impedance.left)
    if ivp.impedance.right is not None:
        if isinstance(ivp.impedance.right, list):
            for plate in ivp.impedance.right:
                u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'right', plate)
        else:
            u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'right', ivp.impedance.right)
    if ivp.impedance.bottom is not None:
        if isinstance(ivp.impedance.bottom, list):
            for plate in ivp.impedance.bottom:
                u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'bottom', plate)
        else:
            u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'bottom', ivp.impedance.bottom)
    if ivp.impedance.top is not None:
        if isinstance(ivp.impedance.top, list):
            for plate in ivp.impedance.top:
                u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'top', plate)
        else:
            u_next = update_impedance_plate(ivp, u_next, u, u_prev, t, 'top', ivp.impedance.top)
    return u_next

def central_diff_scheme(ivp, u, u_prev, t):
    Nx, Ny = ivp.Nx, ivp.Ny
    u_next = np.zeros_like(u)
    index_x = np.arange(1, Nx-1)
    index_y = np.arange(1, Ny-1)
    ixy = np.ix_(index_x, index_y)
    ixm_y = np.ix_(index_x-1, index_y)
    ixp_y = np.ix_(index_x+1, index_y)
    ix_ym = np.ix_(index_x, index_y-1)
    ix_yp = np.ix_(index_x, index_y+1)
    k_val = ivp.c * ivp.dt / ivp.h
    assert k_val <= 1/np.sqrt(2), f"k = {k_val} must be <= {1/np.sqrt(2)}"
    u_next[ixy] = 2*u[ixy] - u_prev[ixy] + k_val**2 * (u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp] - 4*u[ixy])
    u_next = update_boundary(ivp, u_next, u, u_prev, t)
    u_next += ivp.dt**2 * ivp.rhs(ivp.X, ivp.Y, t, ivp.c)
    return u_next

def u_next_first(ivp, u):
    Nx, Ny = ivp.Nx, ivp.Ny
    u_next = np.zeros_like(u, dtype=complex)
    index_x = np.arange(1, Nx-1)
    index_y = np.arange(1, Ny-1)
    ixy = np.ix_(index_x, index_y)
    ixm_y = np.ix_(index_x-1, index_y)
    ixp_y = np.ix_(index_x+1, index_y)
    ix_ym = np.ix_(index_x, index_y-1)
    ix_yp = np.ix_(index_x, index_y+1)
    k_val = ivp.c * ivp.dt / ivp.h
    assert k_val <= 1/np.sqrt(2), f"k = {k_val} must be <= {1/np.sqrt(2)}"
    u_next[ixy] = u[ixy] + 0.5 * k_val**2 * (u[ixm_y] + u[ixp_y] + u[ix_ym] + u[ix_yp] - 4*u[ixy])

    if ivp.bnd == 'Dirichlet':
        g = ivp.dir_func(ivp.X, ivp.Y, ivp.dt)
        u_next[0, :] = g[0, :]
        u_next[-1, :] = g[-1, :]
        u_next[:, 0] = g[:, 0]
        u_next[:, -1] = g[:, -1]
    elif ivp.bnd == 'Neumann':
        Nx_minus = ivp.Nx - 1
        Ny_minus = ivp.Ny - 1
        u_next[1:Nx_minus, 0] = 0.5 * (k_val**2 * (-4*u[1:Nx_minus, 0] + u[2:, 0] + u[0:Nx_minus-1, 0] + 2*u[1:Nx_minus, 1]) 
                                       + 2*u[1:Nx_minus, 0] + 2*ivp.dt*ivp.V0[1:Nx_minus, 0])
        u_next[1:Nx_minus, Ny_minus] = 0.5 * (k_val**2 * (-4*u[1:Nx_minus, Ny_minus] + u[2:, Ny_minus] + u[0:Nx_minus-1, Ny_minus] + 2*u[1:Nx_minus, Ny_minus-1]) 
                                              + 2*u[1:Nx_minus, Ny_minus] + 2*ivp.dt*ivp.V0[1:Nx_minus, Ny_minus])
        u_next[0, 1:Ny_minus] = 0.5 * (k_val**2 * (-4*u[0, 1:Ny_minus] + 2*u[1, 1:Ny_minus] + u[0, 0:Ny_minus-1] + u[0, 2:]) 
                                       + 2*u[0, 1:Ny_minus] + 2*ivp.dt*ivp.V0[0, 1:Ny_minus])
        u_next[Nx_minus, 1:Ny_minus] = 0.5 * (k_val**2 * (-4*u[Nx_minus, 1:Ny_minus] + 2*u[Nx_minus-1, 1:Ny_minus] + u[Nx_minus, 0:Ny_minus-1] + u[Nx_minus, 2:]) 
                                              + 2*u[Nx_minus, 1:Ny_minus] + 2*ivp.dt*ivp.V0[Nx_minus, 1:Ny_minus])
        u_next[0, 0] = 0.5 * (k_val**2 * (-4*u[0, 0] + 2*u[1, 0] + 2*u[0, 1]) + 2*u[0, 0] + 2*ivp.dt*ivp.V0[0, 0])
        u_next[Nx_minus, 0] = 0.5 * (k_val**2 * (2*u[Nx_minus-1, 0] - 4*u[Nx_minus, 0] + 2*u[Nx_minus, 1]) + 2*u[Nx_minus, 0] + 2*ivp.dt*ivp.V0[Nx_minus, 0])
        u_next[0, Ny_minus] = 0.5 * (k_val**2 * (2*u[0, Ny_minus-1] - 4*u[0, Ny_minus] + 2*u[1, Ny_minus]) + 2*u[0, Ny_minus] + 2*ivp.dt*ivp.V0[0, Ny_minus])
        u_next[Nx_minus, Ny_minus] = 0.5 * (k_val**2 * (-4*u[Nx_minus, Ny_minus] + 2*u[Nx_minus-1, Ny_minus] + 2*u[Nx_minus, Ny_minus-1]) + 2*u[Nx_minus, Ny_minus] + 2*ivp.dt*ivp.V0[Nx_minus, Ny_minus])
    if ivp.bnd == 'Dirichlet':
        u_next[ixy] += 0.5 * ivp.dt**2 * ivp.rhs(ivp.X, ivp.Y, 0, ivp.c)[ixy]
    else:
        u_next += 0.5 * ivp.dt**2 * ivp.rhs(ivp.X, ivp.Y, 0, ivp.c)
    return u_next

def run_central_diff_scheme(ivp):
    u = ivp.U0.copy()
    u_next = u_next_first(ivp, u)
    if ivp.animation:
        anim = np.zeros((ivp.Nx, ivp.Ny, int(ivp.t_end / (ivp.animation.delim * ivp.dt)) + 2), dtype=complex)
        anim[:, :, 0] = ivp.U0
        anim[:, :, 1] = u_next
    else:
        anim = None
    t = 2 * ivp.dt
    n = 2
    mic_signal = [ivp.U0[ivp.mic_i, ivp.mic_j], u_next[ivp.mic_i, ivp.mic_j]]
    
    # --------------------------------------------------
    # Set up sensor logging along the walls.
    # Instead of reading every grid point, we place sensors every 0.25 m.
    sensor_step = int(0.25 / ivp.h)  # For h=0.01, sensor_step = 25.
    
    # Place the sensors a bit further in from the wall: 0.05 m offset.
    sensor_offset = int(0.01 / ivp.h)  # For h=0.01, sensor_offset = 5.
    left_sensor_idx = sensor_offset
    right_sensor_idx = ivp.Nx - 1 - sensor_offset
    bottom_sensor_idx = sensor_offset
    top_sensor_idx = ivp.Ny - 1 - sensor_offset

    # Log only every sensor_step element from these rows/columns.
    sensor_left = []   # For left wall: sample along the y-direction.
    sensor_right = []  # For right wall.
    sensor_bottom = [] # For bottom wall: sample along the x-direction.
    sensor_top = []    # For top wall.
    sensor_time = []   # Time vector for sensor logs.

    # Record initial sensor readings at time t = 0 from the initial condition.
    sensor_time.append(0)
    sensor_left.append(u[left_sensor_idx, ::sensor_step].copy())
    sensor_right.append(u[right_sensor_idx, ::sensor_step].copy())
    sensor_bottom.append(u[:, bottom_sensor_idx][::sensor_step].copy())
    sensor_top.append(u[:, top_sensor_idx][::sensor_step].copy())

    total_steps = int((ivp.t_end - t) / ivp.dt)
    
    for _ in tqdm(range(total_steps), desc="Simulating", leave=True):
        u_old = u.copy()
        u = u_next.copy()
        u_prev = u_old.copy()
        u_next = central_diff_scheme(ivp, u, u_prev, t)
        mic_signal.append(u_next[ivp.mic_i, ivp.mic_j])
        
        # Log sensor readings using the defined sensor_step.
        sensor_left.append(u_next[left_sensor_idx, ::sensor_step].copy())
        sensor_right.append(u_next[right_sensor_idx, ::sensor_step].copy())
        sensor_bottom.append(u_next[:, bottom_sensor_idx][::sensor_step].copy())
        sensor_top.append(u_next[:, top_sensor_idx][::sensor_step].copy())
        sensor_time.append(t)
        
        if ivp.animation and (n % ivp.animation.delim == 0):
            frame = int(n / ivp.animation.delim)
            anim[:, :, frame] = u_next
        n += 1
        t += ivp.dt
    mic_signal = np.array(mic_signal)
    
    sensor_logs = {
        'time': np.array(sensor_time),
        'left': np.array(sensor_left),       # Shape: (num_steps+1, N_left)
        'right': np.array(sensor_right),     # Shape: (num_steps+1, N_right)
        'bottom': np.array(sensor_bottom),   # Shape: (num_steps+1, N_bottom)
        'top': np.array(sensor_top)          # Shape: (num_steps+1, N_top)
    }
    
    os.makedirs("sensor_logs", exist_ok=True)
    np.savetxt(os.path.join("sensor_logs", "sensor_left.txt"),
               np.column_stack((sensor_logs['time'], sensor_logs['left'])),
               header="Time(s) " + " ".join([f"sensor_y{i}" for i in range(sensor_logs['left'].shape[1])]))
    np.savetxt(os.path.join("sensor_logs", "sensor_right.txt"),
               np.column_stack((sensor_logs['time'], sensor_logs['right'])),
               header="Time(s) " + " ".join([f"sensor_y{i}" for i in range(sensor_logs['right'].shape[1])]))
    np.savetxt(os.path.join("sensor_logs", "sensor_bottom.txt"),
               np.column_stack((sensor_logs['time'], sensor_logs['bottom'])),
               header="Time(s) " + " ".join([f"sensor_x{i}" for i in range(sensor_logs['bottom'].shape[1])]))
    np.savetxt(os.path.join("sensor_logs", "sensor_top.txt"),
               np.column_stack((sensor_logs['time'], sensor_logs['top'])),
               header="Time(s) " + " ".join([f"sensor_x{i}" for i in range(sensor_logs['top'].shape[1])]))
               
    return u_next, anim, mic_signal, t

# =============================================================================
# Delany Bazley Impedance and Vecfit3 SER Functionality
# =============================================================================
def delany_bazley_impedance(f, d, rho=1.21, c=343, sigma=20000):
    X = rho * f / sigma
    real_part_Z_p = 1.0 + 0.0571 * X**(-0.754)
    imag_part_Z_p = -0.087 * X**(-0.732)
    Z_p = (rho * c) * (real_part_Z_p + 1j * imag_part_Z_p)
    omega = 2 * np.pi * f
    real_part_k_p = 1.0 + 0.098 * X**(-0.7)
    imag_part_k_p = -0.189 * X**(-0.595)
    k_p = (omega / c) * (real_part_k_p + 1j * imag_part_k_p)
    Z = -1j * Z_p * (np.cos(k_p * d) / np.sin(k_p * d))
    return Z

def fit_delany_bazley_impedance_SER(frequencies, d, rho=1.21, c=343, sigma=20000, num_poles=2, Niter=3):
    Z_delany_bazley = delany_bazley_impedance(frequencies, d, rho, c, sigma)
    s = 1j * 2 * np.pi * frequencies

    if num_poles % 2 != 0:
        raise ValueError("num_poles must be even (for conjugate pairs).")
    n_pairs = num_poles // 2
    f_min = frequencies.min()
    f_max = frequencies.max()
    w_min = 2 * np.pi * f_min
    w_max = 2 * np.pi * f_max
    beta_vals = np.logspace(np.log10(w_min), np.log10(w_max), n_pairs)
    alpha_vals = beta_vals / 100.0
    initial_poles = np.empty(2 * n_pairs, dtype=complex)
    initial_poles[0::2] = -alpha_vals - 1j * beta_vals
    initial_poles[1::2] = -alpha_vals + 1j * beta_vals

    weights = np.ones_like(frequencies, dtype=complex)
    opts_iter = opts.copy()
    opts_iter["skip_res"] = True
    opts_iter["spy2"] = False
    opts_iter["asymp"] = 2

    poles = initial_poles
    SER_iter = None
    for itr in range(Niter):
        if itr == Niter - 1:
            opts_iter["spy2"] = True
            opts_iter["skip_res"] = False
        (SER_iter, poles, rmserr, fit) = vectfit(Z_delany_bazley[np.newaxis, :], s, poles, weights, opts_iter)
        print("     ...", itr + 1, " iterations applied")
    print(f"Rmserr: {rmserr}")
    print(f"Fitted poles: {poles}")
    print(f"SER: {SER_iter}")
    return SER_iter

def compute_Phi_and_Gamma(SER, dt):
    A = SER['A']
    Phi = expm(A * dt)
    Gamma = inv(A) @ (Phi - np.eye(A.shape[0])) @ SER['B']
    return Phi, Gamma

# =============================================================================
# Initial Conditions and RHS
# =============================================================================
def U0_func(x, y, t):
    # Adjusted to center the Gaussian in the 0.8x0.6 room
    x0, y0 = 4.0, 3.0
    p0 = 22440
    sigma = 0.01 
    return p0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2*sigma**2)) + 101325

def V0_func(X, Y, t):
    return np.zeros_like(X)

def rhs_func(X, Y, t, c):
    return np.zeros_like(X)

def constant_impedance_SER(Z0):
    
    return {'A': np.array([[-1.0]]),
            'B': np.array([[0.0]]),
            'C': np.array([[0.0]]),
            'D': float(Z0),
            'E': 0.0}

# =============================================================================
# Main Simulation Setup and Run
# =============================================================================
if __name__ == "__main__":
    # Simulation parameters for the small room
    c = 343
    h = 0.01
    Lx = 8.0
    Ly = 6.0
    dt = 1e-5
    t_end = 0.3

    frequencies = np.linspace(50, 2001, 50)
    d_plate = 0.13
    sigma = 15000
    #SER = fit_delany_bazley_impedance_SER(frequencies, d_plate, 1.21, 343, sigma, 6, 3)
    # If preferred, you could use constant impedance by uncommenting:
    Z = 250
    SER = constant_impedance_SER(Z)
    Phi, Gamma = compute_Phi_and_Gamma(SER, dt)

    def meter_to_index(m):
        return int(m / h)

    # Define impedance plates (layout can be adjusted if desired)
    # Bottom wall (y = 0)
    bottom_left_plate = {
        'indices': (meter_to_index(0.0), meter_to_index(1.0)),  # covers x = 0 to 1 m
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(1.0) - meter_to_index(0.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(1.0) - meter_to_index(0.0), dtype=complex)
    }
    bottom_right_plate = {
        'indices': (meter_to_index(7.0), meter_to_index(8.0)),  # covers x = 7 to 8 m (right corner)
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(8.0) - meter_to_index(7.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(8.0) - meter_to_index(7.0), dtype=complex)
    }

    # Top wall (y = Ly; here Ly = 6 m)
    top_left_plate = {
        'indices': (meter_to_index(0.0), meter_to_index(1.0)),  # covers x = 0 to 1 m
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(1.0) - meter_to_index(0.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(1.0) - meter_to_index(0.0), dtype=complex)
    }
    top_right_plate = {
        'indices': (meter_to_index(7.0), meter_to_index(8.0)),  # covers x = 7 to 8 m
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(8.0) - meter_to_index(7.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(8.0) - meter_to_index(7.0), dtype=complex)
    }

    # Left wall (x = 0); indices now refer to y-coordinates.
    left_bottom_plate = {
        'indices': (meter_to_index(0.0), meter_to_index(1.0)),  # covers y = 0 to 1 m
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(1.0) - meter_to_index(0.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(1.0) - meter_to_index(0.0), dtype=complex)
    }
    left_top_plate = {
        'indices': (meter_to_index(5.0), meter_to_index(6.0)),  # covers y = 5 to 6 m (upper corner)
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(6.0) - meter_to_index(5.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(6.0) - meter_to_index(5.0), dtype=complex)
    }

    # Right wall (x = Lx; here Lx = 8 m); indices refer to y-coordinates.
    right_bottom_plate = {
        'indices': (meter_to_index(0.0), meter_to_index(1.0)),  # covers y = 0 to 1 m
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(1.0) - meter_to_index(0.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(1.0) - meter_to_index(0.0), dtype=complex)
    }
    right_top_plate = {
        'indices': (meter_to_index(5.0), meter_to_index(6.0)),  # covers y = 5 to 6 m
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(6.0) - meter_to_index(5.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(6.0) - meter_to_index(5.0), dtype=complex)
    }

    bottom_center_plate = {
        'indices': (meter_to_index(3.5), meter_to_index(4.5)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(4.5) - meter_to_index(3.5)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(4.5) - meter_to_index(3.5), dtype=complex)
    }

    top_center_plate = {
        'indices': (meter_to_index(3.5), meter_to_index(4.5)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(4.5) - meter_to_index(3.5)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(4.5) - meter_to_index(3.5), dtype=complex)
    }

    left_off_center_plate = {
        'indices': (meter_to_index(1.0), meter_to_index(2.0)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(2.0) - meter_to_index(1.0)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(2.0) - meter_to_index(1.0), dtype=complex)
    }
    right_off_center_plate = {
        'indices': (meter_to_index(1.0), meter_to_index(2.0)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(2.0) - meter_to_index(1.)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(2.0) - meter_to_index(1.0), dtype=complex)
    }
    left_center_plate2 = {
        'indices': (meter_to_index(3.5), meter_to_index(4.5)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(4.5) - meter_to_index(3.5)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(4.5) - meter_to_index(3.5), dtype=complex)
    }
    right_center_plate2 = {
        'indices': (meter_to_index(4.5), meter_to_index(3.5)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(4.5) - meter_to_index(3.5)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(4.5) - meter_to_index(3.5), dtype=complex)
    }

    left_center_plate_middle = {
        'indices': (meter_to_index(2.5), meter_to_index(3.5)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(3.5) - meter_to_index(2.5)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(3.5) - meter_to_index(2.5), dtype=complex)
    }

    right_center_plate_middle = {
        'indices': (meter_to_index(2.5), meter_to_index(3.5)),
        'SER': SER,
        'psi': np.zeros((SER['A'].shape[0], meter_to_index(3.5) - meter_to_index(2.5)), dtype=complex),
        'Phi': Phi,
        'Gamma': Gamma,
        'v_prev': np.zeros(meter_to_index(3.5) - meter_to_index(2.5), dtype=complex)
    }

    imp_bnd = impedance_boundary(
        left=[left_top_plate, left_bottom_plate, left_center_plate_middle],
        right=[right_top_plate, right_bottom_plate, right_center_plate_middle],
        bottom=[bottom_left_plate, bottom_right_plate, bottom_center_plate],
        top=[top_left_plate, top_right_plate, top_center_plate],
    )


    simulation_ivp = ivp(c=c, h=h, Lx=Lx, Ly=Ly, dt=dt, bnd='Neumann',
                         impedance=imp_bnd, U0_func=U0_func, V0_func=V0_func,
                         rhs=rhs_func, t_end=t_end, mic_x=2.0, mic_y=3.0)
    
    plot_room = True
    if plot_room:
        import matplotlib.patches as patches
        fig, ax = plt.subplots()
        room_width = simulation_ivp.Lx
        room_height = simulation_ivp.Ly
        # Draw the room boundary as a black rectangle.
        ax.add_patch(patches.Rectangle((0, 0), room_width, room_height,
                                    fill=False, edgecolor='black', lw=2))
        
        # Bottom wall plates: plates appear along the x-direction at y = 0.
        if simulation_ivp.impedance.bottom is not None:
            bottom_plates = (simulation_ivp.impedance.bottom 
                            if isinstance(simulation_ivp.impedance.bottom, list)
                            else [simulation_ivp.impedance.bottom])
            for plate in bottom_plates:
                start_idx, end_idx = plate['indices']
                start_x = start_idx * simulation_ivp.h
                end_x = end_idx * simulation_ivp.h
                ax.plot([start_x, end_x], [0, 0], color='red', lw=4)
        
        # Top wall plates: plates appear along the x-direction at y = room_height.
        if simulation_ivp.impedance.top is not None:
            top_plates = (simulation_ivp.impedance.top 
                        if isinstance(simulation_ivp.impedance.top, list)
                        else [simulation_ivp.impedance.top])
            for plate in top_plates:
                start_idx, end_idx = plate['indices']
                start_x = start_idx * simulation_ivp.h
                end_x = end_idx * simulation_ivp.h
                ax.plot([start_x, end_x], [room_height, room_height], color='red', lw=4)
        
        # Left wall plates: plates appear vertically at x = 0.
        if simulation_ivp.impedance.left is not None:
            left_plates = (simulation_ivp.impedance.left 
                        if isinstance(simulation_ivp.impedance.left, list)
                        else [simulation_ivp.impedance.left])
            for plate in left_plates:
                start_idx, end_idx = plate['indices']
                start_y = start_idx * simulation_ivp.h
                end_y = end_idx * simulation_ivp.h
                ax.plot([0, 0], [start_y, end_y], color='red', lw=4)
        
        # Right wall plates: plates appear vertically at x = room_width.
        if simulation_ivp.impedance.right is not None:
            right_plates = (simulation_ivp.impedance.right 
                            if isinstance(simulation_ivp.impedance.right, list)
                            else [simulation_ivp.impedance.right])
            for plate in right_plates:
                start_idx, end_idx = plate['indices']
                start_y = start_idx * simulation_ivp.h
                end_y = end_idx * simulation_ivp.h
                ax.plot([room_width, room_width], [start_y, end_y], color='red', lw=4)
    
    ax.set_xlim(-1, room_width + 1)
    ax.set_ylim(-1, room_height + 1)
    ax.set_aspect('equal')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Room Layout with Impedance Plates")
    # Remove duplicate labels in the legend.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True)
    plt.show()

    # Run the simulation with sensor logging.
    u_final, anim, mic_signal, final_time = run_central_diff_scheme(simulation_ivp)

    time_array = np.arange(len(mic_signal)) * dt
    plt.figure()
    plt.plot(time_array, np.real(mic_signal), marker='o', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure at microphone")
    plt.title("Microphone Signal")
    plt.grid(True)
    plt.show()

    np.savetxt("mic_response_log.txt",
               np.column_stack((time_array, mic_signal)),
               header="Time(s)  Pressure(Pa)")
