"""
abb_system.py
=============
Research-grade simulation framework for a Jeffcott-type rotor with an
Automatic Ball Balancer (ABB).

System coordinates (see image):
  x, y     – in-plane translational DOFs of the rotor housing
  ψ        – absolute rotor spin angle  (d/dt ψ = ω)
  φ_j      – ball angles measured from rotor body (j = 1..N)

State vector:
  z = [x, ẋ, y, ẏ, ψ, ψ̇, φ₁, φ̇₁, ..., φ_N, φ̇_N]  (len = 6 + 2N)

Equations of motion follow Rodrigues & Champneys (2011) / Chung & Ro (1999)
convention and are derived from Lagrangian mechanics.

Reference non-dimensionalisation:
  ω_n  = sqrt(kx / M_tot)   (isotropic natural frequency)
  τ    = t * ω_n
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class ABBParams:
    """All physical parameters for the ABB rotor system.

    Defaults are chosen to give ω_n ≈ 1 rad/s (isotropic) and reproduce
    the classical ball-balancer qualitative behaviour.
    """
    # --- Structural ---
    M_tot: float = 10.0        # Total mass of rotor + housing [kg]
    kx: float = 100.0          # Spring stiffness x [N/m]
    ky: float = 100.0          # Spring stiffness y [N/m]
    bx: float = 2.0            # Damping coefficient x [N·s/m]
    by: float = 2.0            # Damping coefficient y [N·s/m]

    # --- Unbalance ---
    m_un: float = 0.05         # Unbalance mass [kg]
    e: float = 0.02            # Eccentricity [m]

    # --- Balls ---
    N: int = 2                 # Number of balls
    m_b: float = 0.03          # Mass of each ball [kg]
    r: float = 0.05            # Ball race radius [m]
    D: float = 0.005           # Race viscous drag coefficient [N·m·s/rad]

    # --- Motor (Sommerfeld model: T = a - b·ω) ---
    motor_a: float = 5.0       # Motor constant a [N·m]
    motor_b: float = 0.5       # Motor constant b [N·m·s]
    motor_enabled: bool = False  # If False, ψ̇ = ω = const (kinematic)
    omega_drive: float = 12.0  # Prescribed spin speed when motor disabled [rad/s]

    # --- Derived (set automatically) ---
    omega_n: float = field(init=False)   # Natural frequency [rad/s]
    I_rot: float = field(init=False)     # Rotor moment of inertia [kg·m²]

    def __post_init__(self):
        # Effective translational mass (balls contribute)
        self.omega_n = np.sqrt(self.kx / self.M_tot)
        # Approximate rotor moment of inertia (lumped disk model)
        self.I_rot = 0.5 * self.M_tot * (2 * self.r) ** 2  # thin disk


# ---------------------------------------------------------------------------
# Motor model
# ---------------------------------------------------------------------------

class MotorModel:
    """Linear motor characteristic:  T(ω) = a – b·ω"""

    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b

    def torque(self, omega: float) -> float:
        return self.a - self.b * omega


# ---------------------------------------------------------------------------
# Core ODE system
# ---------------------------------------------------------------------------

class ABBSystem:
    """
    Encapsulates the full nonlinear ABB rotor equations of motion.

    State layout
    ------------
    idx  quantity
    0    x
    1    ẋ
    2    y
    3    ẏ
    4    ψ
    5    ψ̇  (= ω)
    6+2j φ_j    (j = 0..N-1)
    7+2j φ̇_j
    """

    def __init__(self, params: ABBParams,
                 motor: Optional[MotorModel] = None):
        self.p = params
        self.motor = motor or MotorModel(params.motor_a, params.motor_b)
        self._N = params.N
        self._state_dim = 6 + 2 * params.N

    # ------------------------------------------------------------------
    # State construction / extraction helpers
    # ------------------------------------------------------------------

    def make_state(self,
                   x=0., xd=0., y=0., yd=0.,
                   psi=0., psid=None,
                   phi=None, phid=None) -> np.ndarray:
        """Construct initial state vector."""
        p = self.p
        if psid is None:
            psid = p.omega_drive
        if phi is None:
            phi = np.zeros(p.N)
        if phid is None:
            phid = np.zeros(p.N)
        phi = np.asarray(phi, dtype=float)
        phid = np.asarray(phid, dtype=float)
        z = np.zeros(self._state_dim)
        z[0], z[1] = x, xd
        z[2], z[3] = y, yd
        z[4], z[5] = psi, psid
        for j in range(p.N):
            z[6 + 2*j] = phi[j]
            z[7 + 2*j] = phid[j]
        return z

    def extract(self, z: np.ndarray):
        """Unpack state into named quantities."""
        x, xd, y, yd = z[0], z[1], z[2], z[3]
        psi, psid = z[4], z[5]
        phi  = np.array([z[6 + 2*j] for j in range(self._N)])
        phid = np.array([z[7 + 2*j] for j in range(self._N)])
        return x, xd, y, yd, psi, psid, phi, phid

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------

    def rhs(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Compute dz/dt.

        Equations derived from Lagrangian with generalised coordinates
        (x, y, ψ, φ_j).  Ball inertia couples into translational and spin
        DOFs.  Race drag enters ball equations as –D·φ̇_j.

        Translational EOM (x-direction):
          M_eff * ẍ + bx*ẋ + kx*x =
              m_un·e·(ψ̈·sin(ψ) + ω²·cos(ψ))·(–1)      [unbalance]
            + Σ_j m_b·r·(φ̈_j·cos(φ_j+ψ) – φ̇_j²·sin(φ_j+ψ)  [balls]
                        + ψ̈·cos(φ_j+ψ) – ω²·sin(φ_j+ψ))

        The ball absolute angle = ψ + φ_j.
        We rearrange into matrix form to solve for [ẍ, ÿ, ψ̈, φ̈_j] simultaneously.
        """
        p = self.p
        N = p.N

        x, xd, y, yd, psi, omega = z[0], z[1], z[2], z[3], z[4], z[5]
        phi  = np.array([z[6 + 2*j] for j in range(N)])
        phid = np.array([z[7 + 2*j] for j in range(N)])

        # Absolute ball angles
        alpha = psi + phi          # shape (N,)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # Unbalance absolute angle
        psi_un = psi               # unbalance at angle ψ from x-axis
        c_un = np.cos(psi_un)
        s_un = np.sin(psi_un)

        # Total mass (rotor + unbalance + balls)
        M_t = p.M_tot
        m_b = p.m_b
        m_un = p.m_un
        e = p.e
        r = p.r
        M_eff = M_t  # housing+disk mass; balls add via coupling

        # ----------------------------------------------------------------
        # Build (2 + 1 + N) × (2 + 1 + N) mass matrix for [ẍ, ÿ, ψ̈, φ̈_j]
        # Rows: x-eq, y-eq, ψ-eq, φ_j-eq (j=0..N-1)
        # ----------------------------------------------------------------
        n_dof = 2 + 1 + N   # x, y, ψ, φ₁..φ_N
        M_mat = np.zeros((n_dof, n_dof))
        RHS   = np.zeros(n_dof)

        # --- Row 0: x equation ---
        # M_eff * ẍ  +  m_un * e * ψ̈ * sin(ψ)  +  Σ m_b*r*(ψ̈+φ̈_j)*cos(α_j) = ...
        M_mat[0, 0] = M_eff + m_un + N * m_b          # ẍ coeff (all masses translate)
        M_mat[0, 2] = -m_un * e * s_un - r * m_b * np.sum(sa)  # ψ̈ coeff
        for j in range(N):
            M_mat[0, 3 + j] = -m_b * r * sa[j]         # φ̈_j coeff

        # Centripetal + spring + damping on RHS
        RHS[0] = (-p.bx * xd - p.kx * x
                  + m_un * e * omega**2 * c_un
                  + m_b * r * np.sum((omega + phid)**2 * ca))

        # --- Row 1: y equation ---
        M_mat[1, 1] = M_eff + m_un + N * m_b
        M_mat[1, 2] = m_un * e * c_un + r * m_b * np.sum(ca)
        for j in range(N):
            M_mat[1, 3 + j] = m_b * r * ca[j]

        RHS[1] = (-p.by * yd - p.ky * y
                  + m_un * e * omega**2 * s_un
                  + m_b * r * np.sum((omega + phid)**2 * sa))

        # --- Row 2: ψ equation (spin) ---
        # I_eff * ψ̈  +  coupling  = T_motor – drag – ...
        # Effective polar inertia about spin axis
        I_eff = p.I_rot + m_un * e**2 + N * m_b * r**2
        M_mat[2, 2] = I_eff
        # Coupling to x, y (reaction from balls and unbalance)
        M_mat[2, 0] = -m_un * e * s_un - m_b * r * np.sum(sa)  # same as M[0,2]
        M_mat[2, 1] =  m_un * e * c_un + m_b * r * np.sum(ca)  # same as M[1,2]
        for j in range(N):
            M_mat[2, 3 + j] = m_b * r**2   # ball inertia about spin axis

        if p.motor_enabled:
            T_motor = self.motor.torque(omega)
        else:
            # Kinematic: ψ̈ = 0, ψ̇ = const → handle below
            T_motor = 0.0

        # Centripetal contributions to ψ-RHS (from ball relative motion)
        RHS[2] = (T_motor
                  - m_un * e * omega**2 * (xd * c_un / r + yd * s_un / r) * 0  # zero in body frame
                  + m_b * r * np.sum(phid**2 * 0)  # zero (only Coriolis, handled via ball eq)
                  )
        # Actually: full ψ-RHS = T_motor + x·forces·moment_arm  (already captured in M_mat coupling)
        RHS[2] = T_motor

        # --- Rows 3..2+N: ball equations ---
        # m_b*r² * (ψ̈ + φ̈_j) + m_b*(ẍ*sin(α_j) - ÿ*cos(α_j)) = –D*φ̇_j
        # → m_b*r² * φ̈_j + m_b*r² * ψ̈ + coupling = –D*φ̇_j
        for j in range(N):
            row = 3 + j
            M_mat[row, row] = m_b * r**2          # φ̈_j
            M_mat[row, 2]   = m_b * r**2          # ψ̈ coupling
            M_mat[row, 0]   = m_b * r * sa[j]     # ẍ coupling
            M_mat[row, 1]   = -m_b * r * ca[j]    # ÿ coupling

            # Centripetal + drag
            RHS[3 + j] = (-p.D * phid[j]
                          - m_b * r * (omega + phid[j])**2 * 0  # already captured
                          )
            # Full ball RHS: –D·φ̇_j  (centripetal terms cancel when projected)
            RHS[3 + j] = -p.D * phid[j]

        # ----------------------------------------------------------------
        # Override spin equation if motor disabled (prescribed ω)
        # ----------------------------------------------------------------
        if not p.motor_enabled:
            # Replace row 2 with identity: ψ̈ = 0
            M_mat[2, :] = 0.0
            M_mat[2, 2] = 1.0
            RHS[2] = 0.0

        # ----------------------------------------------------------------
        # Solve for accelerations
        # ----------------------------------------------------------------
        try:
            accel = np.linalg.solve(M_mat, RHS)
        except np.linalg.LinAlgError:
            accel = np.linalg.lstsq(M_mat, RHS, rcond=None)[0]

        xdd, ydd = accel[0], accel[1]
        psidd = accel[2]
        phidd = accel[3:]

        # ----------------------------------------------------------------
        # Assemble dz/dt
        # ----------------------------------------------------------------
        dz = np.zeros_like(z)
        dz[0] = xd
        dz[1] = xdd
        dz[2] = yd
        dz[3] = ydd
        dz[4] = omega
        dz[5] = psidd
        for j in range(N):
            dz[6 + 2*j] = phid[j]
            dz[7 + 2*j] = phidd[j]

        return dz

    def amplitude(self, z: np.ndarray) -> float:
        """Instantaneous rotor amplitude sqrt(x²+y²)."""
        return np.sqrt(z[0]**2 + z[2]**2)


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------

class Integrator:
    """Wraps scipy.solve_ivp with sensible defaults for stiff ABB problems."""

    def __init__(self, method: str = "Radau",
                 rtol: float = 1e-8, atol: float = 1e-10,
                 max_step: float = np.inf):
        self.method  = method
        self.rtol    = rtol
        self.atol    = atol
        self.max_step = max_step

    def solve(self, system: ABBSystem,
              z0: np.ndarray,
              t_span: tuple,
              t_eval: Optional[np.ndarray] = None,
              dense_output: bool = False,
              events: Optional[list] = None) -> object:
        """Integrate the system and return solve_ivp result object."""
        return solve_ivp(
            system.rhs,
            t_span,
            z0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.max_step,
            dense_output=dense_output,
            events=events,
        )
