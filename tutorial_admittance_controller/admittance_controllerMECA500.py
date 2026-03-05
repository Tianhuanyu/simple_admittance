# admittance_controller.py
# Minimal 1st-order admittance controller (Windows/Linux/macOS, no ROS, no optimizers)
# - Reads a URDF in the current working directory
# - FK via urdfpy; Jacobian via pykin (if available) else numeric finite differences
# - Interface compatible with the original: __call__(target_poses, q, tau, ee_force, collision) -> dq
# - Comments in English

from __future__ import annotations
import os
import numpy as np
from typing import Any, Dict, Optional

# ---- (Optional) compatibility shim for very old NetworkX usage on Py>=3.10
#      Harmless if not needed; prevents rare "collections.Mapping" import errors.
import collections as _collections, collections.abc as _abc
for _n in ("Mapping", "MutableMapping", "Sequence"):
    if not hasattr(_collections, _n):
        setattr(_collections, _n, getattr(_abc, _n))

# ------------------------------
# Small SO(3)/SE(3) helpers
# ------------------------------
def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)

def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) to SO(3)."""
    th = float(np.linalg.norm(phi))
    if th < 1e-12:
        return np.eye(3)
    K = _skew(phi / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

def so3_log(R: np.ndarray) -> np.ndarray:
    """Log map from SO(3) to so(3) (vee)."""
    cos_th = (np.trace(R) - 1.0) * 0.5
    cos_th = np.clip(cos_th, -1.0, 1.0)
    th = float(np.arccos(cos_th))
    if th < 1e-12:
        return np.zeros(3)
    w = (1.0 / (2.0 * np.sin(th))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    return th * w

def quat_from_axis_angle(omega_world: np.ndarray, dt: float) -> np.ndarray:
    """Quaternion increment from world angular velocity over dt."""
    th = float(np.linalg.norm(omega_world) * dt)
    if th < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = omega_world / (np.linalg.norm(omega_world) + 1e-12)
    s = np.sin(th / 2.0)
    return np.array([np.cos(th / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])

def quat_left_multiply(q: np.ndarray) -> np.ndarray:
    """Left quaternion multiplication matrix L(q) such that L(q) * p == q ⊗ p."""
    w, x, y, z = q
    return np.array(
        [
            [w, -x, -y, -z],
            [x,  w, -z,  y],
            [y,  z,  w, -x],
            [z, -y,  x,  w],
        ],
        dtype=float,
    )

# ------------------------------
# URDF kinematics backend
# ------------------------------
from urdfpy import URDF

class URDFBackend:
    """
    FK via urdfpy; Jacobian via pykin if present, else numeric finite differences.
    End link can be provided; if None, we pick a leaf link (no child joints).
    """
    def __init__(self, urdf_path: str, end_link_name: Optional[str] = None):
        self.robot = URDF.load(urdf_path)
        self.joint_order = [j.name for j in self.robot.actuated_joints]
        self.end_link_name = end_link_name or self._guess_end_link()
        self._use_pykin = False
        try:
            from pykin.robot import Robot as _PyKinRobot          # type: ignore
            from pykin.kinematics import Kinematics as _Kine      # type: ignore
            self._pykin_robot = _PyKinRobot(urdf_path)
            self._pykin_kin = _Kine(self._pykin_robot, self.end_link_name)
            self._use_pykin = True
        except Exception:
            self._use_pykin = False

    def _guess_end_link(self) -> str:
        child_links = {j.child for j in self.robot.joints}
        leaf_links = [lnk.name for lnk in self.robot.links if lnk.name not in {j.parent for j in self.robot.joints}]
        # Prefer the last leaf link (often the tool)
        if leaf_links:
            return leaf_links[-1]
        # Fallback: last link
        return self.robot.links[-1].name

    def _cfg_from_q(self, q: np.ndarray) -> dict:
        return {name: float(val) for name, val in zip(self.joint_order, q)}

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (R_world_ee, p_world_ee)."""
        T_map = self.robot.link_fk(self._cfg_from_q(q))
        T = T_map[self.robot.link_map[self.end_link_name]]
        R, p = T[:3, :3].copy(), T[:3, 3].copy()
        return R, p

    def jacobian6(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """6xN spatial Jacobian (world frame). Prefer pykin; else central-diff FD."""
        if self._use_pykin:
            try:
                J = np.asarray(self._pykin_kin.jacobian(q), dtype=float)  # 6xN
                return J
            except Exception:
                pass
        # Numeric central differences
        R0, p0 = self.fk(q)
        n = q.size
        J = np.zeros((6, n), dtype=float)
        for i in range(n):
            q_plus = q.copy();  q_plus[i] += eps
            q_minus = q.copy(); q_minus[i] -= eps
            R_p, p_p = self.fk(q_plus)
            R_m, p_m = self.fk(q_minus)
            dp = (p_p - p_m) / (2 * eps)
            # Orientation: body log, then map to world
            w_body_p = so3_log(R0.T @ R_p) / eps
            w_body_m = so3_log(R0.T @ R_m) / eps
            w_world = R0 @ ((w_body_p - w_body_m) * 0.5)
            J[:, i] = np.hstack([dp, w_world])
        return J

# ------------------------------
# Damped least-squares pinv
# ------------------------------
def dls_pinv(J: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """J^T (J J^T + lam^2 I)^-1, stable near singularities."""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam * lam) * np.eye(m))

# ------------------------------
# Basic 1st-order admittance
# ------------------------------
class AdmittanceController:
    """
    Minimal 1st-order admittance:
      v_body = k_f * f
      w_body = k_t * tau
    Map to world frame, optionally clip, then dq = J^+ * [v_w; w_w].
    """
    def __init__(self, config: Dict[str, Any]) -> None:
        P = config["parameters"]
        self.dt = float(P.get("dt", 0.01))
        self.end_link_name = P.get("end_link_name", None)
        self.urdf_path = P.get("urdf_string", "meca_robot_with_botaandshaft.urdf")

        A = config.get("admittance", {})
        self.force_gain  = float(A.get("force_gain", 0.01))
        self.torque_gain = float(A.get("torque_gain", 0.05))
        self.damping     = float(A.get("damping", 0.7))
        self.rot_damping = float(A.get("rot_damping", self.damping))
        self.dls_lambda  = float(A.get("dls_lambda", 0.1))
        self.task_clip   = float(A.get("task_clip", 0.10))

        self.kin = URDFBackend(self.urdf_path, self.end_link_name)

    def __call__(self,
                 target_poses: np.ndarray,
                 current_joint_state: np.ndarray,
                 current_joint_torque: np.ndarray,
                 ee_force: np.ndarray,
                 collision: Optional[Any] = None) -> np.ndarray:
        """
        Inputs:
          - target_poses: (unused in minimal controller; kept for interface compatibility)
          - current_joint_state: (N,) joint angles [rad]
          - current_joint_torque: (N,) joint torques (unused)
          - ee_force: (6,) wrench [Fx,Fy,Fz,Tx,Ty,Tz] in body/EE frame
          - collision: optional (unused)
        Returns:
          - dq: (N,) joint velocities [rad/s]
        """
        q = np.asarray(current_joint_state, dtype=float).reshape(-1)
        wrench = np.asarray(ee_force, dtype=float).reshape(6)

        # 1) FK to get world orientation of EE
        R_we, _ = self.kin.fk(q)

        # 2) Minimal first-order admittance in body frame
        f_b  = wrench[:3]
        tau_b = wrench[3:]
        v_b = (self.force_gain  * self.damping)     * f_b
        w_b = (self.torque_gain * self.rot_damping) * tau_b

        # 3) Convert to world frame (Jacobian is world-frame)
        v_w = R_we @ v_b
        w_w = R_we @ w_b
        twist_w = np.hstack([v_w, w_w])

        # Optional magnitude clip (stability / anti-chatter)
        nrm = float(np.linalg.norm(twist_w))
        if self.task_clip > 0 and nrm > self.task_clip:
            twist_w *= (self.task_clip / (nrm + 1e-12))

        # 4) dq via DLS pseudo-inverse
        J = self.kin.jacobian6(q)       # 6xN
        dq = dls_pinv(J, self.dls_lambda) @ twist_w
        return dq

# ------------------------------
# Self-test: read URDF in CWD and "turn wrench into motion"
# ------------------------------
def _default_config(urdf_path: str, end_link: Optional[str] = None) -> Dict[str, Any]:
    return {
        "parameters": {
            "dt": 0.01,
            "end_link_name": end_link,        # None -> auto-detect leaf link
            "urdf_string": urdf_path,
        },
        "admittance": {
            "force_gain":  0.02,
            "torque_gain": 0.06,
            "damping":     0.7,
            "rot_damping": 0.7,
            "dls_lambda":  0.1,
            "task_clip":   0.10,
        },
    }

if __name__ == "__main__":
    # 1) Pick URDF in the current working directory
    urdf_file = "meca_robot_with_botaandshaft.urdf"
    if not os.path.exists(urdf_file):
        # Fallback tiny 2-DoF arm if your file isn't here (for quick smoke test)
        urdf_file = "planar2.urdf"
        with open(urdf_file, "w", encoding="utf-8") as f:
            f.write("""<?xml version="1.0"?><robot name="planar2">
  <link name="base"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2.0"/></joint>
  <link name="l1"><visual><geometry><box size="0.5 0.05 0.05"/></geometry>
    <origin xyz="0.25 0 0" rpy="0 0 0"/></visual></link>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="2.0"/></joint>
  <link name="l2"><visual><geometry><box size="0.5 0.05 0.05"/></geometry>
    <origin xyz="0.25 0 0" rpy="0 0 0"/></visual></link>
</robot>""")
        print("[Info] Using fallback URDF:", urdf_file)
    else:
        print("[Info] Using URDF in CWD:", urdf_file)

    # 2) Build controller
    cfg = _default_config(urdf_file)
    ctrl = AdmittanceController(cfg)

    # 3) Initialize joints (zeros), infer DoF from URDF
    N = len(ctrl.kin.joint_order)
    q = np.zeros(N, dtype=float)
    tau = np.zeros(N, dtype=float)  # unused input (kept for interface)
    dt = cfg["parameters"]["dt"]

    # 4) Define a constant wrench at the EE (Fx,Fy,Fz, Tx,Ty,Tz) in body frame
    ee_wrench = np.array([2.0, 0.0, 0.0,   0.0, 0.0, 0.05], dtype=float)

    # 5) Simulate a short horizon: force -> motion (integrate dq)
    steps = 200
    for t in range(steps):
        dq = ctrl(target_poses=np.zeros(7),
                  current_joint_state=q,
                  current_joint_torque=tau,
                  ee_force=ee_wrench,
                  collision=None)
        q = q + dq * dt
        if (t+1) % 50 == 0:
            print(f"Step {t+1:3d}: |dq|={np.linalg.norm(dq):.4f}, q={np.round(q, 4)}")

    print("Final q (rad):", np.round(q, 6))
