import optas
from optas.spatialmath import *
import numpy as np
import logging
from typing import Any, Dict, Optional, Sequence, Callable, Tuple, List

import os
from pathlib import Path
try:
    from ament_index_python import get_package_share_directory
except Exception:
    get_package_share_directory = None
import csv

try:
    from .config_schema import ControllerConfig, ParameterConfig, parse_controller_config  # type: ignore
    from .qp_builder import TwistTrackingQPBuilder  # type: ignore
except ImportError:
    from config_schema import ControllerConfig, ParameterConfig, parse_controller_config  # type: ignore
    from qp_builder import TwistTrackingQPBuilder  # type: ignore


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])


def format_vector(vec: Optional[np.ndarray], unit: Optional[str] = None) -> str:
    
    if vec is None:
        return "[]"
    arr = np.asarray(vec).reshape(-1)
    if arr.size == 0:
        return "[]"
    if unit == "deg":
        arr = np.rad2deg(arr)
    formatted = ", ".join(f"{v:+.4f}" for v in arr)
    return f"[{formatted}]"


def _normalize_axis_vector(axis: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    
    if axis is None:
        return None
    v = np.asarray(axis, dtype=float).reshape(-1)
    if v.size == 0:
        return None
    if v.size != 3:
        raise ValueError(f"Axis vector must have length 3, got {v.size}")
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    return v / n


# ====================================================================== #
# Translated note in English.
# ====================================================================== #
class TwistTrackingLoss:
    

    def __init__(
        self,
        initial_weights: Sequence[float] = (1.0, 1.0, 1.0),
        param_name: str = "loss_weights",
        redundant_pos_axis: Optional[Sequence[float]] = None,
        redundant_rot_axis: Optional[Sequence[float]] = None,
    ) -> None:
        self.param_name = param_name
        self._weights = np.asarray(initial_weights, dtype=float).flatten()
        if self._weights.size != 3:
            raise ValueError("initial_weights must have length 3")

        # Translated note in English.
        self.redundant_pos_axis = _normalize_axis_vector(redundant_pos_axis)
        self.redundant_rot_axis = _normalize_axis_vector(redundant_rot_axis)

        # Translated note in English.
        self.param_symbol = None  # CasADi MX
        self._features = None     # CasADi MX

    def build(
        self,
        builder: "optas.OptimizationBuilder",
        dq,
        dp,
        twist_des,
        tip_rot=None,
    ) -> None:
        
        if tip_rot is None:
            tip_rot = optas.eye(3)

        # Translated note in English.
        v_err_world = dp[:3] - twist_des[:3]
        w_err_world = dp[3:] - twist_des[3:]

        # Translated note in English.
        v_err_tip = optas.mtimes(tip_rot.T, v_err_world)
        w_err_tip = optas.mtimes(tip_rot.T, w_err_world)

        # Translated note in English.
        if self.redundant_pos_axis is not None:
            axis_v = optas.DM(self.redundant_pos_axis.reshape((3, 1)))
            v_err_tip = v_err_tip - axis_v * optas.mtimes(axis_v.T, v_err_tip)
        if self.redundant_rot_axis is not None:
            axis_w = optas.DM(self.redundant_rot_axis.reshape((3, 1)))
            w_err_tip = w_err_tip - axis_w * optas.mtimes(axis_w.T, w_err_tip)

        # Translated note in English.
        self._features = optas.vertcat(
            optas.sumsqr(v_err_tip),
            optas.sumsqr(w_err_tip),
            optas.sumsqr(dq),
        )

        # Translated note in English.
        self.param_symbol = builder.add_parameter(self.param_name, 3)

        cost_expr = (
            self.param_symbol[0] * self._features[0]
            + self.param_symbol[1] * self._features[1]
            + self.param_symbol[2] * self._features[2]
        )
        builder.add_cost_term("pref_loss", cost_expr)

        # Translated note in English.





    # Translated note in English.
    def get_weights(self) -> np.ndarray:
        # Translated note in English.
        return self._weights.copy()

    def set_weights(self, weights: np.ndarray) -> None:
        # Translated note in English.
        w = np.asarray(weights, dtype=float).flatten()
        if w.size != 3:
            raise ValueError(f"loss weight dim must be 3, got {w.size}")
        self._weights = w


# ====================================================================== #
# Translated note in English.
# ====================================================================== #
class SecondOrderAdmittance:
    

    def __init__(
        self,
        admittance_cfg: Dict[str, Any],
        dt: float,
        jacobian_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.dt = dt

        # Translated note in English.
        self.force_gain = admittance_cfg.get("force_gain", 0.01)  # Translated note in English.
        self.torque_gain = admittance_cfg.get("torque_gain", 0.1)
        self.damping = admittance_cfg.get("damping", 0.7)
        self.rot_damping = admittance_cfg.get("rot_damping", self.damping)

        self.mass = admittance_cfg.get("mass", 0.1)
        self.rot_mass = max(admittance_cfg.get("rot_mass", 0.1), 1e-3)

        # Translated note in English.
        # compliance_offset = [x, y, z] = p_SC
        # Translated note in English.
        self.compliance_offset = np.array(
            admittance_cfg.get("compliance_offset", [0.0, 0.0, 0.0]),
            dtype=float,
        )

        # Translated note in English.
        adm_redundant_pos = admittance_cfg.get(
            "reduandant_pos_axis",
            admittance_cfg.get("redundant_pos_axis", None),
        )

        # Translated note in English.
        adm_redundant_pos = admittance_cfg.get(
            "reduandant_pos_axis",
            admittance_cfg.get("redundant_pos_axis", None),
        )
        adm_redundant_rot = admittance_cfg.get(
            "reduandant_rot_axis",
            admittance_cfg.get("redundant_rot_axis", None),
        )
        self.redundant_pos_axis = _normalize_axis_vector(adm_redundant_pos)
        self.redundant_rot_axis = _normalize_axis_vector(adm_redundant_rot)

        # Translated note in English.
        self.enable_manipulability_scaling = admittance_cfg.get(
            "enable_manipulability_scaling", True
        )
        self.manipulability_target = admittance_cfg.get("manipulability_target", 0.05)
        self.min_manipulability_scale = admittance_cfg.get(
            "min_manipulability_scale", 0.2
        )
        self.max_manipulability_scale = admittance_cfg.get(
            "max_manipulability_scale", 1.0
        )
        self.manipulability_epsilon = admittance_cfg.get("manipulability_epsilon", 1e-4)
        self.debug_jitter = admittance_cfg.get("debug_jitter", False)

        # Translated note in English.
        self.jacobian_func = jacobian_func

        # Translated note in English.
        self.current_vel = np.zeros(3)
        self.current_ang_vel = np.zeros(3)
        self.prev_angle_vel = np.zeros(3)

    # Translated note in English.
    def reset_state(self) -> None:
        self.current_vel[:] = 0.0
        self.current_ang_vel[:] = 0.0
        self.prev_angle_vel[:] = 0.0

    def get_param_vector(self) -> np.ndarray:
        
        return np.array(
            [
                self.force_gain,
                self.torque_gain,
                self.mass,
                self.rot_mass,
                self.damping,
                self.rot_damping,
            ],
            dtype=float,
        )

    def set_param_vector(self, theta: np.ndarray) -> None:
        
        theta = np.asarray(theta, dtype=float).flatten()
        if theta.size != 6:
            raise ValueError(f"admittance param dim must be 6, got {theta.size}")

        (
            self.force_gain,
            self.torque_gain,
            self.mass,
            self.rot_mass,
            self.damping,
            self.rot_damping,
        ) = theta

        # Translated note in English.
        self.mass = max(self.mass, 1e-6)
        self.rot_mass = max(self.rot_mass, 1e-6)
        self.damping = max(self.damping, 0.0)
        self.rot_mass = max(self.rot_mass, 1e-6)
        self.rot_damping = max(self.rot_damping, 0.0)

    # Translated note in English.
    def compute_rotational_manipulability_gain(self, joint_state: np.ndarray) -> np.ndarray:
        if (not self.enable_manipulability_scaling) or (self.jacobian_func is None):
            return np.eye(3)

        try:
            J = np.array(self.jacobian_func(joint_state))
            J_rot = J[3:6, :]
            if J_rot.size == 0:
                return np.eye(3)

            U, singular_values, _ = np.linalg.svd(J_rot, full_matrices=False)
            if singular_values.size == 0:
                return np.eye(3)

            scales = singular_values / (self.manipulability_target + self.manipulability_epsilon)
            scales = np.clip(scales, self.min_manipulability_scale, self.max_manipulability_scale)
            return U @ np.diag(scales) @ U.T

        except Exception as exc:
            logging.warning("Manipulability gain computation failed: %s", exc)
            return np.eye(3)

    # Translated note in English.
    # Translated note in English.
    def step(
        self,
        wrench: np.ndarray,
        rot_gain: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if rot_gain is None:
            rot_gain = np.eye(3)

        # Translated note in English.
        force_S = np.asarray(wrench[:3], dtype=float)
        torque_S = np.asarray(wrench[3:], dtype=float)

        # Translated note in English.
        force_eff = force_S.copy()
        if self.redundant_pos_axis is not None:
            # F_eff = F - (a·F) a
            f_parallel = float(np.dot(self.redundant_pos_axis, force_eff))
            force_eff = force_eff - f_parallel * self.redundant_pos_axis

        # Translated note in English.
        p_SC = self.compliance_offset
        torque_C = torque_S - np.cross(p_SC, force_eff)

        # Translated note in English.
        torque_eff = torque_C.copy()
        if self.redundant_rot_axis is not None:
            # τ_eff = τ - (a·τ) a
            t_parallel = float(np.dot(self.redundant_rot_axis, torque_eff))
            torque_eff = torque_eff - t_parallel * self.redundant_rot_axis

        # Translated note in English.
        force = force_eff
        torque = torque_eff

        # Translated note in English.
        acceleration = (self.force_gain * force - self.damping * self.current_vel) / self.mass
        new_vel = self.current_vel + acceleration * self.dt

        # Translated note in English.
        scaled_torque = torque * self.torque_gain
        angular_acc = (scaled_torque - self.rot_damping * self.current_ang_vel) / self.rot_mass
        predicted_ang_vel = self.current_ang_vel + angular_acc * self.dt
        angle_vel = rot_gain @ predicted_ang_vel

        if self.debug_jitter:
            jitter = np.linalg.norm(angle_vel - self.prev_angle_vel) / max(self.dt, 1e-6)
            try:
                eigvals = np.linalg.eigvalsh(rot_gain)
            except np.linalg.LinAlgError:
                eigvals = np.array([-1.0, -1.0, -1.0])
            angle_str = format_vector(angle_vel, unit="deg")
            gain_str = format_vector(eigvals)
            jitter_deg = float(np.rad2deg(jitter))
            # logging.info(...)

        self.prev_angle_vel = angle_vel
        self.current_vel = new_vel
        self.current_ang_vel = angle_vel

        # Translated note in English.
        if self.force_gain == 0.0:
            self.current_vel[:] = 0.0
            new_vel = self.current_vel.copy()
        if self.torque_gain == 0.0:
            self.current_ang_vel[:] = 0.0
            self.prev_angle_vel[:] = 0.0
            angle_vel = self.current_ang_vel.copy()

        return new_vel, angle_vel


# ====================================================================== #
# Translated note in English.
# ====================================================================== #
class IntegratedAdmittanceController:
    

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def __init__(self, config: Dict[str, Any]) -> None:
        # Translated note in English.
        self.config = config
        self.controller_config: ControllerConfig = parse_controller_config(config)
        self._ensure_urdf_path()

        self._init_parameters()

        # Translated note in English.
        # Translated note in English.
        self.loss_model = TwistTrackingLoss(
            initial_weights=[self.wp, self.wr, self.wdq],
            redundant_pos_axis=self.redundant_pos_axis,
            redundant_rot_axis=self.redundant_rot_axis,
        )

        self._setup_robot_model()
        self._init_kinematics()
        self._init_admittance()
        self._build_optimization_problem()
        self._init_state()
        self.nn_model = self.config.get("nn_model") if isinstance(self.config, dict) else None

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def _ensure_urdf_path(self) -> None:
        
        params_cfg = self.controller_config.parameters
        if params_cfg.urdf_string != "load_in_python_script":
            return

        try:
            urdf_path = None
            if get_package_share_directory is not None:
                try:
                    urdf_path = os.path.join(
                        get_package_share_directory("mecademic_description"),
                        "urdf",
                        "meca_robot_with_botaandshaft.urdf",
                    )
                except Exception:
                    urdf_path = None

            if urdf_path is None:
                local_candidate = (
                    Path(__file__).resolve().parents[1]
                    / "urdf"
                    / "meca_robot_with_botaandshaft.urdf"
                )
                if not local_candidate.exists():
                    raise FileNotFoundError(local_candidate)
                urdf_path = str(local_candidate)

            params_cfg.urdf_string = urdf_path
            # Translated note in English.
            if isinstance(self.config, dict):
                self.config.setdefault("parameters", {})["urdf_string"] = urdf_path
        except Exception as exc:
            raise FileNotFoundError("load_in_python_script") from exc

    def _init_parameters(self) -> None:
        
        params: ParameterConfig = self.controller_config.parameters
        self.parameters_cfg = params
        self.end_link_name = params.end_link_name

        # Translated note in English.
        # Translated note in English.
        self.tip_link_name = params.resolved_tip_link_name()

        self.dt = params.dt
        self.smooth = params.smooth
        self.wp = params.wp  # Translated note in English.
        self.wdq = params.wdq  # Translated note in English.
        self.wr = params.wr  # Translated note in English.
        self.ki = params.ki  # Translated note in English.
        self.urdf_string = params.urdf_string

        # Translated note in English.
        # Translated note in English.
        self.record = bool(params.record)
        self._record_buffer: List[Dict[str, Any]] = []

        # Translated note in English.
        self.cbf_kappa = params.cbf_kappa
        self.cbf_manip_min = params.cbf_manip_min
        self.cbf_eps = params.cbf_eps
        self.rcm_tol = params.rcm_tol
        self.rcm_kp = params.rcm_kp
        self.w_dq = params.w_dq
        self.w_posture = params.w_posture

        # Translated note in English.
        admittance_cfg = self.controller_config.admittance
        pos_axis_cfg = admittance_cfg.get(
            "reduandant_pos_axis",
            admittance_cfg.get("redundant_pos_axis", None),
        )
        rot_axis_cfg = admittance_cfg.get(
            "reduandant_rot_axis",
            admittance_cfg.get("redundant_rot_axis", None),
        )
        self.redundant_pos_axis = _normalize_axis_vector(pos_axis_cfg)
        self.redundant_rot_axis = _normalize_axis_vector(rot_axis_cfg)

    def _setup_robot_model(self) -> None:
        
        self.robot = optas.RobotModel(
            xacro_filename=self.urdf_string,
            time_derivs=[1],  # Include joint velocity
        )
        self.name = self.robot.get_name()
        self.ndof = self.robot.ndof

    def _init_kinematics(self) -> None:
        
        self.base_link = "meca_base_link"  # Translated note in English.

        # Translated note in English.
        self.ee_link_pos_func = self.robot.get_link_position_function(
            self.end_link_name, self.base_link
        )
        self.ee_link_rot_func = self.robot.get_link_rotation_function(
            self.end_link_name, self.base_link
        )

        # Translated note in English.
        self.tip_link_rot_func = self.robot.get_link_rotation_function(
            self.tip_link_name, self.base_link
        )

        # Translated note in English.
        try:
            self.jacobian_func = (
                self.robot.get_global_link_geometric_jacobian_function(
                    self.end_link_name
                )
            )
        except Exception as exc:
            logging.warning(
                "Unable to create Jacobian function for manipulability scaling: %s", exc
            )
            self.jacobian_func = None


    def _init_admittance(self) -> None:
        # Translated note in English.
        admittance_cfg = self.controller_config.admittance
        self.admittance = SecondOrderAdmittance(
            admittance_cfg,
            dt=self.dt,
            jacobian_func=getattr(self, "jacobian_func", None),
        )
        self.base_damping = float(self.admittance.damping)
        self.base_rot_damping = float(self.admittance.rot_damping)

    def _init_state(self) -> None:
        
        # Translated note in English.
        self.dq = np.zeros(self.ndof)
        # Translated note in English.
        self.admittance.reset_state()
        # Translated note in English.
        self._last_step_data: Dict[str, Any] = {}

    # Translated note in English.
    def get_loss_weights(self) -> np.ndarray:
        # Translated note in English.
        return self.loss_model.get_weights()

    def set_loss_weights(self, weights: np.ndarray) -> None:
        # Translated note in English.
        self.loss_model.set_weights(weights)

    def get_admittance_params(self) -> np.ndarray:
        # Translated note in English.
        return self.admittance.get_param_vector()

    def set_admittance_params(self, theta: np.ndarray) -> None:
        # Translated note in English.
        self.admittance.set_param_vector(theta)

    # Translated note in English.
    def get_record_buffer(self) -> List[Dict[str, Any]]:
        # Translated note in English.
        return self._record_buffer

    def clear_record_buffer(self) -> None:
        # Translated note in English.
        self._record_buffer = []

    def get_last_step_data(self) -> Dict[str, Any]:
        # Translated note in English.
        data = getattr(self, "_last_step_data", None)
        if not data:
            return {}

        copied: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                copied[key] = value.copy()
            else:
                copied[key] = value
        return copied

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def _compute_rotational_manipulability_gain(
        self, joint_state: np.ndarray
    ) -> np.ndarray:
        
        return self.admittance.compute_rotational_manipulability_gain(joint_state)

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def _compute_numeric_manipulability(self, joint_state: np.ndarray) -> Optional[float]:
        # Translated note in English.
        if self.jacobian_func is None:
            return None
        try:
            J = np.array(self.jacobian_func(joint_state))  # 6 x ndof
            Jv = J[:3, :]
            JJ = Jv @ Jv.T
            det_JJ = float(np.linalg.det(JJ))
            safe_det = max(det_JJ, 0.0)
            return float(np.sqrt(safe_det + self.cbf_eps))
        except Exception as exc:
            logging.debug("Failed to compute numeric manipulability: %s", exc)
            return None

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def gen_motion_from_wrench(
        self,
        wrench: np.ndarray,
        rot_gain: Optional[np.ndarray] = None,
    ):
        
        return self.admittance.step(wrench, rot_gain)


    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def _build_optimization_problem(self) -> None:
        
        self.qp_builder = TwistTrackingQPBuilder(
            robot=self.robot,
            loss_model=self.loss_model,
            parameters=self.parameters_cfg,
            base_link=self.base_link,
            tip_link_name=self.tip_link_name,
            end_link_name=self.end_link_name,
            dt=self.dt,
        )
        self.qp_builder.build()

        # Translated note in English.
        self.builder = self.qp_builder.builder
        self.solver = self.qp_builder.solver
        self.solution = self.qp_builder.solution

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def compute_target_velocity(
        self,
        qc: np.ndarray,
        twist_des: np.ndarray,
        dq_null_ref: Optional[np.ndarray] = None,
        w_null: float = 0.0,
    ) -> np.ndarray:
        dq_null_ref = (
            np.zeros(self.ndof, dtype=float)
            if dq_null_ref is None
            else np.asarray(dq_null_ref, dtype=float).reshape(self.ndof,)
        )

        seed = self.qp_builder.get_seed(qc)
        dq = self.qp_builder.solve(
            qc=qc,
            twist_des=twist_des,
            loss_weights=self.loss_model.get_weights(),
            dq_prev=self.dq,
            dq_null_ref=dq_null_ref,
            w_null=w_null,
            seed=seed,
        )

        # Translated note in English.
        self.solution = self.qp_builder.solution
        self.solver = self.qp_builder.solver
        return dq


    def tracking(
        self,
        twist_des: np.ndarray,
        current_joint_state: np.ndarray,
        current_joint_torque: np.ndarray,
        collision: Optional[Any] = None,
        dq_null_ref: Optional[np.ndarray] = None,
        w_null: float = 0.0,
    ) -> np.ndarray:
        
        self._validate_inputs(current_joint_state, current_joint_torque)

        dq = self.compute_target_velocity(
            current_joint_state,
            twist_des,
            dq_null_ref=dq_null_ref,
            w_null=w_null,
        )

        # Translated note in English.
        self.dq = self.dq * self.smooth + (1.0 - self.smooth) * dq
        return self.dq


    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def _validate_inputs(self, q: np.ndarray, tau_ext: np.ndarray) -> None:
        
        if tau_ext.size != self.ndof or q.size != self.ndof:
            raise BufferError(
                f"Expected joint position and torque with {self.ndof} dof, "
                f"got {q.size} and {tau_ext.size} dof."
            )

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def _record_step(
        self,
        current_joint_state: np.ndarray,
        current_joint_torque: np.ndarray,
        ee_force: np.ndarray,
        vel: np.ndarray,
        ang_vel: np.ndarray,
        twist_des: np.ndarray,
        dq: np.ndarray,
    ) -> None:
        # Translated note in English.
        if not self.record:
            return

        try:
            ee_pos = np.array(self.ee_link_pos_func(current_joint_state))
            ee_rot = np.array(self.ee_link_rot_func(current_joint_state))
            ee_quat = rot_to_quat(ee_rot)
        except Exception as exc:
            logging.debug("Logging: failed to compute EE pose: %s", exc)
            ee_pos, ee_quat = None, None

        manipulability = self._compute_numeric_manipulability(current_joint_state)

        try:
            record = {
                "q": current_joint_state.copy(),
                "tau_ext": current_joint_torque.copy(),
                "dq_cmd": dq.copy(),
                "ee_force": ee_force.copy(),
                "vel": vel.copy(),
                "ang_vel": ang_vel.copy(),
                "twist_des": twist_des.copy(),
                "ee_pos": ee_pos,
                "ee_quat": ee_quat,
                "manipulability": manipulability,
                "admittance_params": self.admittance.get_param_vector().copy(),
                "loss_weights": self.loss_model.get_weights().copy(),
            }
            self._record_buffer.append(record)
        except Exception as exc:
            # Translated note in English.
            logging.warning("Failed to append record: %s", exc)

    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        target_poses: np.ndarray,  # Translated note in English.
        current_joint_state: np.ndarray,
        current_joint_torque: np.ndarray,
        ee_force: np.ndarray,
        collision: Optional[Any] = None,
        rl_cmd: Optional[np.ndarray] = None,  # Translated note in English.
    ) -> np.ndarray:
        

        # Translated note in English.
        # rot_gain = self._compute_rotational_manipulability_gain(current_joint_state)
        rot_gain = np.eye(3)  # Translated note in English.

        # Translated note in English.
        dq_null_ref = np.zeros(self.ndof, dtype=float)
        damping_scales = np.ones(6, dtype=float)
        w_null = 0.0

        if self.nn_model is not None:
            try:
                manipulability = self._compute_numeric_manipulability(current_joint_state)
                manip_value = 0.0 if manipulability is None else float(manipulability)

                nn_input = np.hstack(
                    [current_joint_state, ee_force, [manip_value], np.zeros(3, dtype=float)]
                )
                nn_outputs = self.nn_model.predict(nn_input)
                dq_null_ref, damping_scales, w_null = nn_outputs

                dq_null_ref = np.asarray(dq_null_ref, dtype=float).reshape(self.ndof,)
                damping_scales = np.asarray(damping_scales, dtype=float).reshape(6,)
                w_null = float(np.asarray(w_null).reshape(-1)[0])
            except Exception as exc:
                logging.warning("NN guidance failed: %s", exc)

        # Translated note in English.
        linear_scale = float(np.mean(damping_scales[:3]))
        rot_scale = float(np.mean(damping_scales[3:]))
        self.admittance.damping = max(self.base_damping * linear_scale, 0.0)
        self.admittance.rot_damping = max(self.base_rot_damping * rot_scale, 0.0)

        # Translated note in English.
        vel, ang_vel = self.gen_motion_from_wrench(ee_force, rot_gain)

        # Translated note in English.
        current_rot_tip = np.array(self.tip_link_rot_func(current_joint_state))
        R = current_rot_tip   # tip_link -> world
        v_s = R @ vel
        w_s = R @ ang_vel
        twist_des = np.concatenate([v_s, w_s])

        if rl_cmd is not None:
            # Translated note in English.
            twist_des = twist_des + rl_cmd

        # Translated note in English.
        dq = self.tracking(
            twist_des,
            current_joint_state,
            current_joint_torque,
            collision,
            dq_null_ref=dq_null_ref,
            w_null=w_null,
        )

        # Translated note in English.
        try:
            self._last_step_data = {
                "vel": vel.copy(),
                "ang_vel": ang_vel.copy(),
                "twist_des": twist_des.copy(),
                "dq_cmd": dq.copy(),
            }
        except Exception as exc:
            logging.debug("Failed to stash last step data: %s", exc)

        return dq



    # ------------------------------------------------------------------ #
    # Translated note in English.
    # ------------------------------------------------------------------ #
    def calculate_interaction_point(
        self,
        force: np.ndarray,
        torque: np.ndarray,
    ):
        # Translated note in English.
        force_norm = np.linalg.norm(force)
        # Translated note in English.

        if force_norm > 5e-1:  # Translated note in English.
            interaction_point = np.cross(torque, force) / (force_norm**2)
        else:
            interaction_point = None  # Translated note in English.

        return interaction_point
    
    def save_records_to_csv(self, filename: str = "admittance_log.csv") -> None:
        
        if not self.record:
            logging.warning("save_records_to_csv called but record flag is False.")
        if not self._record_buffer:
            logging.warning("save_records_to_csv: no records to save.")
            return

        # Translated note in English.
        keys = [
            "q",
            "tau_ext",
            "dq_cmd",
            "ee_force",
            "vel",
            "ang_vel",
            "twist_des",
            "ee_pos",
            "ee_quat",
            "manipulability",
            "admittance_params",
            "loss_weights",
        ]

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            # Translated note in English.
            writer.writerow(keys)

            for rec in self._record_buffer:
                row = []
                for k in keys:
                    v = rec.get(k, None)
                    if isinstance(v, np.ndarray):
                        # Translated note in English.
                        row.append(" ".join(map(str, v.flatten())))
                    else:
                        # Translated note in English.
                        row.append("" if v is None else str(v))
                writer.writerow(row)

        logging.info("Saved %d records to CSV file: %s", len(self._record_buffer), filename)

    def __del__(self):
        
        try:
            if getattr(self, "record", False) and getattr(self, "_record_buffer", None):
                # Translated note in English.
                self.save_records_to_csv()
        except Exception as exc:
            # Translated note in English.
            logging.warning("Failed to save records in __del__: %s", exc)
