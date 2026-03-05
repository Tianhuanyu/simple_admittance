from typing import Any, Dict, Optional, Sequence, Tuple

import casadi as cs
import numpy as np
import optas

# Allow running both as a package module and as a standalone script.
try:
    from .config_schema import ParameterConfig  # type: ignore
except ImportError:
    from config_schema import ParameterConfig  # type: ignore


class TwistTrackingQPBuilder:
    

    def __init__(
        self,
        robot: optas.RobotModel,
        loss_model: Any,
        parameters: ParameterConfig,
        *,
        base_link: str,
        tip_link_name: str,
        end_link_name: str,
        dt: float,
    ) -> None:
        self.robot = robot
        self.loss_model = loss_model
        self.params = parameters
        self.base_link = base_link
        self.tip_link_name = tip_link_name
        self.end_link_name = end_link_name
        self.dt = dt

        self.name = self.robot.get_name()
        self.ndof = self.robot.ndof

        self.builder: Optional[optas.OptimizationBuilder] = None
        self.solver = None
        self.solution = None

        # Placeholders for symbolic handles
        self.dq_prev = None
        self.qc = None
        self.twist_des = None
        self.dq_null_ref = None
        self.w_null = None
        self.J = None

    @staticmethod
    def _get_array_param(params: ParameterConfig, key: str, default: Sequence[float], ndof: int) -> np.ndarray:
        
        if key == "q_min":
            return params.get_q_min(ndof) if params.q_min is None else np.array(params.q_min, dtype=float)
        if key == "q_max":
            return params.get_q_max(ndof) if params.q_max is None else np.array(params.q_max, dtype=float)
        if key == "dq_max":
            return params.get_dq_max(ndof) if params.dq_max is None else np.array(params.dq_max, dtype=float)
        return np.array(default, dtype=float)
    
    def build(self) -> None:
        
        P = self.params

        # Translated note in English.
        self.cbf_kappa = float(P.cbf_kappa)
        self.cbf_manip_min = float(P.cbf_manip_min)
        self.cbf_eps = float(P.cbf_eps)
        self.w_dq = float(P.w_dq)
        self.w_posture = float(P.w_posture)
        self.jit = bool(getattr(P, "jit", True))

        # Translated note in English.
        # Translated note in English.
        q_lo_urdf = np.array(self.robot.lower_actuated_joint_limits.toarray()).flatten()
        q_hi_urdf = np.array(self.robot.upper_actuated_joint_limits.toarray()).flatten()

        # Translated note in English.
        dq_lim_urdf = np.array(self.robot.velocity_actuated_joint_limits.toarray()).flatten()

        # Translated note in English.
        q_min_cfg = self._get_array_param(P, "q_min", q_lo_urdf, self.ndof)
        q_max_cfg = self._get_array_param(P, "q_max", q_hi_urdf, self.ndof)
        dq_max_cfg = self._get_array_param(P, "dq_max", dq_lim_urdf, self.ndof)

        self.q_min = np.maximum(q_lo_urdf, q_min_cfg)
        self.q_max = np.minimum(q_hi_urdf, q_max_cfg)
        self.dq_max = np.minimum(dq_lim_urdf, dq_max_cfg)

        # Translated note in English.
        self.q_posture = P.get_q_posture(self.ndof, self.q_min, self.q_max)

        # Translated note in English.
        self.builder = optas.OptimizationBuilder(1)

        # parameters
        self.dq_prev = self.builder.add_parameter("dq_prev", self.ndof)
        self.qc = self.builder.add_parameter("qc", self.ndof)
        self.twist_des = self.builder.add_parameter("twist_des", 6)

        self.dq_null_ref = self.builder.add_parameter("dq_null_ref", self.ndof)  # Translated note in English.
        self.w_null = self.builder.add_parameter("w_null", 1)  # Translated note in English.

        # decision variables
        ddq = self.builder.add_decision_variables("ddq", self.ndof)

        # predicted velocity / position
        dq = self.dq_prev + self.dt * ddq
        q_next = self.qc + self.dt * dq

        # --------- 3. Twist Tracking Cost --------- #
        self.J = self.robot.get_global_link_geometric_jacobian(self.end_link_name, self.qc)
        dp = self.J @ dq

        try:
            tip_rot_qc = self.robot.get_link_rotation(self.tip_link_name, self.qc, self.base_link)
        except TypeError:
            tip_rot_qc = self.robot.get_link_rotation(self.tip_link_name, self.qc)

        self.loss_model.build(
            self.builder,
            dq,
            dp,
            self.twist_des,
            tip_rot=tip_rot_qc,
        )

        # Translated note in English.
        # Translated note in English.
        self.builder.add_cost_term("base_reg", 1e-5 * cs.sumsqr(dq))
        # Translated note in English.
        self.builder.add_cost_term(
            "global_guidance",
            self.w_null[0] * cs.sumsqr(dq - self.dq_null_ref),
        )

        # Translated note in English.
        # Translated note in English.
        for i in range(self.ndof):
            self.builder.add_leq_inequality_constraint(f"vel_max_{i}", dq[i], self.dq_max[i])
            self.builder.add_leq_inequality_constraint(f"vel_min_{i}", -dq[i], self.dq_max[i])

        # Translated note in English.
        for i in range(self.ndof):
            self.builder.add_leq_inequality_constraint(f"qmax_{i}", q_next[i], self.q_max[i])
            self.builder.add_leq_inequality_constraint(f"qmin_{i}", -q_next[i], -self.q_min[i])

        # Translated note in English.
        if self.w_posture > 0.0:
            q_posture_dm = cs.DM(self.q_posture)
            posture_err = q_next - q_posture_dm
            self.builder.add_cost_term(
                "posture_regularization",
                self.w_posture * cs.sumsqr(posture_err[1:]),
            )

        # Translated note in English.
        self._init_solver()




    def _init_solver(self) -> None:
        
        assert self.builder is not None
        optimization = self.builder.build()

        solver_options = {
            "print_time": False,
            "print_header": False,
            "print_iteration": False,
            "print_status": False,
            "print_in": False,
            "print_out": False,
            "verbose": False,
            "verbose_init": False,
            "qpsol_options": {"printLevel": "none"},
        }
        if self.jit:
            solver_options["jit"] = True

        try:
            self.solver = optas.CasADiSolver(optimization).setup("sqpmethod", solver_options)
        except Exception as exc:
            if self.jit:
                solver_options.pop("jit", None)
                self.solver = optas.CasADiSolver(optimization).setup("sqpmethod", solver_options)
            else:
                raise exc
        self.solution = None

    def initial_seed(self, qc: np.ndarray) -> Dict[str, Any]:
        
        return {
            "ddq": optas.DM.zeros(self.ndof, 1),
        }



    def get_seed(self, qc: np.ndarray) -> Dict[str, Any]:
        
        if self.solution is not None:
            return self.solution
        return self.initial_seed(qc)

    def solve(
        self,
        *,
        qc: np.ndarray,
        twist_des: np.ndarray,
        loss_weights: np.ndarray,
        dq_prev: np.ndarray,
        dq_null_ref: np.ndarray,
        w_null: float,
        seed: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        
        seed_to_use = seed if seed is not None else self.get_seed(qc)
        self.solver.reset_initial_seed(seed_to_use)

        self.solver.reset_parameters({
            "qc": optas.DM(qc),
            "twist_des": optas.DM(twist_des),
            self.loss_model.param_name: optas.DM(loss_weights),
            "dq_prev": optas.DM(dq_prev),
            "dq_null_ref": optas.DM(dq_null_ref),
            "w_null": optas.DM([w_null]),
        })

        self.solution = self.solver.solve()

        if self.solver.did_solve():
            ddq_sol = self.solution["ddq"].toarray().flatten()
            dq_cmd = dq_prev + self.dt * ddq_sol
            return dq_cmd

        # Translated note in English.
        return np.zeros(self.ndof)
