from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass
class ParameterConfig:
    """Typed view of the controller parameters loaded from config files."""

    end_link_name: str
    dt: float
    smooth: float
    wp: float
    wdq: float
    wr: float
    urdf_string: str
    tip_link_name: Optional[str] = None
    ki: float = 0.0
    record: bool = False

    # Optimization-related parameters
    cbf_kappa: float = 0.3
    cbf_manip_min: float = 1e-3
    cbf_eps: float = 1e-12
    rcm_tol: float = 1e-3
    rcm_kp: float = 1.0
    w_dq: float = 0.01
    w_posture: float = 0.0
    # CasADi JIT (codegen) for NLP solver
    jit: bool = True

    # Limits / posture targets (optional; defaults are filled with ndof later)
    q_min: Optional[Sequence[float]] = None
    q_max: Optional[Sequence[float]] = None
    dq_max: Optional[Sequence[float]] = None
    q_posture: Optional[Sequence[float]] = None

    def resolved_tip_link_name(self) -> str:
        """Use tip_link_name if provided; otherwise fall back to end_link_name."""
        return self.tip_link_name or self.end_link_name

    def _array_or_default(self, arr: Optional[Sequence[float]], default_value: float, ndof: int) -> np.ndarray:
        if arr is None:
            return np.array([default_value] * ndof, dtype=float)
        return np.array(arr, dtype=float)

    def get_q_min(self, ndof: int) -> np.ndarray:
        return self._array_or_default(self.q_min, -1e6, ndof)

    def get_q_max(self, ndof: int) -> np.ndarray:
        return self._array_or_default(self.q_max, 1e6, ndof)

    def get_dq_max(self, ndof: int) -> np.ndarray:
        return self._array_or_default(self.dq_max, 1e3, ndof)

    def get_q_posture(self, ndof: int, q_min: Optional[np.ndarray] = None, q_max: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the target posture vector. If not specified in config, default to the midpoint of joint limits.
        """
        if self.q_posture is not None:
            return np.array(self.q_posture, dtype=float)

        q_min_arr = q_min if q_min is not None else self.get_q_min(ndof)
        q_max_arr = q_max if q_max is not None else self.get_q_max(ndof)
        q_mid = [0.5 * (lo + hi) for lo, hi in zip(q_min_arr, q_max_arr)]
        return np.array(q_mid, dtype=float)


@dataclass
class ControllerConfig:
    """Encapsulates all controller-related configuration."""

    parameters: ParameterConfig
    admittance: Dict[str, Any] = field(default_factory=dict)


def _require(params: Dict[str, Any], key: str) -> Any:
    if key not in params:
        raise KeyError(f"Config missing required parameters['{key}']")
    return params[key]


def parse_controller_config(raw_config: Dict[str, Any]) -> ControllerConfig:
    """
    Convert raw dict config (e.g., from YAML/JSON) to typed ControllerConfig.

    This keeps defaults aligned with the original controller logic.
    """
    if isinstance(raw_config, ControllerConfig):
        return raw_config

    params = raw_config.get("parameters", {})

    param_cfg = ParameterConfig(
        end_link_name=_require(params, "end_link_name"),
        dt=_require(params, "dt"),
        smooth=_require(params, "smooth"),
        wp=_require(params, "wp"),
        wdq=_require(params, "wdq"),
        wr=_require(params, "wr"),
        urdf_string=_require(params, "urdf_string"),
        tip_link_name=params.get("tip_link_name"),
        ki=params.get("ki", 0.0),
        record=bool(params.get("record", False)),
        cbf_kappa=float(params.get("cbf_kappa", 0.3)),
        cbf_manip_min=float(params.get("cbf_manip_min", 1e-3)),
        cbf_eps=float(params.get("cbf_eps", 1e-12)),
        rcm_tol=float(params.get("rcm_tol", 1e-4)),
        rcm_kp=float(params.get("rcm_kp", 1.0)),
        w_dq=float(params.get("w_dq", 0.1)),
        w_posture=float(params.get("w_posture", 0.0)),
        jit=bool(params.get("jit", True)),
        q_min=params.get("q_min"),
        q_max=params.get("q_max"),
        dq_max=params.get("dq_max"),
        q_posture=params.get("q_posture"),
    )

    admittance_cfg = raw_config.get("admittance", {}) or {}

    return ControllerConfig(parameters=param_cfg, admittance=admittance_cfg)
