import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from hardware.ftsensor import CustomBotaSerialSensor
from controller.integrated_admittance_controller import IntegratedAdmittanceController


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "config" / "config_admit.yaml"


def _load_and_resolve_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    params = dict(config.get("parameters", {}))
    urdf_str = str(params.get("urdf_string", "")).strip()
    if urdf_str == "":
        raise ValueError("parameters.urdf_string is required")
    if not Path(urdf_str).is_absolute():
        params["urdf_string"] = str((config_path.parent / urdf_str).resolve())
    config["parameters"] = params
    return config


def _init_joint_state(config: dict) -> np.ndarray:
    params = config.get("parameters", {})
    q_min = np.asarray(params.get("q_min", [-1.0] * 6), dtype=float)
    q_max = np.asarray(params.get("q_max", [1.0] * 6), dtype=float)
    return 0.5 * (q_min + q_max)


def run_sim_loop(config: dict, steps: int, print_every: int) -> None:
    controller = IntegratedAdmittanceController(config)
    params = config.get("parameters", {})
    dt = float(params.get("dt", 0.01))
    q = _init_joint_state(config)
    tau_ext = np.zeros_like(q)
    q_min = np.asarray(params.get("q_min", [-10.0] * q.size), dtype=float)
    q_max = np.asarray(params.get("q_max", [10.0] * q.size), dtype=float)

    infinite_mode = steps <= 0
    print("Standalone optas admittance demo started (sim)")
    print(f"steps={'infinite' if infinite_mode else steps}, dt={dt:.4f}")
    if infinite_mode:
        print("Press Ctrl+C to stop.")

    i = 0
    try:
        while True:
            if (not infinite_mode) and i >= steps:
                break

            ee_force = np.array(
                [
                    2.0 * np.sin(2.0 * np.pi * 0.2 * i * dt),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.05 * np.cos(2.0 * np.pi * 0.1 * i * dt),
                ],
                dtype=float,
            )
            dq = controller(
                target_poses=None,
                current_joint_state=q,
                current_joint_torque=tau_ext,
                ee_force=ee_force,
            )
            q = np.clip(q + dq * dt, q_min, q_max)

            if i % print_every == 0:
                print(
                    f"step={i:04d} | |dq|={np.linalg.norm(dq):.4f} | "
                    f"q[:3]=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}]"
                )
            i += 1
    except KeyboardInterrupt:
        print("\nStop signal received (Ctrl+C).")
    finally:
        print("Standalone optas admittance demo finished")


def run_hardware_loop(
    config: dict,
    steps: int,
    print_every: int,
    robot_ip: str,
    sensor_port: str,
    sensor_alpha: float,
    home_robot: bool,
    use_deadband: bool,
) -> None:
    try:
        import mecademicpy.robot as mdr
    except Exception as exc:
        raise ImportError("mecademicpy is required for hardware mode") from exc

    controller = IntegratedAdmittanceController(config)
    params = config.get("parameters", {})
    dt = float(params.get("dt", 0.01))
    q_min = np.asarray(params.get("q_min", [-10.0] * 6), dtype=float)
    q_max = np.asarray(params.get("q_max", [10.0] * 6), dtype=float)
    tau_ext = np.zeros(6, dtype=float)

    robot = None
    sensor = None
    infinite_mode = steps <= 0
    print("Standalone optas admittance demo started (hardware)")
    print(f"steps={'infinite' if infinite_mode else steps}, dt={dt:.4f}, robot_ip={robot_ip}, sensor_port={sensor_port}")
    if infinite_mode:
        print("Press Ctrl+C to stop.")

    try:
        robot = mdr.Robot()
        robot.Connect(address=robot_ip, enable_synchronous_mode=False)
        if home_robot:
            robot.ActivateAndHome()
            robot.WaitHomed()
        else:
            robot.ActivateRobot()
        time.sleep(0.5)
        print("Robot connected.")

        sensor = CustomBotaSerialSensor(port=sensor_port, alpha=sensor_alpha)
        sensor.start()
        print("Force sensor started.")

        i = 0
        while True:
            if (not infinite_mode) and i >= steps:
                break

            tic = time.time()
            q = np.deg2rad(np.asarray(robot.GetJoints(), dtype=float))
            wrench = (
                np.asarray(sensor.get_wrench_deadband(), dtype=float)
                if use_deadband
                else np.asarray(sensor.get_wrench_alpha(), dtype=float)
            )
            dq = controller(
                target_poses=None,
                current_joint_state=q,
                current_joint_torque=tau_ext,
                ee_force=wrench,
            )

            q_cmd = np.clip(q + dq * dt, q_min, q_max)
            robot.MoveJoints(*[float(v) for v in np.rad2deg(q_cmd)])

            if i % print_every == 0:
                print(
                    f"step={i:04d} | |dq|={np.linalg.norm(dq):.4f} | "
                    f"Fxyz=[{wrench[0]:.2f}, {wrench[1]:.2f}, {wrench[2]:.2f}]"
                )
            i += 1

            sleep_t = max(0.0, dt - (time.time() - tic))
            if sleep_t > 0.0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nStop signal received (Ctrl+C).")
    finally:
        if sensor is not None:
            try:
                sensor.end()
                print("Force sensor stopped.")
            except Exception as exc:
                print(f"Force sensor cleanup failed: {exc}")

        if robot is not None:
            try:
                robot.WaitIdle()
            except Exception:
                pass
            try:
                robot.DeactivateRobot()
            except Exception:
                pass
            try:
                robot.Disconnect()
                print("Robot disconnected.")
            except Exception as exc:
                print(f"Robot cleanup failed: {exc}")

        print("Standalone optas admittance demo finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone optas admittance module")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to YAML config")
    parser.add_argument("--mode", choices=["sim", "hardware"], default="sim")
    parser.add_argument("--steps", type=int, default=0, help="0 or negative for infinite run")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--robot-ip", type=str, default="192.168.0.100")
    parser.add_argument("--sensor-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--sensor-alpha", type=float, default=0.1)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--raw-wrench", action="store_true")
    args = parser.parse_args()

    config = _load_and_resolve_config(args.config)

    if args.mode == "hardware":
        run_hardware_loop(
            config=config,
            steps=args.steps,
            print_every=max(1, args.print_every),
            robot_ip=args.robot_ip,
            sensor_port=args.sensor_port,
            sensor_alpha=float(args.sensor_alpha),
            home_robot=not args.no_home,
            use_deadband=not args.raw_wrench,
        )
    else:
        run_sim_loop(
            config=config,
            steps=args.steps,
            print_every=max(1, args.print_every),
        )


if __name__ == "__main__":
    main()
