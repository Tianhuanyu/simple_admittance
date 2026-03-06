import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

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


def _extract_home_joint_rad(config: dict, ndof: int = 6) -> np.ndarray:
    params = config.get("parameters", {})
    if "home_joint_deg" in params:
        home_deg = np.asarray(params["home_joint_deg"], dtype=float).reshape(-1)
        if home_deg.size != ndof:
            raise ValueError(f"parameters.home_joint_deg must contain {ndof} values")
        return np.deg2rad(home_deg)
    if "home_joint_rad" in params:
        home_rad = np.asarray(params["home_joint_rad"], dtype=float).reshape(-1)
        if home_rad.size != ndof:
            raise ValueError(f"parameters.home_joint_rad must contain {ndof} values")
        return home_rad

    q_min = np.asarray(params.get("q_min", [-1.0] * ndof), dtype=float)
    q_max = np.asarray(params.get("q_max", [1.0] * ndof), dtype=float)
    return 0.5 * (q_min + q_max)


class HardwareAdmittanceSession:
    """Encapsulates robot/sensor lifecycle and runtime actions for hardware mode."""

    def __init__(
        self,
        config: dict,
        robot_ip: str,
        sensor_port: str,
        sensor_alpha: float,
        use_deadband: bool,
    ) -> None:
        self.config = config
        self.robot_ip = robot_ip
        self.sensor_port = sensor_port
        self.sensor_alpha = float(sensor_alpha)
        self.use_deadband = bool(use_deadband)

        params = config.get("parameters", {})
        self.dt = float(params.get("dt", 0.01))
        self.q_min = np.asarray(params.get("q_min", [-10.0] * 6), dtype=float)
        self.q_max = np.asarray(params.get("q_max", [10.0] * 6), dtype=float)
        self.home_q_rad = _extract_home_joint_rad(config, ndof=6)

        self.controller = IntegratedAdmittanceController(config)
        self.tau_ext = np.zeros(6, dtype=float)

        self.robot = None
        self.sensor = None

        self._handguide_stop = threading.Event()
        self._handguide_thread: Optional[threading.Thread] = None

        self._record_stop = threading.Event()
        self._record_thread: Optional[threading.Thread] = None
        self._record_file = None
        self._record_writer = None

        self._status_lock = threading.Lock()
        self.last_q = np.zeros(6, dtype=float)
        self.last_wrench = np.zeros(6, dtype=float)

        self._record_lock = threading.Lock()
        self._io_lock = threading.Lock()

    def connect(self, home_robot: bool = True) -> None:
        if self.robot is not None:
            return

        try:
            import mecademicpy.robot as mdr
        except Exception as exc:
            raise ImportError("mecademicpy is required for hardware mode") from exc

        self.robot = mdr.Robot()
        self.robot.Connect(address=self.robot_ip, enable_synchronous_mode=False)

        if home_robot:
            self.robot.ActivateAndHome()
            self.robot.WaitHomed()
        else:
            self.robot.ActivateRobot()

        time.sleep(0.5)
        self.sensor = CustomBotaSerialSensor(port=self.sensor_port, alpha=self.sensor_alpha)
        self.sensor.start()

    def disconnect(self) -> None:
        self.stop_record()
        self.stop_handguide()

        if self.sensor is not None:
            try:
                self.sensor.end()
            except Exception as exc:
                print(f"Force sensor cleanup failed: {exc}")
            finally:
                self.sensor = None

        if self.robot is not None:
            try:
                self.robot.WaitIdle()
            except Exception:
                pass
            try:
                self.robot.DeactivateRobot()
            except Exception:
                pass
            try:
                self.robot.Disconnect()
            except Exception as exc:
                print(f"Robot cleanup failed: {exc}")
            finally:
                self.robot = None

    def _get_wrench(self) -> np.ndarray:
        if self.sensor is None:
            raise RuntimeError("Force sensor is not connected")
        if self.use_deadband:
            return np.asarray(self.sensor.get_wrench_deadband(), dtype=float)
        return np.asarray(self.sensor.get_wrench_alpha(), dtype=float)

    def read_state(self) -> tuple[np.ndarray, np.ndarray]:
        if self.robot is None:
            raise RuntimeError("Robot is not connected")
        with self._io_lock:
            q = np.deg2rad(np.asarray(self.robot.GetJoints(), dtype=float))
            wrench = self._get_wrench()
        with self._status_lock:
            self.last_q = q.copy()
            self.last_wrench = wrench.copy()
        return q, wrench

    def move_home(self) -> None:
        if self.robot is None:
            raise RuntimeError("Robot is not connected")
        self.stop_handguide()
        with self._io_lock:
            self.robot.MoveJoints(*[float(v) for v in np.rad2deg(self.home_q_rad)])
            self.robot.WaitIdle()

    def _handguide_loop(self, print_every: int, max_steps: int) -> None:
        i = 0
        while not self._handguide_stop.is_set():
            if max_steps > 0 and i >= max_steps:
                break
            tic = time.time()

            q, wrench = self.read_state()
            dq = self.controller(
                target_poses=None,
                current_joint_state=q,
                current_joint_torque=self.tau_ext,
                ee_force=wrench,
            )
            q_cmd = np.clip(q + dq * self.dt, self.q_min, self.q_max)
            with self._io_lock:
                self.robot.MoveJoints(*[float(v) for v in np.rad2deg(q_cmd)])

            if i % print_every == 0:
                print(
                    f"step={i:04d} | |dq|={np.linalg.norm(dq):.4f} | "
                    f"Fxyz=[{wrench[0]:.2f}, {wrench[1]:.2f}, {wrench[2]:.2f}]"
                )

            i += 1
            sleep_t = max(0.0, self.dt - (time.time() - tic))
            if sleep_t > 0.0:
                time.sleep(sleep_t)

    def start_handguide(self, print_every: int = 50, max_steps: int = 0) -> bool:
        if self._handguide_thread is not None and self._handguide_thread.is_alive():
            return False
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        self._handguide_stop.clear()
        self._handguide_thread = threading.Thread(
            target=self._handguide_loop,
            args=(max(1, print_every), int(max_steps)),
            daemon=True,
        )
        self._handguide_thread.start()
        return True

    def stop_handguide(self) -> bool:
        if self._handguide_thread is None:
            return False
        self._handguide_stop.set()
        self._handguide_thread.join(timeout=2.0)
        self._handguide_thread = None
        return True

    def is_handguide_running(self) -> bool:
        return self._handguide_thread is not None and self._handguide_thread.is_alive()

    def _record_loop(self) -> None:
        t0 = time.time()
        while not self._record_stop.is_set():
            tic = time.time()
            q, wrench = self.read_state()
            q_deg = np.rad2deg(q)
            ts = time.time()
            elapsed = ts - t0

            row = [ts, elapsed]
            row.extend(q.tolist())
            row.extend(q_deg.tolist())
            row.extend(wrench.tolist())

            with self._record_lock:
                if self._record_writer is not None:
                    self._record_writer.writerow(row)
                    self._record_file.flush()

            sleep_t = max(0.0, self.dt - (time.time() - tic))
            if sleep_t > 0.0:
                time.sleep(sleep_t)

    def start_record(self, output_csv: Path) -> bool:
        if self._record_thread is not None and self._record_thread.is_alive():
            return False
        if self.robot is None:
            raise RuntimeError("Robot is not connected")

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._record_file = output_csv.open("w", newline="", encoding="utf-8")
        self._record_writer = csv.writer(self._record_file)

        header = ["timestamp_unix_s", "elapsed_s"]
        header.extend([f"q{i+1}_rad" for i in range(6)])
        header.extend([f"q{i+1}_deg" for i in range(6)])
        header.extend(["fx", "fy", "fz", "mx", "my", "mz"])
        self._record_writer.writerow(header)

        self._record_stop.clear()
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        return True

    def stop_record(self) -> bool:
        if self._record_thread is None:
            return False

        self._record_stop.set()
        self._record_thread.join(timeout=2.0)
        self._record_thread = None

        with self._record_lock:
            if self._record_file is not None:
                self._record_file.flush()
                self._record_file.close()
            self._record_file = None
            self._record_writer = None
        return True

    def is_recording(self) -> bool:
        return self._record_thread is not None and self._record_thread.is_alive()


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
    session = HardwareAdmittanceSession(
        config=config,
        robot_ip=robot_ip,
        sensor_port=sensor_port,
        sensor_alpha=sensor_alpha,
        use_deadband=use_deadband,
    )
    infinite_mode = steps <= 0
    print("Standalone optas admittance demo started (hardware)")
    print(
        f"steps={'infinite' if infinite_mode else steps}, "
        f"dt={session.dt:.4f}, robot_ip={robot_ip}, sensor_port={sensor_port}"
    )
    if infinite_mode:
        print("Press Ctrl+C to stop.")

    try:
        session.connect(home_robot=home_robot)
        print("Robot and force sensor connected.")
        session.start_handguide(print_every=max(1, print_every), max_steps=max(0, steps))
        while session.is_handguide_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStop signal received (Ctrl+C).")
    finally:
        session.disconnect()
        print("Standalone optas admittance demo finished")


def run_hardware_gui(
    config: dict,
    robot_ip: str,
    sensor_port: str,
    sensor_alpha: float,
    home_robot: bool,
    use_deadband: bool,
    record_dir: Path,
) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception as exc:
        raise ImportError("tkinter is required for GUI mode") from exc

    session = HardwareAdmittanceSession(
        config=config,
        robot_ip=robot_ip,
        sensor_port=sensor_port,
        sensor_alpha=sensor_alpha,
        use_deadband=use_deadband,
    )

    session.connect(home_robot=home_robot)

    root = tk.Tk()
    root.title("Admittance Controller Panel")
    root.geometry("460x220")
    root.resizable(False, False)

    status_var = tk.StringVar(value="Connected. Ready.")
    record_file_var = tk.StringVar(value="Record file: <none>")

    def set_status(text: str) -> None:
        status_var.set(text)

    def run_background(task: Callable[[], None], err_title: str) -> None:
        def wrapped() -> None:
            try:
                task()
            except Exception as exc:
                root.after(0, lambda: messagebox.showerror(err_title, str(exc)))
                root.after(0, lambda: set_status(f"Error: {exc}"))

        threading.Thread(target=wrapped, daemon=True).start()

    def on_hand_guide_click() -> None:
        if session.is_handguide_running():
            session.stop_handguide()
            hand_guide_btn.configure(text="Hand-Guide")
            set_status("Hand-guide stopped.")
            return

        session.start_handguide(print_every=50, max_steps=0)
        hand_guide_btn.configure(text="Stop Hand-Guide")
        set_status("Hand-guide running.")

    def on_home_click() -> None:
        def task() -> None:
            root.after(0, lambda: set_status("Moving to home point..."))
            session.move_home()
            root.after(0, lambda: hand_guide_btn.configure(text="Hand-Guide"))
            root.after(0, lambda: set_status("Reached home point."))

        run_background(task, "Home Error")

    def on_record_click() -> None:
        if session.is_recording():
            session.stop_record()
            record_btn.configure(text="Record")
            set_status("Recording stopped.")
            return

        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = record_dir / f"admittance_record_{now}.csv"
        session.start_record(output_csv)
        record_btn.configure(text="Stop Record")
        record_file_var.set(f"Record file: {output_csv}")
        set_status("Recording started.")

    hand_guide_btn = tk.Button(root, text="Hand-Guide", width=16, height=2, command=on_hand_guide_click)
    home_btn = tk.Button(root, text="Home", width=16, height=2, command=on_home_click)
    record_btn = tk.Button(root, text="Record", width=16, height=2, command=on_record_click)

    hand_guide_btn.grid(row=0, column=0, padx=12, pady=16)
    home_btn.grid(row=0, column=1, padx=12, pady=16)
    record_btn.grid(row=0, column=2, padx=12, pady=16)

    status_label = tk.Label(root, textvariable=status_var, anchor="w")
    status_label.grid(row=1, column=0, columnspan=3, sticky="we", padx=12, pady=8)

    record_label = tk.Label(root, textvariable=record_file_var, anchor="w", justify="left", wraplength=430)
    record_label.grid(row=2, column=0, columnspan=3, sticky="we", padx=12)

    for col in range(3):
        root.grid_columnconfigure(col, weight=1)

    def on_close() -> None:
        try:
            session.disconnect()
        finally:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone optas admittance module")
    parser.add_argument("--config", type=Path, default=_default_config_path(), help="Path to YAML config")
    parser.add_argument("--mode", choices=["sim", "hardware", "gui"], default="sim")
    parser.add_argument("--steps", type=int, default=0, help="0 or negative for infinite run")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--robot-ip", type=str, default="192.168.0.100")
    parser.add_argument("--sensor-port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--sensor-alpha", type=float, default=0.1)
    parser.add_argument("--no-home", action="store_true")
    parser.add_argument("--raw-wrench", action="store_true")
    parser.add_argument("--record-dir", type=Path, default=Path("records"))
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
    elif args.mode == "gui":
        run_hardware_gui(
            config=config,
            robot_ip=args.robot_ip,
            sensor_port=args.sensor_port,
            sensor_alpha=float(args.sensor_alpha),
            home_robot=not args.no_home,
            use_deadband=not args.raw_wrench,
            record_dir=args.record_dir,
        )
    else:
        run_sim_loop(
            config=config,
            steps=args.steps,
            print_every=max(1, args.print_every),
        )


if __name__ == "__main__":
    main()
