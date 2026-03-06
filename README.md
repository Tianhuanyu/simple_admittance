# Standalone Admittance Module

This folder is an independent admittance-control module using local `IntegratedAdmittanceController` (optas-based).
This project received hardware support from the BOTA System.

## Contents

- `admittance_entry.py`: runtime entry (sim + hardware)
- `controller/integrated_admittance_controller.py`: standalone integrated controller
- `controller/qp_builder.py`: standalone QP builder
- `controller/config_schema.py`: standalone config schema
- `config/config_admit.yaml`: standalone config
- `urdf/meca_robot_with_botaandshaft.urdf`: standalone URDF
- `meshes/`: URDF mesh assets
- `hardware/ftsensor.py`: force sensor driver

## Dependencies

- `numpy`
- `pyoptas`
- `mecademicpy`
- `pyyaml`
- `optas`
- `casadi`
- `pyserial`
- `crc`
- `mecademicpy` (hardware mode only)

## Optas Setup

Install required packages:

```bash
pip install optas casadi numpy pyyaml pyserial crc mecademicpy pyoptas
```

Quick check:

```bash
python -c "import optas, casadi; print('optas/casadi ok')"
```

If you run in ROS environment, `ament_index_python` can be used automatically.
In standalone mode, this module falls back to local URDF path from config.

## Config Notes (`config/config_admit.yaml`)

Required:

- `parameters.urdf_string`:
  - relative path is resolved from `config/` folder
  - default: `../urdf/meca_robot_with_botaandshaft.urdf`
- `parameters.end_link_name`: tracking link for Jacobian/QP
- `parameters.tip_link_name`: admittance mount frame used for twist transform
- `parameters.dt`: controller step time

QP / limits:

- `parameters.wp`, `parameters.wr`, `parameters.wdq`: twist/velocity costs
- `parameters.q_min`, `parameters.q_max`, `parameters.dq_max`: joint bounds

Admittance:

- `admittance.force_gain`, `admittance.torque_gain`
- `admittance.mass`, `admittance.rot_mass`
- `admittance.damping`, `admittance.rot_damping`
- `admittance.compliance_offset`
- `admittance.reduandant_pos_axis`, `admittance.reduandant_rot_axis` (optional)

## Run

Simulation:

```bash
python admittance_entry.py --mode sim --steps 0
```

Hardware:
(Linux)
```bash
python admittance_entry.py --mode hardware --steps 0 --robot-ip 192.168.0.100 --sensor-port /dev/ttyUSB0
```
(Windows)
```bash
python admittance_entry.py --mode hardware --steps 0 --robot-ip 192.168.0.100 --sensor-port COM3
```

GUI (pure Python `tkinter`, hardware only):
```bash
python admittance_entry.py --mode gui --robot-ip 192.168.0.100 --sensor-port /dev/ttyUSB0
```

Buttons in GUI:
- `Hand-Guide`: enter/stop admittance hand-guiding mode
- `Home`: move robot to predefined home joint angles from config (`parameters.home_joint_deg`)
- `Record`: start/stop CSV recording
- `Replay`: replay the latest recorded CSV trajectory using joint points (`q1_deg ... q6_deg`)

Replay behavior details:
- Replay uses the most recent file in `records/` matching `admittance_record_*.csv` (or the file from the latest `Record` action in the current GUI session).
- Replay timing follows the recorded `elapsed_s` column to preserve trajectory timing.
- Replay command source is joint angle columns `q1_deg ... q6_deg`.
- Starting replay automatically stops `Hand-Guide` and `Record` to avoid command conflicts.
- `Home` also stops replay before moving to the home point.
- Press `Replay` again while replay is running to stop playback.

Recording CSV format:
- `timestamp_unix_s`, `elapsed_s`
- `q1_rad ... q6_rad`
- `q1_deg ... q6_deg`
- `fx, fy, fz, mx, my, mz`
