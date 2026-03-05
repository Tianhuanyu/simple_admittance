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
