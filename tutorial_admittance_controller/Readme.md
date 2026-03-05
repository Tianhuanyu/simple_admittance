# Minimal Admittance Controller — README

## What it is
A single-file, first-order **admittance controller** that:
- Loads a URDF from the **current folder** (default: `meca_robot_with_botaandshaft.urdf`).
- FK via **urdfpy**; 6×N Jacobian via **pykin** (if installed) or **numeric finite differences**.
- Maps an EE wrench → joint velocities using **damped least squares** (no ROS, no optimizers).

Public API (matches original):
dq = AdmittanceController(cfg)(target_poses, q, tau, ee_wrench, collision=None)

## Dependencies (Python 3.10)
Core: urdfpy==0.0.22, numpy>=1.23, scipy>=1.10,<1.12, lxml>=4.6,<5, trimesh>=4.0, networkx==2.8.8  
Optional (robust import / rendering): pyrender==0.1.45, PyOpenGL==3.1.0, freetype-py, pyglet<2.2, pillow  
Optional (analytic Jacobian): pykin

## Quick setup (recommended: clean venv)
python3 -m venv ~/venvs/urdfctrl
source ~/venvs/urdfctrl/bin/activate
python -m pip install --upgrade pip
pip install --no-deps urdfpy==0.0.22
pip install "networkx==2.8.8" "numpy>=1.23" "scipy>=1.10,<1.12" "lxml>=4.6,<5" trimesh
pip install "pyrender==0.1.45" "PyOpenGL==3.1.0" freetype-py "pyglet<2.2" pillow
# optional:
pip install pykin

## Run
python admittance_controller.py

## Principle (first-order admittance)
1) Body-frame mapping:
   v_b = k_f * f_b,   ω_b = k_τ * τ_b
2) World-frame conversion (R_we from FK):
   v_w = R_we v_b,    ω_w = R_we ω_b
3) Joint velocities via damped least squares:
   dq = J⁺ [v_w; ω_w],  J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹

Notes:
- No inertia/second-order dynamics; this is a minimal variant.
- If pykin is missing, the Jacobian falls back to **central-difference** FD.
- The controller outputs joint **velocities**; integrate externally if needed.
