
# Virtual simulation ![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)

## 🧬 What is Opto-Loop-Sim?
Opto-Loop-Sim is a highly modular and efficient simulation framework for virtual microscopy experiments and optical feedback loops. It is designed to facilitate the prototyping, testing, and validation of advanced microscope control—including optogenetic experiments and drug-response assays—without physical hardware.

The simulator emulates realistic cell dynamics, optogenetic stimulation, drug effects, and microscope camera imaging. It integrates seamlessly as a Python device in [pymmcore-plus](https://github.com/tlambert03/pymmcore-plus), enabling automated imaging, stage movement, and environmental control in a virtual environment.

---

## 📂 Project Structure

* **src/microscope_simulation/cell_base.py** — Implements the fundamental cell model: geometry, movement, realistic membrane physics, and collision.
* **src/microscope_simulation/cell_optogenetic.py** — Defines cells that react to spatial stimulation masks as in light-dependent experiments.
* **src/microscope_simulation/cell_drug.py** — Cells with tunable responses to simulated drug addition (growth, motility, apoptosis).
* **src/microscope_simulation/cell_normal.py** — Simpler cells for baseline population; supports random fluorescence traits and basic behavior.
* **src/microscope_simulation/renderer.py** — Uses OpenCV to render realistic brightfield/fluorescence images, including out-of-focus blur and camera noise.
* **src/microscope_simulation/spatial_grid.py** — Fast spatial lookup for cell-cell collision detection and resolution.
* **src/microscope_simulation/microscope_sim_optimized.py** — Main simulation engine: orchestrates all cells, updates, collisions, modes, and supports tight pymmcore-plus integration.

---

## 🤝 pymmcore-plus Integration & Usage

Opto-Loop-Sim is designed for smooth use as a virtual microscope in your [pymmcore-plus](https://github.com/tlambert03/pymmcore-plus) environment:

- **Device simulation:** Swap in your real `Camera`, `Stage`, and other devices with their virtual (Python) counterparts that use Opto-Loop-Sim as the physical backend.
- **Snap/acquire:** "Camera" snaps call the simulator, receiving synthetic images (with realistic cells and noise) as NumPy arrays.
- **Stimulation:** Send optogenetic or drug conditions as virtual "illumination" or environmental changes, either pixelwise or to the entire field.

*Example* (using pymmcore-plus API):
```python
from pymmcore_plus.experimental.unicore import UniMMCore
from pymmcore_camera_sim import SimCameraDevice # example of a python Camera
from pymmcore_stage_sim import SimStageDevice # example of a python XY-Stage
core = UniMMCore()
# load device
core.loadPyDevice("Camera", SimCameraDevice(core=core, microscope_sim=microscope_simulation))
core.loadPyDevice("XYStage", SimStageDevice(microscope_sim=microscope_simulation))
core.setXYPosition(100, 100)      # move virtual stage
img = core.snapImage()            # get a virtual image
np_img = core.getImage() # np.ndarray of the simulation
```

---

## ⚙️ Installation

**Requirements:**
- Python 3.12
- Conda (recommended)

### 1. Clone this repository
```shell
git clone https://github.com/yourusername/Opto-Loop-Sim.git
cd Opto-Loop-Sim
```

### 2. Create and activate a conda environment
```shell
conda create -n optoloop python=3.12
conda activate optoloop
```

### 3. Install dependencies
```shell
pip install -r requirements.txt
# or (with conda for best performance on numba, opencv, etc)
# conda install --file requirements.txt
```

---

## 🚀 Example: Simulate and Save an Animation
Here’s an example script that creates a video of growing cells under a drug, highlighting simple usage:

```python
import numpy as np
import cv2
from src.microscope_simulation.microscope_sim_optimized import MicroscopeSimOptmized

FIELD_SIZE = 512
sim = MicroscopeSimOptmized(width=FIELD_SIZE, height=FIELD_SIZE, nb_cells=30, cell_type='drug')
mask = np.ones((FIELD_SIZE, FIELD_SIZE), dtype=bool)  # apply drug everywhere

# Video setup
out = cv2.VideoWriter('growth_demo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (FIELD_SIZE, FIELD_SIZE))
for t in range(120):
	sim.drug_type = 'growth'  # set drug
	sim.concentration = 0.03  # set concentration
	frame = sim.snap_frame(mask=mask)
	frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
	out.write(frame)
	print(f'Frame {t+1}/120')
out.release()
```

---

## 📄 License & Contributions

This package is open-source. Please see the LICENSE file. Contributions are welcome!

---

## 🎬 Demos


#### Normal cells simulation
<video src="https://github.com/user-attachments/assets/11a8e76c-8be7-4ff5-8486-25c406ba32f2" normal></video>

The video shows the simulation of cells that move accordin to Brownian motion and collide with each other. Each cell have a nucleus and a membrane.

#### Optogenetic simulation
<video src="https://github.com/user-attachments/assets/aa1d49d1-98fc-4765-876e-3304f3f31c8c" opto_loop_1></video>

The video shows a simulation of a laser that moves across a stage and the cells interact with the light moving towards the source of light.

<video src="https://github.com/user-attachments/assets/04042e2d-6312-4a3b-9980-cc1642ddf8a5" opto-loop_2></video>

#### Drug simulation -Enhanced mobility
<video src="https://github.com/user-attachments/assets/7fbc7739-98ac-49df-b366-afbacdbba1d9" mobility></video>

The video shows a simulation of cells that have an increase Brownian motion thanks to a drug.

#### Drug simulation - Enhanced cell growth
<video src="https://github.com/user-attachments/assets/e200f358-8ec8-4e6c-b0f6-f98bda6d2c73" growth></video>

The video shows a simulation of cells that becomes bigger after given a drug.

#### Drug simulation - Enhanced apoptosis

<video src="https://github.com/user-attachments/assets/c89ad2e5-0b0b-4ff0-93ef-a139b61ed92d" apoptosis></video>

The video shows a simulation of cells that die, decreasing their size emulating cells behaviour during apoptosis.

---
