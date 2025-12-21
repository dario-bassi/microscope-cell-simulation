from src.microscope_simulation.microscope_sim_optimized import MicroscopeSimOptmized
import numpy as np

# Singleton to make available for all pyDevice the bridge
GLOBAL_BRIDGE = None

class SimulationBridge:
    """
    This class is a simulation bridge that connect the python device from UniMMcore
    to the simulation. In this way we differentiate totally any devices and we 
    will be able add pyDevice from within a configuration file. In addition, any other
    change in the simulation, will directly change in the SimulationBridge, without
    modifing the pyDevice.
    """
    def __init__(self, microscope_sim : MicroscopeSimOptmized) -> None:
        self._sim = microscope_sim

    
    def snap(self, *args, **kwargs) -> np.ndarray:
        return self._sim.snap_frame(*args, *kwargs)
    
    def set_stage(self, x: float, y: float) -> None:
        self._sim.camera_offset = (x, y)

    def set_focus(self, z: float) -> None:
        self._sim.set_focal_plane(z)

    def set_state(self, dict_state: dict) -> None:
        self._sim.state_devices.update(dict_state)

    def set_objective(self, idx):
        mag = {0: 10, 1: 20, 2: 40}
        # ADD part of the objective