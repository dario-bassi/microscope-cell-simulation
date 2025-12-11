# animation_drug_mobility.py
import numpy as np
import cv2
from src.microscope_simulation.microscope_sim_optimized import MicroscopeSimOptmized

FIELD_SIZE = 512
N_FRAMES = 540
fps = 20
video_filename = "drug_mobility_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (FIELD_SIZE, FIELD_SIZE))

sim = MicroscopeSimOptmized(width=FIELD_SIZE, height=FIELD_SIZE, nb_cells=30, cell_type='drug')

mask = np.ones((FIELD_SIZE, FIELD_SIZE), dtype=bool)

for t in range(N_FRAMES):
    sim.drug_type = "mobility"
    sim.concentration = 0.05
    frame = sim.snap_frame(mask=mask)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # Overlay blue
    overlay = np.zeros_like(frame)
    overlay[mask] = (255,0,0)
    frame_strip = cv2.addWeighted(frame, 1.0, overlay, 0.12, 0)

    out.write(frame_strip)
    print(f"Frame {t+1}/{N_FRAMES}")

out.release()
print(f"Video saved as {video_filename}")