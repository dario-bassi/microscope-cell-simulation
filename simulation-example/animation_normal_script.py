import numpy as np
import cv2
from src.microscope_simulation.microscope_sim_optimized import MicroscopeSimOptmized

FIELD_SIZE = 512
N_FRAMES = 540
fps = 20
video_filename = "normal_simulation_demo.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (FIELD_SIZE, FIELD_SIZE))

# Initialize simulation with only DrugResponseCells
sim = MicroscopeSimOptmized(width=FIELD_SIZE, height=FIELD_SIZE, nb_cells=30, cell_type='normal')

for t in range(N_FRAMES):
    # Cycle drug application every 3 frames for better visibility

    # Advance simulation and get frame (no illumination, just drug state)
    frame = sim.snap_frame()

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    #frame_bgr = cv2.cvtColor(frame_strip, cv2.COLOR_RGB2BGR)
    out.write(frame)
    # Optional: cv2.imwrite(f'frames/drug_frame_{t:03d}.png', frame_bgr)
    print(f"Frame {t+1}/{N_FRAMES}")

out.release()
print(f"Video saved as {video_filename}")