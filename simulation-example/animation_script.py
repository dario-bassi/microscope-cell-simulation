# animation_script.py
import numpy as np
import cv2
from src.microscope_simulation.microscope_sim_optimized import MicroscopeSimOptmized

# Simulation setup
FIELD_SIZE = 512
sim = MicroscopeSimOptmized(width=FIELD_SIZE, height=FIELD_SIZE, nb_cells=30, cell_type='optogenetic')

# Video writer setup (change 'XVID' to 'mp4v' for .mp4 output if preferred)
fps = 20
video_filename = 'opto_simulation.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (FIELD_SIZE, FIELD_SIZE))

# Animation parameters
n_frames = 400
spot_radius = 50
trail = []

for t in range(n_frames):
    # Moving light spot: circle moves horizontally across the field
    mask = np.zeros((FIELD_SIZE, FIELD_SIZE), dtype=bool)
    cx = int((FIELD_SIZE/10) + (FIELD_SIZE-2*spot_radius) * t / n_frames)  # moves left to right
    cy = FIELD_SIZE // 2
    yy, xx = np.ogrid[:FIELD_SIZE, :FIELD_SIZE]
    mask_area = (xx - cx)**2 + (yy - cy)**2 <= spot_radius**2
    mask[mask_area] = True

    # Advance simulation and get frame
    frame = sim.snap_frame(mask=mask)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Overlay the mask as a semi-transparent red area
    mask_rgb = np.zeros_like(frame_bgr)
    mask_rgb[mask] = (0, 0, 255)  # Red in BGR
    alpha = 0.35  # transparency for mask overlay
    frame_overlay = cv2.addWeighted(frame_bgr, 1.0, mask_rgb, alpha, 0)

    out.write(frame_overlay)

    # Save images as PNGs too (optional)
    # cv2.imwrite(f"frames/frame_{t:03d}.png", frame_bgr)

    print(f"Frame {t+1}/{n_frames}")

out.release()
print(f"Video saved as {video_filename}")