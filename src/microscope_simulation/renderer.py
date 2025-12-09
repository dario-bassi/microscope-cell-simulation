"""Optimize rendering using OpenCV"""
import numpy as np
import cv2
from typing import List, Tuple, Optional
from src.microscope_simulation.cell_base import CellBase


class Renderer:
    """Fast renderer using OpenCV."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.contrast = 1.5
        self.brightness = 1.2
        self.blur_radius = 3

    
    def render_cells(self, cells: List[CellBase], mode: int = 0,
                     camera_offset: Tuple[float, float] = (0,0),
                     focal_plane: float = 0.0) -> np.ndarray:
        """Render cells to image array using OpenCV."""
        # Create base image
        img = np.full((self.height, self.width, 3), 235, dtype=np.uint8)

        # Get visibile cells
        visible_cells = self._get_visible_cells(cells, camera_offset)

        for cell in visible_cells:
            self._draw_cell(img, cell, mode, camera_offset, focal_plane)

        # apply microscope filters
        img = self._apply_filters(img, mode)

        return img
    
    def _get_visible_cells(self, cells: List[CellBase], camera_offset: Tuple[float, float]) -> List[CellBase]:
        """Filter cells visible in current viewport"""

        margin = 100
        vx, vy = camera_offset

        visible = []
        for cell in cells:
            if (vx - margin <= cell.center[0] <= vx + self.width + margin and 
                vy - margin <= cell.center[1] <= vy + self.height + margin):
                visible.append(cell)
        
        return visible
    
    def _smooth_contour(self, pts: np.ndarray, smoothing_factor: int = 3) -> np.ndarray:
        """Smooth cell contour using spline interpolation for more natural appearance."""
        if len(pts) < 4:
            return pts
            
        # Close the contour by adding first point at end
        pts_closed = np.vstack([pts, pts[0]])
        
        # Use cv2.approxPolyDP for initial smoothing
        epsilon = 0.5  # Adjust for smoothness
        smoothed = cv2.approxPolyDP(pts_closed, epsilon, True)
        
        # Further smooth using moving average
        if smoothing_factor > 1:
            kernel = np.ones(smoothing_factor) / smoothing_factor
            smoothed_x = np.convolve(smoothed[:, 0, 0], kernel, mode='same')
            smoothed_y = np.convolve(smoothed[:, 0, 1], kernel, mode='same')
            smoothed = np.column_stack([smoothed_x, smoothed_y]).astype(np.int32)
        
        return smoothed
    
    def _draw_smooth_cell(self, img: np.ndarray, center: np.ndarray, 
                         vertices: np.ndarray, base_r: float, 
                         color: tuple, thickness: int = -1) -> None:
        """Draw a smooth cell shape using bezier curves or ellipse fitting."""
        # Option 1: Use more vertices by interpolating
        n_interp = 100  # More points for smoother appearance
        angles_interp = np.linspace(0, 2 * np.pi, n_interp, endpoint=False)
        
        # Interpolate radii to get smooth transitions
        angles_orig = np.linspace(0, 2 * np.pi, len(vertices), endpoint=False)
        radii_orig = np.linalg.norm(vertices - center, axis=1)
        radii_interp = np.interp(angles_interp, angles_orig, radii_orig, period=2*np.pi)
        
        # Generate smooth vertices
        smooth_verts = np.zeros((n_interp, 2))
        smooth_verts[:, 0] = center[0] + radii_interp * np.cos(angles_interp)
        smooth_verts[:, 1] = center[1] + radii_interp * np.sin(angles_interp)
        smooth_pts = smooth_verts.astype(np.int32)
        
        # Draw the smooth polygon
        cv2.fillPoly(img, [smooth_pts], color, lineType=cv2.LINE_AA)  # LINE_AA for anti-aliasing
        
        return smooth_pts
    
    def _draw_cell(self, img: np.ndarray, cell: CellBase, mode: int, camera_offset: Tuple[float, float], focal_plane: float) -> None:
        """Draw a single cell using OpenCV."""
        # Calculate focus effects
        z_distance = abs(cell.z_position - focal_plane)
        opacity = max(0.1, 1.0 - z_distance / 50.0)

        # Get vertex postion adjusted for camera
        vertices = cell.vertices_positions - camera_offset
        center_screen = cell.center - camera_offset

        # Skip if center is outside viewport
        if not (0 <= center_screen[0] < self.width and 0 <= center_screen[1] < self.height):
            if not np.any((vertices[:, 0] >= -50) & (vertices[:, 0] < self.width + 50) & 
                         (vertices[:, 1] >= -50) & (vertices[:, 1] < self.height + 50)):
                return
        
        # fluorescence mode
        if mode == 0: # brightfield
            # Draw layered cell body
            for i in range(6, 0, -1):
                scale = 0.7 + (i / 6.0) * 0.3
                shade = int(100 + 80 * (i / 6.0))
                color = (shade, shade, min(255, int(shade * 1.1)))


                # Scale vertices smoothly
                scaled_verts = center_screen + (vertices - center_screen) * scale
                
                # Draw smooth shape
                smooth_pts = self._draw_smooth_cell(img, center_screen, scaled_verts, 
                                                   cell.base_r * scale, color)
            
            # Draw smooth membrane outline with anti-aliasing
            smooth_outline = self._draw_smooth_cell(img, center_screen, vertices, 
                                                   cell.base_r, (50, 50, 60), thickness=1)
            
            # Draw nucleus with gradient effect
            nucleus_pos = tuple(center_screen.astype(int))
            nucleus_radius = int(0.35 * cell.base_r)
            
            # Draw multiple circles for gradient effect
            for i in range(3, 0, -1):
                radius = int(nucleus_radius * (i / 3.0))
                shade = 60 + 30 * (3 - i)
                cv2.circle(img, nucleus_pos, radius, (shade, shade, shade + 40), 
                          -1, lineType=cv2.LINE_AA)

        elif mode == 1: # nucleus fluorescence
            # Draw cell outline for reference (very faint)
            smooth_outline = self._draw_smooth_cell(img, center_screen, vertices, 
                                               cell.base_r, (20, 20, 20), thickness=1)
            nucleus_pos = tuple(center_screen.astype(int))
            # Draw glowing nucleus with multiple circles
            for i in range(3, 0, -1):
                radius = int(0.45 * cell.base_r * opacity * (1 + i * 0.1))
                intensity = int(136 * (i / 3.0) * cell.nucleus_fluorescence)
                cv2.circle(img, nucleus_pos, radius, (8, 8, intensity), 
                            -1, lineType=cv2.LINE_AA)
            
            # Bright center
            cv2.circle(img, nucleus_pos, int(0.4 * cell.base_r * opacity), 
                        (0, 255, 255), -1, lineType=cv2.LINE_AA)
                
            # check with color (0, 255, 255, 200)

        elif mode == 2: # membrane fluorescence
            if np.any(cell.membrane_fluorescence > 0):
                avg_fluorescence = np.mean(cell.membrane_fluorescence)
            # Draw smooth membrane with glow effect
                # Outer glow
                glow_pts = self._draw_smooth_cell(img, center_screen, vertices * 1.05, 
                                                 cell.base_r * 1.05, (4, 4, int(68 * avg_fluorescence)))
                
                # Main membrane
                membrane_pts = self._draw_smooth_cell(img, center_screen, vertices, 
                                                     cell.base_r, (8, 8, int(136 * avg_fluorescence)))
                
                # Inner darker area for contrast
                inner_pts = self._draw_smooth_cell(img, center_screen, vertices * 0.85, 
                                                  cell.base_r * 0.85, (10, 10, 10))

    def _apply_filters(self, img: np.ndarray, mode: int) -> np.ndarray:
        """Apply microscope-like filters using OpenCV"""
        # Apply Gaussian blur FIRST for smoother appearance
        img = cv2.GaussianBlur(img, (5, 5), 1.5)
        
        # Convert to grayscale if brightfield
        if mode == 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Enhance edges slightly for brightfield
            edges = cv2.Canny(img, 30, 60)
            img = cv2.addWeighted(img, 0.9, edges, 0.1, 0)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Apply contrast and brightness
        img = cv2.convertScaleAbs(img, alpha=self.contrast, beta=self.brightness * 30)
        
        # Final smoothing
        img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        # Add slight noise for realism
        noise = np.random.normal(0, 2, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        return img