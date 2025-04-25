"""
Visualizes generated spatial qa annotations from a JSON file onto the corresponding image.

This script loads an image and its associated annotations (in a specific JSON
format) and provides functionalities to overlay various types of grounding and
spatial relationship information.

Features:
- Displays the image using OpenCV.
- Overlays 2D rectangular bounding boxes (from the 'bbox' field).
- Overlays 3D oriented bounding boxes (from the 'bbox_3d' field) projected onto the image,
  including an orientation arrow.
- Cycles through unary spatial relationship points ('context' mode) using Prev/Next buttons.
- Cycles through pairwise spatial relationships ('pair' mode) using Prev/Next buttons,
  highlighting the involved objects.
- Optionally displays a top-down 2D occupancy map of object bounding boxes using Matplotlib
  when cycling through pairwise compatibility information.
- Optionally displays a 3D visualization of bounding boxes using Open3D (in a separate
  thread) when cycling through pairwise compatibility information.

Usage:
  python visualize_output.py --image_path <path_to_image.jpg> \
                                   --annotation_file <path_to_annotation.json> \
                                   [options]

Arguments:
  --image_path         : Required. Direct path to the image file to visualize.
  --annotation_file    : Required. Path to the specific JSON annotation file to visualize.
  --verbose, -v        : If set, prints detailed status messages and warnings.

Object Grounding Options:
  --object_2d_grounding : If set, overlays 2D rectangular bounding boxes ('bbox' field).
  --object_3d_grounding : If set, overlays projected 3D bounding boxes ('bbox_3d' field).

Spatial Relationship Cycling Modes (Select one):
  --context            : If set, cycles through unary spatial relations found in
                         `spatial_relationships.unary_relations`, showing associated
                         context points (`point_space_2d`).
  --configuration      : If set, cycles through pairwise spatial relations found in
                         `spatial_relationships.pairwise_relations`, highlighting the pair
                         and printing their spatial configuration information.
  --compatibility      : If set, cycles through pairwise spatial relations, highlighting
                         the pair, printing compatibility information, and displaying
                         the 2D occupancy map and 3D bounding box view (requires Open3D).

Controls:
  - Click 'Prev'/'Next' buttons (bottom corners) to cycle through selected spatial relationships.
  - Press 'q' to quit the visualization.
"""
import os
import json
import cv2
import argparse
import numpy as np
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import threading # Import threading
import open3d as o3d

import time # Import time for sleep
from collections import deque # For thread-safe communication

# --- Verbose Print Helper ---
verbose_mode = False # Global flag, set in main()
def vprint(*args, **kwargs):
    """Prints only if verbose_mode is True."""
    if verbose_mode:
        print(*args, **kwargs)

# --- Global or shared state (using a dictionary) ---
# This will hold the index and total combinations count for the mouse callback
visualization_state = {
    'current_combination_idx': 0,
    'total_combinations': 0,
    'needs_update': True # Flag to redraw after button click
}

# --- Button Definitions ---
# Define regions for buttons (x, y, width, height)
# Position them at the bottom-left and bottom-right
BUTTON_HEIGHT = 30
BUTTON_WIDTH = 60
PADDING = 10

# Will be set properly once image dimensions are known
prev_button_rect = None
next_button_rect = None

# Add state for the map window
map_figure_state = {
    'fig': None,
    'ax': None,
    'is_active': False
}

# Revised Open3D Visualizer State
o3d_visualizer_state = {
    'vis': None,         # The Visualizer object
    'thread': None,      # The visualization thread
    'run_thread': False, # Flag to keep the thread running
    'is_initialized': False, # Flag if vis window is created
    'update_queue': deque(maxlen=1), # Queue to send new geometry list
    'lock': threading.Lock() # Lock for accessing shared state if needed (queue is thread-safe)
}

# --- Mouse Callback Function ---
def handle_mouse_click(event, x, y, flags, param):
    global visualization_state, prev_button_rect, next_button_rect # Access shared state and button rects

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is within Previous button bounds
        if prev_button_rect and prev_button_rect[0] <= x <= prev_button_rect[0] + prev_button_rect[2] and \
           prev_button_rect[1] <= y <= prev_button_rect[1] + prev_button_rect[3]:
            total = visualization_state['total_combinations']
            if total > 0: # Prevent division by zero if no items
                visualization_state['current_combination_idx'] = (visualization_state['current_combination_idx'] - 1 + total) % total
            visualization_state['needs_update'] = True # Signal redraw needed
            # vprint("Prev clicked") # Debug

        # Check if click is within Next button bounds
        elif next_button_rect and next_button_rect[0] <= x <= next_button_rect[0] + next_button_rect[2] and \
             next_button_rect[1] <= y <= next_button_rect[1] + next_button_rect[3]:
            total = visualization_state['total_combinations']
            if total > 0: # Prevent division by zero if no items
                visualization_state['current_combination_idx'] = (visualization_state['current_combination_idx'] + 1) % total
            visualization_state['needs_update'] = True # Signal redraw needed
            # vprint("Next clicked") # Debug

def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        return json.load(f)

# --- NEW: Function to draw simple rectangular bounding boxes ---
def draw_bbox_rect(image, bbox, color=(0, 255, 0), thickness=2):
    """Draws a rectangular bounding box.

    Args:
        image: The image to draw on.
        bbox (list or tuple): [xmin, ymin, xmax, ymax]
        color: The color for the rectangle.
        thickness: The thickness of the lines.
    """
    if bbox and len(bbox) == 4:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

def draw_points(image, points, color=(0, 0, 255), radius=3):
    if not points: # Handle empty lists gracefully
        return
    for point in points:
        # Assuming point is a list of [x, y]
        try:
            x, y = map(int, point)
            cv2.circle(image, (x, y), radius, color, -1)
        except (ValueError, TypeError):
            vprint(f"Warning: Skipping invalid point data: {point}")

def draw_text(image, text, position, color=(255, 255, 255), font_scale=0.5, thickness=1):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

def _9dof_to_box(box, color=None):
    """Convert 9-DoF box from array/tensor to open3d.OrientedBoundingBox.

    Args:
        box (numpy.ndarray|list): 9-DoF box with shape (9,).
        color (tuple[int], optional): Color of the box. Defaults to None.

    Returns:
        OrientedBoundingBox: The converted Open3D box or None if Open3D is not available.
    """

    if isinstance(box, list):
        box = np.array(box)

    center = box[:3].reshape(3, 1)
    scale = box[3:6].reshape(3, 1)
    rot = box[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    if color is not None:
        geo.color = [x / 255.0 for x in color]

    return geo

def draw_box3d_on_image(image, box, color, label, extrinsic, intrinsic):
    """Draw 3D boxes on the image with an orientation arrow anchored to the visible centroid."""
    if box is None: # box is valid
        return image

    EPS = 1e-4
    ALPHA = 0.75
    ARROW_LENGTH = 30 # Fixed pixel length for the orientation arrow

    # Ensure matrices are NumPy arrays
    extrinsic = np.array(extrinsic)
    intrinsic = np.array(intrinsic)

    # Check matrix dimensions
    if extrinsic.shape != (4, 4) or intrinsic.shape != (4, 4):
        vprint(f"Warning: Invalid camera matrix dimensions for drawing 3D box '{label}'. Extrinsic: {extrinsic.shape}, Intrinsic: {intrinsic.shape}")
        return image

    try:
        extrinsic_w2c = np.linalg.inv(extrinsic)
    except np.linalg.LinAlgError:
        vprint(f"Warning: Extrinsic matrix for drawing 3D box '{label}' is singular. Cannot invert.")
        return image

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T

    # Check if camera is inside box
    camera_pos_in_world = (extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()
    if _inside_box(box, camera_pos_in_world):
        return image

    # Get box corners and project
    corners_world = np.asarray(box.get_box_points())
    corners_ordered_indices = [0, 1, 7, 2, 3, 6, 4, 5]
    corners_world_ordered = corners_world[corners_ordered_indices]

    corners_hom = np.concatenate([corners_world_ordered, np.ones((corners_world_ordered.shape[0], 1))], axis=1)
    corners_img = intrinsic @ extrinsic_w2c @ corners_hom.transpose()
    corners_img = corners_img.transpose()

    valid_depth_mask = corners_img[:, 2] > EPS

    corners_pixel = np.zeros((corners_img.shape[0], 2))
    for i in range(corners_img.shape[0]):
         if corners_img[i, 2] > EPS:
              corners_pixel[i] = corners_img[i][:2] / corners_img[i][2]

    # --- Draw Box Lines and Faces ---
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
             [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7],
             [0, 3, 7, 4], [1, 2, 6, 5]]

    image_with_box = image.copy()
    # Draw lines
    for line in lines:
        if corners_img[line[0], 2] > EPS and corners_img[line[1], 2] > EPS:
            px = corners_pixel[line[0]].astype(np.int32)
            py = corners_pixel[line[1]].astype(np.int32)
            cv2.line(image_with_box, tuple(px), tuple(py), color, 2)

    # Fill faces
    all_mask = np.zeros((h, w), dtype=bool)
    if np.all(valid_depth_mask):
        for face in faces:
            pts = corners_pixel[face]
            p = mpath.Path(pts[:, :2])
            try:
                mask = p.contains_points(pixel_points).reshape((h, w))
                all_mask = np.logical_or(all_mask, mask)
            except ValueError as e:
                vprint(f"Warning: Could not generate mask for face {face}. Error: {e}")

        face_pixels = image_with_box[all_mask]
        blended_color = face_pixels * ALPHA + (1 - ALPHA) * np.array(color)
        image_with_box[all_mask] = blended_color.astype(image.dtype)

    # --- Calculate Visible Centroid ---
    visible_corners_pixels = corners_pixel[valid_depth_mask]
    visible_centroid_2d = None
    if len(visible_corners_pixels) > 0:
        if len(visible_corners_pixels) >= 3:
            try:
                hull = cv2.convexHull(visible_corners_pixels.astype(np.float32))
                M = cv2.moments(hull)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    visible_centroid_2d = np.array([cx, cy])
            except Exception:
                # Fallback to mean if hull/centroid fails
                 pass
        if visible_centroid_2d is None: # Fallback if hull failed or < 3 points
            visible_centroid_2d = np.mean(visible_corners_pixels, axis=0).astype(np.int32)

        # Add text label near the centroid or top-left visible corner
        textpos = np.min(visible_corners_pixels, axis=0).astype(np.int32) # Use min for text still
        textpos[0] = np.clip(textpos[0], a_min=0, a_max=w-1)
        textpos[1] = np.clip(textpos[1], a_min=0, a_max=h-1)

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        tx1 = np.clip(textpos[0], 0, w - text_w)
        ty1 = np.clip(textpos[1], 0, h - text_h)
        tx2 = tx1 + text_w
        ty2 = ty1 + text_h

        cv2.rectangle(image_with_box, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(image_with_box, label, (tx1, ty1 + text_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Add Orientation Arrow (Anchored at Visible Centroid) ---
    if visible_centroid_2d is not None: # Only proceed if we have a valid visible centroid
        try:
            # 1. Get points along the front direction vector in world space
            center_world = box.center
            # Check if box.R is valid - sometimes might be None or incorrect
            if box.R is not None and box.R.shape == (3, 3):
                front_direction_world = box.R[:, 0]
                # Use two points along the direction vector for projection
                p1_world = center_world
                p2_world = center_world + front_direction_world # A point 1 unit in front

                # 2. Project these two points
                points_to_project = np.array([p1_world, p2_world])
                points_hom = np.concatenate([points_to_project, np.ones((2, 1))], axis=1)
                points_img = intrinsic @ extrinsic_w2c @ points_hom.transpose()
                points_img = points_img.transpose()

                # 3. Calculate 2D direction *only if both points project validly*
                if points_img[0, 2] > EPS and points_img[1, 2] > EPS:
                    proj_p1 = (points_img[0][:2] / points_img[0][2])
                    proj_p2 = (points_img[1][:2] / points_img[1][2])

                    arrow_direction_2d = proj_p2 - proj_p1
                    norm = np.linalg.norm(arrow_direction_2d)

                    if norm > EPS: # Ensure direction is valid
                        arrow_direction_2d /= norm # Normalize

                        # 4. Calculate arrow start and end points
                        arrow_start = visible_centroid_2d
                        arrow_end = (arrow_start + arrow_direction_2d * ARROW_LENGTH).astype(np.int32)

                        # 5. Clip and Draw
                        as_x, as_y = np.clip(arrow_start[0], 0, w-1), np.clip(arrow_start[1], 0, h-1)
                        ae_x, ae_y = np.clip(arrow_end[0], 0, w-1), np.clip(arrow_end[1], 0, h-1)

                        if as_x != ae_x or as_y != ae_y: # Draw if distinct after clipping
                            arrow_color = tuple(int(c * 0.7) for c in color)
                            cv2.arrowedLine(image_with_box, (as_x, as_y), (ae_x, ae_y),
                                            arrow_color, 2, tipLength=0.3)
            # else: # Do not draw arrow if direction projection is invalid or R is invalid
            #     pass

        except Exception as e_arrow:
             vprint(f"Warning: Could not draw orientation arrow for '{label}'. Error: {e_arrow}")

    return image_with_box

def _inside_box(box, point):
    """Check if any points are in the box.

    Args:
        box (open3d.geometry.OrientedBoundingBox): Oriented Box.
        point (np.ndarray): N points represented by nx4 array (x, y, z, 1).

    Returns:
        bool: The result. Returns False if Open3D not available or box invalid.
    """

    point_vec = o3d.utility.Vector3dVector(point[:, :3])
    try:
        inside_idx = box.get_point_indices_within_bounding_box(point_vec)
        return len(inside_idx) > 0
    except Exception as e:
        vprint(f"Warning: Error checking point inside box: {e}")
        return False


def close_map_window():
    """Closes the matplotlib window if it exists."""
    global map_figure_state
    if map_figure_state['fig'] is not None:
        try:
            plt.close(map_figure_state['fig'])
        except Exception:
            pass # Ignore errors if already closed
    map_figure_state['fig'] = None
    map_figure_state['ax'] = None
    map_figure_state['is_active'] = False

# Modified function to display colored occupancy map with legend
def update_map_display(occupancy_id_map, grid, object_id_to_name, title="Object Occupancy Map"):
    """Updates the matplotlib window with the colored object occupancy map."""
    global map_figure_state


    # --- Create Figure if it doesn't exist ---
    if map_figure_state['fig'] is None:
        plt.ion()
        map_figure_state['fig'], map_figure_state['ax'] = plt.subplots(figsize=(10, 10)) # Increased size again
        map_figure_state['is_active'] = True
        # Ensure window closing is handled
        map_figure_state['fig'].canvas.mpl_connect('close_event', lambda evt: close_map_window())


    fig = map_figure_state['fig']
    ax = map_figure_state['ax']
    ax.cla()

    # --- Prepare Colormap and Normalization ---
    unique_ids = np.unique(occupancy_id_map)
    # Filter out the empty ID (-1) if present
    object_ids_present = sorted([uid for uid in unique_ids if uid >= 0])

    if not object_ids_present: # Only empty space
        cmap = mcolors.ListedColormap(['white'])
        norm = mcolors.BoundaryNorm([-1.5, -0.5], cmap.N)
        legend_handles = []
    else:
        num_objects = len(object_ids_present)
        # Use a qualitative colormap like tab20, which has 20 distinct colors
        # If more objects, colors might repeat, but it's a start.
        base_cmap = plt.get_cmap('tab20')
        # Get colors for the objects present + one for empty space
        colors = ['white'] + [base_cmap(i % base_cmap.N) for i in range(num_objects)]

        cmap = mcolors.ListedColormap(colors)

        # Create boundaries for discrete colormap: -1.5 -> -0.5 (empty), -0.5 -> 0.5 (obj 0), 0.5 -> 1.5 (obj 1), ...
        boundaries = [-1.5] + [i - 0.5 for i in object_ids_present] + [object_ids_present[-1] + 0.5]
        norm = mcolors.BoundaryNorm(boundaries, cmap.N)

        # --- Create Legend Handles ---
        legend_handles = []
        for i, obj_id in enumerate(object_ids_present):
            obj_name = object_id_to_name.get(obj_id, f"Unknown ID: {obj_id}")
            patch = mpatches.Patch(color=colors[i+1], label=f"{obj_id}: {obj_name}") # i+1 because colors[0] is white for empty
            legend_handles.append(patch)

    # --- Display Map ---
    x_min, x_max = grid[0].min(), grid[0].max()
    y_min, y_max = grid[1].min(), grid[1].max()
    extent = [x_min, x_max, y_min, y_max]

    im = ax.imshow(occupancy_id_map, extent=extent, origin='lower', cmap=cmap, norm=norm, interpolation='none')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    # --- Add Legend ---
    if legend_handles:
        ax.legend(handles=legend_handles,
                  bbox_to_anchor=(0.5, -0.05), # Place below center of the axis
                  loc='upper center',          # Anchor point of the legend
                  borderaxespad=0.,
                  fontsize='x-small',        # Smaller font
                  ncol=4)                  # More columns

    # Adjust layout to prevent legend cutoff - let tight_layout handle it
    plt.tight_layout(pad=0.5)

    # --- Update Display ---
    try:
        if fig.canvas.manager is not None and hasattr(fig.canvas.manager, 'window'):
             if hasattr(fig.canvas.manager.window, 'raise_'): fig.canvas.manager.window.raise_()
             elif hasattr(fig.canvas.manager.window, 'activateWindow'): fig.canvas.manager.window.activateWindow()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001) # Add tiny pause to help event loop cooperation
    except Exception as e:
        vprint(f"Warning: Error updating map display - {e}")
        close_map_window()

# --- New Open3D Non-Blocking Visualization Thread Function ---

def o3d_visualization_thread_func():
    """
    Manages the Open3D Visualizer lifecycle and rendering loop in a separate thread.
    """
    global o3d_visualizer_state


    vis = None # Local reference
    initial_view_set = False

    try:
        with o3d_visualizer_state['lock']:
            if o3d_visualizer_state['vis'] is None:
                 vprint("Initializing Open3D Visualizer...")
                 vis = o3d.visualization.Visualizer()
                 # Increased window size for better visibility
                 vis.create_window(window_name="3D Bounding Boxes", width=1024, height=768)

                 # Improve rendering options
                 render_option = vis.get_render_option()
                 render_option.background_color = np.array([0.1, 0.1, 0.1])  # Darker background
                 render_option.point_size = 5.0
                 render_option.line_width = 2.0  # Thicker lines
                 render_option.show_coordinate_frame = True

                 # Better camera control
                 view_control = vis.get_view_control()
                 view_control.set_zoom(0.8)  # Zoom out a bit

                 o3d_visualizer_state['vis'] = vis
                 o3d_visualizer_state['is_initialized'] = True
                 vprint("Open3D Visualizer window created.")
            else:
                 vis = o3d_visualizer_state['vis'] # Reuse if somehow exists

        if vis is None:
             vprint("Error: Failed to initialize Open3D Visualizer.")
             o3d_visualizer_state['run_thread'] = False # Stop the thread loop
             return

        # Add initial coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(frame, reset_bounding_box=True) # Add frame initially

        while o3d_visualizer_state['run_thread']:

            new_geometries = None
            try:
                # Check queue non-blockingly
                new_geometries = o3d_visualizer_state['update_queue'].popleft()
            except IndexError:
                pass # Queue was empty

            if new_geometries is not None:
                 vprint(f"Updating Open3D with {len(new_geometries)} geometries...")
                 vis.clear_geometries() # Remove old geometries
                 vis.add_geometry(frame, reset_bounding_box=False) # Re-add frame

                 if new_geometries:  # Only process if we have geometries
                     reset_bounding_box_needed = True
                     for geom in new_geometries:
                          try:
                              # Ensure geometry is valid before adding
                              if geom is not None:
                                  vis.add_geometry(geom, reset_bounding_box=reset_bounding_box_needed)
                                  reset_bounding_box_needed = False  # Only reset for the first geometry
                          except Exception as e:
                              vprint(f"Error adding geometry: {e}")

                     # Explicitly reset view for better framing
                     if not initial_view_set or True:  # Always reset view when new geometries arrive
                         vprint("Resetting view...")
                         try:
                             vis.reset_view_point(True)
                             # Additional view adjustments
                             view_control = vis.get_view_control()
                             view_control.set_zoom(0.8)  # Zoom out for better view
                             initial_view_set = True
                         except Exception as e:
                             vprint(f"Error resetting view: {e}")
                 else:
                     vprint("Warning: Received empty geometries list")

            # Poll events and update renderer - crucial for non-blocking
            try:
                if not vis.poll_events():
                     vprint("Open3D window closed by user via poll_events.")
                     break # Exit loop if window closed
                vis.update_renderer()
            except Exception as e:
                vprint(f"Error in vis.poll_events() or vis.update_renderer(): {e}")
                break  # Exit the loop on rendering errors

            time.sleep(0.01) # Small sleep to prevent high CPU usage

    except Exception as e:
        vprint(f"Error in Open3D visualization thread: {e}")
    finally:
        vprint("Open3D visualization thread finishing...")
        with o3d_visualizer_state['lock']:
            if o3d_visualizer_state['vis'] is not None:
                vprint("Destroying Open3D window...")
                try:
                    o3d_visualizer_state['vis'].destroy_window()
                except Exception as e_destroy:
                    vprint(f"Warning: Error destroying O3D window: {e_destroy}")
                o3d_visualizer_state['vis'] = None
            o3d_visualizer_state['is_initialized'] = False
            o3d_visualizer_state['run_thread'] = False
            o3d_visualizer_state['thread'] = None
            # Clear queue on exit
            o3d_visualizer_state['update_queue'].clear()
        vprint("Open3D thread cleanup complete.")


def start_o3d_visualizer_thread():
    """Starts the O3D visualizer thread if not already running."""
    global o3d_visualizer_state

    with o3d_visualizer_state['lock']:
        if o3d_visualizer_state['thread'] is None or not o3d_visualizer_state['thread'].is_alive():
            vprint("Starting Open3D visualizer thread...")
            o3d_visualizer_state['run_thread'] = True
            o3d_visualizer_state['thread'] = threading.Thread(target=o3d_visualization_thread_func, daemon=True)
            o3d_visualizer_state['thread'].start()
            # Give the thread a moment to initialize the window
            # This is a bit hacky, synchronization might be better
            time.sleep(0.5)
        else:
             # Ensure run flag is set if thread exists but might have been stopped
             o3d_visualizer_state['run_thread'] = True

def stop_o3d_visualizer_thread():
    """Signals the O3D visualizer thread to stop."""
    global o3d_visualizer_state

    with o3d_visualizer_state['lock']:
        if o3d_visualizer_state['thread'] is not None and o3d_visualizer_state['thread'].is_alive():
            vprint("Signaling Open3D visualizer thread to stop...")
            o3d_visualizer_state['run_thread'] = False
        else:
             # Ensure flags are reset even if thread died unexpectedly
             o3d_visualizer_state['run_thread'] = False
             o3d_visualizer_state['thread'] = None
             o3d_visualizer_state['is_initialized'] = False
             if o3d_visualizer_state['vis'] is not None:
                  # Attempt cleanup if vis object somehow still exists
                  try: o3d_visualizer_state['vis'].destroy_window()
                  except: pass
                  o3d_visualizer_state['vis'] = None


def update_o3d_geometries(boxes, obj_id_to_name):
    """Prepares geometries and sends them to the O3D thread via queue."""
    global o3d_visualizer_state

    try:
        vprint("\n=== Starting Open3D Visualization Update ===")

        # First clear any existing geometries (send empty list)
        if o3d_visualizer_state['is_initialized']:
            o3d_visualizer_state['update_queue'].append([])
            time.sleep(0.1)  # Give time for clear to take effect

        geometries = []

        # Create the coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geometries.append(frame)

        # Get base colormap (same one used for map visualization)
        base_cmap = plt.get_cmap('tab20')

        if boxes is None: boxes = [] # Handle case where no boxes are provided
        if obj_id_to_name is None: obj_id_to_name = {}

        for obj_id, box_data in enumerate(boxes):
            # Check if box_data is valid O3D geometry
            if box_data is None or not isinstance(box_data, o3d.geometry.OrientedBoundingBox):
                 vprint(f"Warning: Skipping invalid box data at index {obj_id}")
                 continue

            obj_name = obj_id_to_name.get(obj_id, f"Unknown_{obj_id}")

            try:
                # Get box properties - with error checking
                if not hasattr(box_data, 'center') or not hasattr(box_data, 'R') or not hasattr(box_data, 'extent'):
                    continue

                center = box_data.center
                R = box_data.R
                extent = box_data.extent

                # Skip excessively large objects, likely the floor/scene bounds if misidentified
                if obj_name.lower() != 'floor' and (extent is None or np.max(extent) > 20.0):
                    vprint(f"Warning: Skipping potentially invalid large object '{obj_name}' (extent: {extent})")
                    continue
                # Also skip if specifically named floor and large
                if obj_name.lower() == 'floor' and (extent is not None and np.max(extent) > 20.0):
                     continue

                # Get color from colormap for consistency with top-down map
                color_idx = obj_id % base_cmap.N
                rgb_color = base_cmap(color_idx)[:3]  # Get RGB (exclude alpha)

                # Assign color directly to the existing box object
                box_data.color = rgb_color
                geometries.append(box_data)

            except Exception as e:
                vprint(f"Error processing box for {obj_name} (ID {obj_id}): {e}")

        # Send the geometry list to the thread queue
        if geometries:
            vprint(f"Queueing {len(geometries)} geometries for Open3D.")
            o3d_visualizer_state['update_queue'].append(geometries)
        else:
            vprint("Warning: No valid geometries generated to display in 3D view")

        vprint("=== Open3D Visualization Updated ===")

    except Exception as e:
        vprint(f"Error in Open3D update: {e}")
        import traceback
        if verbose_mode: traceback.print_exc()


# --- Main Update Function (Refactored for New Format) ---

def update_display(
    image,
    annotation_data,
    mode,
    # Indices/data identifying the item to display for the current mode:
    unary_relation_idx, # For context mode
    frame_idx,          # For context mode
    direction_idx,      # For context mode
    pair_relation_idx,  # For pair mode
    # Flags indicating what to draw:
    object_2d_grounding,
    object_3d_grounding,
    context,          # Corresponds to unary relations
    configuration,    # Corresponds to pairwise relations
    compatibility,    # Corresponds to pairwise relations & map/3D view trigger
    # Other data:
    overlay_text,
    all_boxes_for_map=None, # List of OBBs for map/3D
    floor_box_for_map=None, # OBB or bounds for map
    map_obj_id_to_name=None # Dict: object_grounding index -> name
):
    global prev_button_rect, next_button_rect, map_figure_state, o3d_visualizer_state

    display_image = image.copy()
    img_h, img_w = display_image.shape[:2]

    # --- Button Setup ---
    if prev_button_rect is None:
         prev_button_rect = (PADDING, img_h - BUTTON_HEIGHT - PADDING, BUTTON_WIDTH, BUTTON_HEIGHT)
    if next_button_rect is None:
         next_button_rect = (img_w - BUTTON_WIDTH - PADDING, img_h - BUTTON_HEIGHT - PADDING, BUTTON_WIDTH, BUTTON_HEIGHT)

    # --- Data Access (New Format) ---
    object_grounding_data = annotation_data.get("object_grounding", [])
    spatial_relationships = annotation_data.get("spatial_relationships", {})
    unary_relations = spatial_relationships.get("unary_relations", [])
    pairwise_relations = spatial_relationships.get("pairwise_relations", [])
    # Get camera parameters safely
    extrinsic = np.array(annotation_data["camera_annotations"].get('extrinsic', np.eye(4)))
    intrinsic = np.array(annotation_data["camera_annotations"].get('intrinsic', np.eye(4)))

    # Helper to find object index in object_grounding_data
    def find_object_index(name, category=None):
        for idx, obj in enumerate(object_grounding_data):
            # Match primarily on name, optionally on category if provided
            if obj.get("Name") == name and (category is None or obj.get("category") == category):
                return idx
        return -1 # Not found


    # --- Determine if Map/3D View should be active ---
    should_map_and_3d_be_active = (mode == 'pair' and compatibility and
                                   all_boxes_for_map is not None and
                                   floor_box_for_map is not None and
                                   map_obj_id_to_name is not None)

    # --- Manage Map Window ---
    if not should_map_and_3d_be_active and map_figure_state['is_active']:
        close_map_window()

    # --- Manage Open3D Window ---
    if should_map_and_3d_be_active:
        start_o3d_visualizer_thread() # Ensure thread is running
        if o3d_visualizer_state['is_initialized']: # Only update if window is ready
            update_o3d_geometries(all_boxes_for_map, map_obj_id_to_name)
        else:
            vprint("Waiting for O3D visualizer to initialize...") # Might happen on first call

    else: # Should not be active
        stop_o3d_visualizer_thread() # Signal thread to stop if running

    # --- Mode-Specific Drawing & Logic ---
    if mode == 'context':
        # --- Context Mode (Unary Relations) ---
        if context and 0 <= unary_relation_idx < len(unary_relations):
            relation_data = unary_relations[unary_relation_idx]
            rel_obj_name = relation_data.get("Name")
            rel_obj_category = relation_data.get("category")

            # Find the corresponding object in object_grounding
            primary_obj_idx = find_object_index(rel_obj_name, rel_obj_category)

            if primary_obj_idx != -1:
                primary_obj_data = object_grounding_data[primary_obj_idx]
                display_name = f"{rel_obj_name}" # ({rel_obj_category})" # Use name from relation

                frames = ["objectcentric", "cameracentric", "worldcentric"]
                directions = ["infront", "behind", "right", "left", "above", "below"] # Assume these standard directions
                current_frame = frames[frame_idx] if 0 <= frame_idx < len(frames) else "invalid_frame"
                current_direction = directions[direction_idx] if 0 <= direction_idx < len(directions) else "invalid_direction"
                # vprint(f"Drawing context: Object='{display_name}', Frame={current_frame}, Direction={current_direction}") # Covered by overlay

                # Draw primary object bounding boxes
                if object_2d_grounding and "bbox" in primary_obj_data:
                    draw_bbox_rect(display_image, primary_obj_data["bbox"], color=(0, 165, 255))

                if object_3d_grounding and "bbox_3d" in primary_obj_data:
                    try:
                        bbox_9dof = primary_obj_data["bbox_3d"] # Assume it's the 9dof list/array
                        if bbox_9dof: # Check if bbox_3d exists and is not empty
                             color = (0, 165, 255)
                             o3d_box = _9dof_to_box(bbox_9dof, color)
                             display_image = draw_box3d_on_image(display_image, o3d_box, color, display_name, extrinsic, intrinsic)
                    except Exception as e:
                        vprint(f"Error rendering PRIMARY 3D box '{display_name}' (Context Mode): {e}")

                # Draw context points from the relation data
                point_space_2d = relation_data.get("point_space_2d", {})
                frame_points = point_space_2d.get(current_frame, {})
                points = frame_points.get(current_direction, [])

                if points:
                    vprint(f"Drawing {len(points)} points for {display_name}/{current_frame}/{current_direction}")
                    draw_points(display_image, points, color=(255, 0, 255), radius=4) # Use a distinct color
                else:
                    vprint(f"No points found for {display_name}/{current_frame}/{current_direction}")

            else:
                 vprint(f"Warning: Could not find object grounding for unary relation: name='{rel_obj_name}', category='{rel_obj_category}'")


    elif mode == 'pair':
        # --- Pair Mode (Pairwise Relations) ---
        if (configuration or compatibility) and 0 <= pair_relation_idx < len(pairwise_relations):
            relationship_data = pairwise_relations[pair_relation_idx]

            # Extract pair names/categories using the "pair" key
            pair_names = relationship_data.get("pair", ["N/A", "N/A"]) # Get the list from "pair" key
            obj1 = pair_names[0] if len(pair_names) > 0 else "N/A"
            obj2 = pair_names[1] if len(pair_names) > 1 else "N/A"

            # vprint(f"\n--- Displaying Pair: ({obj1}, {obj2}) ---") # Covered by overlay

            # Find corresponding indices in object_grounding
            idx1 = find_object_index(obj1)
            idx2 = find_object_index(obj2)

            # --- Calculate and Display Colored Occupancy Map ---
            if should_map_and_3d_be_active: # Check combined flag
                 try:
                    grid = create_floor_grid(floor_box_for_map, grid_resolution=0.05)
                    occupancy_id_map = create_object_occupancy_map(grid, all_boxes_for_map, map_obj_id_to_name)
                    map_title = f"Occupancy Map - Pair: {obj1} & {obj2}"
                    update_map_display(occupancy_id_map, grid, map_obj_id_to_name, title=map_title)
                    map_figure_state['is_active'] = True # Set map active flag
                 except ValueError as e_val: # Catch specific error from grid creation
                     vprint(f"Error preparing occupancy map (likely invalid floor box): {e_val}")
                     close_map_window()
                     # Disable further attempts for this run if prep failed
                     compatibility = False # Turn off flag to prevent retries
                 except Exception as e_map:
                      vprint(f"Error generating/displaying occupancy map: {e_map}")
                      close_map_window()
                      compatibility = False # Turn off flag
                      stop_o3d_visualizer_thread()


            # --- Draw bounding boxes for the pair ---
            indices_to_draw = {}
            if idx1 != -1: indices_to_draw[idx1] = (0, 255, 0)
            else: vprint(f"  Warning: Object 1 '{obj1}' not found in grounding data.")
            if idx2 != -1: indices_to_draw[idx2] = (255, 255, 0) if idx1 != idx2 else (0, 255, 0) # Use same color if self-pair
            else: vprint(f"  Warning: Object 2 '{obj2}' not found in grounding data.")

            for current_obj_idx, box_color in indices_to_draw.items():
                if current_obj_idx < len(object_grounding_data):
                    current_obj_data = object_grounding_data[current_obj_idx]
                    current_obj_name = current_obj_data.get("Name", "Unknown") # Get name from grounding data
                    # box_color_3d_clip = box_color # No longer used
                    box_color_9dof = tuple(int(c*0.8) for c in box_color) # Slightly darker for 3D

                    if object_2d_grounding and "bbox" in current_obj_data:
                        draw_bbox_rect(display_image, current_obj_data["bbox"], color=box_color)

                    if object_3d_grounding and "bbox_3d" in current_obj_data:
                        try:
                            bbox_9dof = current_obj_data["bbox_3d"]
                            if bbox_9dof:
                                 o3d_b = _9dof_to_box(bbox_9dof, box_color_9dof)
                                 display_image = draw_box3d_on_image(display_image, o3d_b, box_color_9dof, current_obj_name, extrinsic, intrinsic)
                        except Exception as e: vprint(f"  Error rendering 3D box for '{current_obj_name}' (Pair Mode): {e}")


            # --- Print configuration/compatibility from the relationship data ---
            if configuration and "spatial_configuration" in relationship_data:
                print(f"    Configuration for Pair: ({obj1}, {obj2}):")
                config = relationship_data["spatial_configuration"]
                for frame, relations in config.items():
                    # Assuming relations is a dict like {"left": true, "right": false}
                    active_relations = [rel for rel, is_active in relations.items() if is_active]
                    print(f"      {frame}: {', '.join(active_relations) if active_relations else 'none'}")

            if compatibility and "spatial_compatibility" in relationship_data:
                print(f"    Compatibility for Pair: ({obj1}, {obj2}):")
                compat = relationship_data["spatial_compatibility"]
                for frame, directions in compat.items():
                    # Assuming directions is a dict like {"left": true, "behind": false}
                    compatible_dirs = [dir_name for dir_name, is_compatible in directions.items() if is_compatible]
                    print(f"      {frame}: {', '.join(compatible_dirs) if compatible_dirs else 'none'}")

            # vprint("-" * 30) # Optional verbose separator
        elif not (configuration or compatibility):
             pass # No pair mode flags enabled
        else:
             vprint(f"Warning: Invalid pair_relation_idx: {pair_relation_idx}")


    # Draw the main overlay text (passed from the main loop) at the top-left
    if overlay_text:
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(display_image, (0, 0), (text_width + 10, text_height + baseline + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(display_image, overlay_text, (5, text_height + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # --- Draw Buttons ---
    # Previous Button
    if prev_button_rect:
        cv2.rectangle(display_image, prev_button_rect[:2], (prev_button_rect[0] + prev_button_rect[2], prev_button_rect[1] + prev_button_rect[3]), (200, 200, 200), -1)
        cv2.putText(display_image, "Prev", (prev_button_rect[0] + 5, prev_button_rect[1] + BUTTON_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
    # Next Button
    if next_button_rect:
        cv2.rectangle(display_image, next_button_rect[:2], (next_button_rect[0] + next_button_rect[2], next_button_rect[1] + next_button_rect[3]), (200, 200, 200), -1)
        cv2.putText(display_image, "Next", (next_button_rect[0] + 5, next_button_rect[1] + BUTTON_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    cv2.imshow("Image with Annotations", display_image)

# Function to project the bounding box to the floor (2D) - needed for map
def project_to_floor(box):
    """Projects OBB corners to XY plane."""
    try:
        corners = np.asarray(box.get_box_points())
        return corners[:, :2] # Return only the x, y coordinates
    except Exception as e:
        vprint(f"Warning: Failed to project box to floor: {e}")
        return None


# Function to create the grid representing the floor - needed for map
def create_floor_grid(floor_box, grid_resolution=0.1):
    """Creates a 2D grid based on floor bounds or OBB."""

    min_bound, max_bound = None, None
    if isinstance(floor_box, list) and len(floor_box) == 2: # Explicit Bounds representation [min_vec, max_vec]
        min_b = np.array(floor_box[0])
        max_b = np.array(floor_box[1])
        if min_b.shape[0] >= 2 and max_b.shape[0] >= 2:
            min_bound = min_b[:2]
            max_bound = max_b[:2]
    elif isinstance(floor_box, o3d.geometry.OrientedBoundingBox): # OBB representation
        try:
            min_bound = floor_box.get_min_bound()[:2]
            max_bound = floor_box.get_max_bound()[:2]
        except Exception as e:
             raise ValueError(f"Could not get bounds from floor OBB: {e}")
    elif isinstance(floor_box, o3d.geometry.AxisAlignedBoundingBox): # AABB representation
         try:
            min_bound = floor_box.get_min_bound()[:2]
            max_bound = floor_box.get_max_bound()[:2]
         except Exception as e:
              raise ValueError(f"Could not get bounds from floor AABB: {e}")


    if min_bound is None or max_bound is None:
         raise ValueError("Invalid or unusable floor_box representation for grid creation.")

    # Ensure bounds are reasonable (e.g., min < max)
    if np.any(min_bound >= max_bound):
        # Attempt to swap if inverted, but warn if dimensions are still zero/negative
        swapped = False
        new_min = min_bound.copy()
        new_max = max_bound.copy()
        for i in range(2):
            if min_bound[i] > max_bound[i]:
                new_min[i], new_max[i] = max_bound[i], min_bound[i]
                swapped = True
        if swapped: vprint("Warning: Floor bounds seem inverted, attempting swap.")
        min_bound = new_min
        max_bound = new_max
        if np.any(min_bound >= max_bound):
             raise ValueError(f"Invalid floor bounds after potential swap: min={min_bound}, max={max_bound}")


    x_range = np.arange(min_bound[0], max_bound[0], grid_resolution)
    y_range = np.arange(min_bound[1], max_bound[1], grid_resolution)

    # Ensure ranges are not empty after arange
    if len(x_range) == 0: x_range = np.array([min_bound[0]])
    if len(y_range) == 0: y_range = np.array([min_bound[1]])

    return np.meshgrid(x_range, y_range)

# Function to create the object occupancy map
def create_object_occupancy_map(grid, boxes, object_id_to_name):
    """Creates a map where grid cells contain the ID of the occupying object."""
    from scipy.spatial import ConvexHull # Local import OK here

    # Initialize map with -1 (empty)
    occupancy_id_map = np.full(grid[0].shape, -1, dtype=int)

    # Flatten the grid points
    x_flat = grid[0].ravel()
    y_flat = grid[1].ravel()
    points_array = np.column_stack((x_flat, y_flat))

    if boxes is None: boxes = [] # Handle empty input
    if object_id_to_name is None: object_id_to_name = {}

    # Iterate through boxes with their original index (which we use as ID)
    for obj_id, box in enumerate(boxes):
        if box is None: continue # Skip if OBB creation failed earlier
        obj_name = object_id_to_name.get(obj_id, f"Unknown_{obj_id}")
        if obj_name.lower() == 'floor': continue # Skip explicit floor object

        try:
            projected_points = project_to_floor(box)
            if projected_points is None or len(np.unique(projected_points, axis=0)) < 3: continue # Need 3 unique points for hull

            hull = ConvexHull(projected_points)
            hull_vertices = projected_points[hull.vertices]
            path = mpath.Path(hull_vertices)

            inside = path.contains_points(points_array)

            # Update the map - last object drawn wins in overlap
            occupancy_id_map.ravel()[inside] = obj_id
        except ImportError:
             vprint("Warning: scipy.spatial not found. Cannot compute convex hull for occupancy map.")
             return occupancy_id_map # Return partially filled map maybe? Or raise?
        except Exception as e_hull:
             vprint(f"Warning: Could not process hull/path for object '{obj_name}' (ID: {obj_id}): {e_hull}")

    return occupancy_id_map

# Main Visualization Function (Refactored for New Format)
def visualize_image_with_annotations(
    image_path,
    annotation_data,
    object_2d_grounding=False,
    object_3d_grounding=False,
    context=False,
    configuration=False,
    compatibility=False
):
    global visualization_state, map_figure_state, o3d_visualizer_state

    # --- Initial Setup ---
    image = cv2.imread(image_path)
    if image is None:
        vprint(f"Error: Could not load image {image_path}")
        return

    # --- Data Access (New Format) ---
    object_grounding_data = annotation_data.get("object_grounding", [])
    spatial_relationships = annotation_data.get("spatial_relationships", {})
    unary_relations = spatial_relationships.get("unary_relations", [])
    pairwise_relations = spatial_relationships.get("pairwise_relations", [])

    # --- Prepare Data for Map and 3D View (Calculate ONCE) ---
    all_boxes_for_map = None # List of OBBs
    floor_box_for_map = None # OBB or bounds
    map_obj_id_to_name = None # Dictionary: object_grounding index -> name

    # Map/3D view requires compatibility flag and Open3D
    if compatibility:
        try:
            temp_boxes = [] # List to hold OBBs corresponding to object_grounding indices
            min_bounds_list = []
            max_bounds_list = []
            temp_obj_id_to_name = {}
            found_floor_obb = None # Store the potential floor OBB

            vprint("Preparing OBBs for map/3D view...")
            for idx, obj_data in enumerate(object_grounding_data):
                obj_name = obj_data.get("Name", f"Unknown_{idx}")
                temp_obj_id_to_name[idx] = obj_name

                o3d_box = None # Initialize for this object
                if "bbox_3d" in obj_data and obj_data["bbox_3d"]:
                    bbox_9dof = obj_data["bbox_3d"] # Assume it's the 9dof list/array
                    try:
                        o3d_box = _9dof_to_box(bbox_9dof)
                        if o3d_box is not None:
                             min_b = o3d_box.get_min_bound()
                             max_b = o3d_box.get_max_bound()
                             # Basic check for valid bounds
                             if np.any(np.isnan(min_b)) or np.any(np.isnan(max_b)) or np.any(min_b >= max_b):
                                 vprint(f"Warning: Invalid bounds for OBB from {obj_name}. Min: {min_b}, Max: {max_b}. Skipping for map.")
                                 o3d_box = None # Invalidate box if bounds are bad
                             else:
                                 min_bounds_list.append(min_b)
                                 max_bounds_list.append(max_b)
                                 # Check if this looks like a floor object
                                 # Heuristic: Large XY extent, relatively small Z extent, close to Z=0?
                                 extent = o3d_box.extent
                                 center = o3d_box.center
                                 if extent[0] > 5 and extent[1] > 5 and extent[2] < 0.5 and abs(center[2]) < 0.5:
                                      if found_floor_obb is None: # Take the first likely candidate
                                           found_floor_obb = o3d_box
                                           vprint(f"  Identified potential floor: {obj_name}")
                                      # Add floor to main list too, might be visualized if not skipped later
                                 # Also check by name explicitly
                                 if obj_name.lower() == 'floor':
                                      found_floor_obb = o3d_box # Prioritize named floor
                                      vprint(f"  Identified named floor: {obj_name}")

                    except Exception as e_box:
                         vprint(f"Warning: Could not create/validate OBB for map/3D view from {obj_name}: {e_box}")
                         o3d_box = None # Ensure it's None if creation failed

                temp_boxes.append(o3d_box) # Append the OBB or None

            if any(b is not None for b in temp_boxes): # Check if at least one valid box was created
                all_boxes_for_map = temp_boxes # Store the list (can contain Nones)
                map_obj_id_to_name = temp_obj_id_to_name

                # Determine floor bounds for the grid
                if found_floor_obb is not None:
                    floor_box_for_map = found_floor_obb # Use the identified floor OBB
                    vprint(f"Using identified floor OBB for map grid.")
                elif min_bounds_list: # Fallback: Use scene bounds if no floor identified
                    scene_min_bound = np.min(min_bounds_list, axis=0)
                    scene_max_bound = np.max(max_bounds_list, axis=0)
                    # Create an AABB for the floor boundary based on scene extents
                    floor_box_for_map = o3d.geometry.AxisAlignedBoundingBox(min_bound=scene_min_bound, max_bound=scene_max_bound)
                    vprint(f"Using scene bounds for map grid: Min={scene_min_bound}, Max={scene_max_bound}")
                else:
                     vprint("Warning: No valid OBBs found, cannot determine floor bounds for map.")
                     # Disable map/3D features if floor bounds couldn't be determined
                     all_boxes_for_map = None
                     map_obj_id_to_name = None
                     compatibility = False # Turn off the flag

            else: # No valid boxes created at all
                 vprint("Warning: No valid OBBs created. Map/3D view disabled.")
                 compatibility = False # Turn off the flag

            # Final check if we can proceed with map/3D
            if all_boxes_for_map is None or floor_box_for_map is None or map_obj_id_to_name is None:
                 vprint("Warning: Could not prepare necessary data for occupancy map / 3D view. Views disabled.")
                 compatibility = False # Ensure flag is off

        except Exception as e_prep:
            vprint(f"Warning: Error preparing data for occupancy map / 3D view: {e_prep}. Views disabled.")
            compatibility = False

    # --- Determine cycling mode and prepare items list ---
    mode = None
    valid_items = []
    object_names = [obj.get("Name", f"Unknown_{i}") for i, obj in enumerate(object_grounding_data)] # Needed for overlay text

    # Pairwise Relations Mode (Configuration or Compatibility)
    if configuration or compatibility:
        
        if configuration or compatibility: # Proceed if at least one is still True
            mode = 'pair'
            vprint("Searching for valid pairwise relationships...")
            filtered_pair_indices = []
            for idx, relationship_data in enumerate(pairwise_relations):
                # Check if the required data exists for the enabled flags
                has_config = "spatial_configuration" in relationship_data
                has_compat = "spatial_compatibility" in relationship_data
                # Include if configuration is requested AND present OR compatibility is requested AND present
                if (configuration and has_config) or (compatibility and has_compat):
                     obj1_info = relationship_data.get("object1", {})
                     obj2_info = relationship_data.get("object2", {})
                     vprint(f"  Found valid pair: ({obj1_info.get('Name')}, {obj2_info.get('Name')}) at index {idx}")
                     filtered_pair_indices.append(idx) # Store index of the valid pairwise relation
            valid_items = filtered_pair_indices
            if not valid_items:
                 vprint("No object relationships found matching the requested --configuration/--compatibility flags.")
            else:
                 vprint(f"Found {len(valid_items)} valid pairwise relationships.")

    # Unary Relations Mode (Context) - only if pair mode wasn't activated
    if mode is None and context:
        mode = 'context'
        frames = ["objectcentric", "cameracentric", "worldcentric"]
        directions = ["infront", "behind", "right", "left", "above", "below"] # Standard directions assumed

        vprint("Searching for valid unary point annotations (--context)...")
        temp_valid_items = []
        for rel_idx, relation_data in enumerate(unary_relations):
            rel_obj_name = relation_data.get("Name", f"UnknownRel_{rel_idx}")
            vprint(f"Checking relation for: {rel_obj_name} (relation index {rel_idx})")

            point_space_2d = relation_data.get("point_space_2d", {})
            found_points_for_relation = False
            for frame_idx, frame in enumerate(frames):
                frame_data = point_space_2d.get(frame, {})
                # vprint(f"  Frame: {frame}")

                for dir_idx, direction in enumerate(directions):
                    points = frame_data.get(direction, [])
                    if points and len(points) > 0:
                        vprint(f"    Direction: {direction} - Found {len(points)} points")
                        temp_valid_items.append((rel_idx, frame_idx, dir_idx)) # Store (relation_idx, frame_idx, dir_idx)
                        found_points_for_relation = True
                    # else:
                    #     vprint(f"    Direction: {direction} - No points or empty list")

        valid_items = temp_valid_items
        if not valid_items:
            vprint("No valid unary context points found to cycle through.")
        else:
             vprint(f"Found {len(valid_items)} valid unary context point sets.")

    # --- Handle cases where no mode is active or no items found ---
    if mode is None:
        vprint("No visualization mode selected (--context, --configuration, or --compatibility). Displaying base image.")
        cv2.imshow("Image with Annotations", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    if not valid_items:
        vprint(f"No valid items found for the selected mode ('{mode}'). Displaying base image.")
        # Optionally draw grounding if enabled, even without cycling items
        display_image = image.copy()
        if object_2d_grounding or object_3d_grounding:
             cam_ann = annotation_data.get("camera_annotations", {})
             extr = np.array(cam_ann.get('extrinsic', np.eye(4)))
             intr = np.array(cam_ann.get('intrinsic', np.eye(4)))
             for obj_idx, obj_data in enumerate(object_grounding_data):
                  obj_name = obj_data.get("Name", f"Obj_{obj_idx}")
                  if object_2d_grounding and "bbox" in obj_data:
                       draw_bbox_rect(display_image, obj_data["bbox"])
                  if object_3d_grounding and "bbox_3d" in obj_data:
                       try:
                            o3d_b = _9dof_to_box(obj_data["bbox_3d"], (128, 128, 128))
                            display_image = draw_box3d_on_image(display_image, o3d_b, (128,128,128), obj_name, extr, intr)
                       except Exception as e: vprint(f"Error drawing static 3D box for {obj_name}: {e}")

        cv2.imshow("Image with Annotations", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # --- Initialize Open3D visualization thread if needed ---
    if compatibility and mode == 'pair': # Only start if compatibility is still enabled
        vprint("\n=== Initializing Open3D Visualization Thread ===")
        start_o3d_visualizer_thread()
        # Initial update might happen in the first loop iteration via update_display
        vprint("============================================\n")


    # --- Final Setup & Main Loop ---
    visualization_state['total_combinations'] = len(valid_items)
    visualization_state['current_combination_idx'] = 0
    visualization_state['needs_update'] = True

    window_name = "Image with Annotations"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, handle_mouse_click)

    # Define frames/directions again for overlay text lookup
    context_frames = ["objectcentric", "cameracentric", "worldcentric"]
    context_directions = ["infront", "behind", "right", "left", "above", "below"]

    try:
        while True:
            if visualization_state['needs_update']:
                current_idx_in_list = visualization_state['current_combination_idx']
                current_item = valid_items[current_idx_in_list] # This is either pair_relation_idx or (unary_relation_idx, frame_idx, dir_idx)

                overlay_text = ""
                unary_relation_idx = -1; frame_idx = -1; direction_idx = -1 # For context mode
                pair_relation_idx = -1 # For pair mode

                if mode == 'context':
                    unary_relation_idx, frame_idx, direction_idx = current_item # Unpack the tuple
                    # Get the corresponding relation data to find the object name
                    if 0 <= unary_relation_idx < len(unary_relations):
                         relation_data = unary_relations[unary_relation_idx]
                         obj_name = relation_data.get("Name", "N/A")
                         obj_cat = relation_data.get("category", "N/A")
                         frame_name = context_frames[frame_idx] if 0 <= frame_idx < len(context_frames) else "N/A"
                         dir_name = context_directions[direction_idx] if 0 <= direction_idx < len(context_directions) else "N/A"
                         overlay_text = f"Context: {obj_name} | {frame_name} | {dir_name} ({current_idx_in_list+1}/{len(valid_items)})"
                    else: overlay_text = f"Context Error (Idx {unary_relation_idx})"

                elif mode == 'pair':
                    pair_relation_idx = current_item # Item is just the index
                    if 0 <= pair_relation_idx < len(pairwise_relations):
                        pair_data = pairwise_relations[pair_relation_idx]
                        # Get object names using the correct key from pair_data
                        pair_names_list = pair_data.get("pair", ["N/A", "N/A"]) # Get list from "pair" key
                        obj1_name = pair_names_list[0] if len(pair_names_list) > 0 else "N/A"
                        obj2_name = pair_names_list[1] if len(pair_names_list) > 1 else "N/A"
                        overlay_text = f"Pair: {obj1_name} & {obj2_name} ({current_idx_in_list+1}/{len(valid_items)}) [RelIdx:{pair_relation_idx}]"
                    else: overlay_text = f"Pair Error (Idx {pair_relation_idx})"
                # --- End Overlay Text Generation ---

                # Call update_display with all necessary info
                update_display(
                    image, annotation_data, mode,
                    unary_relation_idx, frame_idx, direction_idx, # Context params
                    pair_relation_idx,                            # Pair param
                    object_2d_grounding, object_3d_grounding,     # Grounding flags
                    context, configuration, compatibility,        # Mode flags
                    overlay_text,                                 # Text
                    all_boxes_for_map, floor_box_for_map, map_obj_id_to_name # Map/3D data
                )
                visualization_state['needs_update'] = False

            key = cv2.waitKey(20) & 0xFF # Use a small wait time
            if key == ord('q'):
                break

            # Check map window status
            if map_figure_state['is_active'] and map_figure_state['fig'] is not None:
                # Check if the figure still exists
                try:
                    if not plt.fignum_exists(map_figure_state['fig'].number):
                        vprint("Map window closed by user.")
                        close_map_window() # Ensure state is reset
                except Exception: # Handle cases where fig might become invalid
                     close_map_window()


            # Check if the O3D thread has finished (e.g., user closed window)
            with o3d_visualizer_state['lock']:
                if o3d_visualizer_state['thread'] is not None and not o3d_visualizer_state['thread'].is_alive() and o3d_visualizer_state['run_thread']:
                    # If thread died but was supposed to be running
                    vprint("Detected O3D thread finished unexpectedly.")
                    # Reset state variables, the thread's finally block should also do this ideally
                    o3d_visualizer_state['thread'] = None
                    o3d_visualizer_state['is_initialized'] = False
                    o3d_visualizer_state['vis'] = None # Ensure vis object is cleared
                    o3d_visualizer_state['run_thread'] = False


    finally:
        vprint("Cleaning up visualization...")
        cv2.destroyAllWindows()
        close_map_window()
        stop_o3d_visualizer_thread() # Signal O3D thread to stop on exit

        # Optional: Wait briefly for the thread to finish cleanup
        thread_to_join = None
        with o3d_visualizer_state['lock']: # Access thread safely
                thread_to_join = o3d_visualizer_state['thread']

        if thread_to_join is not None and thread_to_join.is_alive():
            vprint("Waiting briefly for O3D thread to exit...")
            thread_to_join.join(timeout=1.0) # Wait max 1 second
            if thread_to_join.is_alive():
                vprint("O3D thread did not exit cleanly.")

        plt.ioff()
        vprint("Visualization finished.") # Keep final message

def main():
    global verbose_mode # Allow modification of the global flag
    parser = argparse.ArgumentParser(description="Visualize annotations on images.")
    parser.add_argument('--image_path', type=str, required=True, help='Direct path to the image file to visualize.')
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to the specific JSON annotation file.')
    parser.add_argument('--verbose', '-v', action='store_true', help='If set, prints detailed status messages and warnings.')


    # Object grounding options
    grounding_group = parser.add_argument_group('Object Grounding')
    grounding_group.add_argument('--object_2d_grounding', action='store_true', help='Overlay 2D rectangular bounding boxes (bbox field) on images.')
    grounding_group.add_argument('--object_3d_grounding', action='store_true', help='Overlay 3D bounding boxes (bbox_3d field) on images.')

    # Spatial relationship options
    spatial_group = parser.add_argument_group('Spatial Relationships')
    spatial_group.add_argument('--context', action='store_true', help='Cycle through unary relations (points in space relative to single objects).')
    spatial_group.add_argument('--configuration', action='store_true', help='Cycle through pairwise relations, showing spatial configuration.')
    spatial_group.add_argument('--compatibility', action='store_true', help='Cycle through pairwise relations, showing compatibility and enabling map/3D view (requires open3d).')

    args = parser.parse_args()
    verbose_mode = args.verbose # Set the global flag

    # Make Open3D optional at runtime based on flag and import success
    if args.compatibility:
        # Check if Open3D was successfully imported at the top
        if 'o3d' not in globals():
             vprint("Error: --compatibility flag requires the 'open3d' library, but it could not be imported.")
             vprint("Please install it (e.g., 'pip install open3d') or run without --compatibility.")
             return # Exit if required library is missing
        # Additionally, check if the import actually worked (sometimes imports fail silently)
        try:
             o3d.geometry.TriangleMesh() # Try creating a basic object
        except NameError:
             vprint("Error: --compatibility flag requires the 'open3d' library, but it seems it failed to initialize correctly.")
             vprint("Please check your installation or run without --compatibility.")
             return


    # Load annotation data first to potentially get other info if needed
    try:
        annotation_data = load_annotations(args.annotation_file)
    except FileNotFoundError:
        vprint(f"Error: Annotation file not found at {args.annotation_file}")
        return
    except json.JSONDecodeError:
        vprint(f"Error: Could not decode JSON from {args.annotation_file}")
        return
    except Exception as e:
        vprint(f"Error loading annotations: {e}")
        return

    # Use the provided image path directly
    image_path = args.image_path

    # Check if the image file exists
    if not os.path.isfile(image_path):
        vprint(f"Error: Image file not found at the provided path: {image_path}")
        return

    vprint(f"Loading image: {image_path}")
    vprint(f"Visualizing annotations from: {args.annotation_file}")
    vprint(f"Modes enabled: 2D={args.object_2d_grounding}, 3D={args.object_3d_grounding}, Context={args.context}, Config={args.configuration}, Compat={args.compatibility}")


    visualize_image_with_annotations(
        image_path,
        annotation_data,
        object_2d_grounding=args.object_2d_grounding,
        object_3d_grounding=args.object_3d_grounding,
        context=args.context,
        configuration=args.configuration,
        compatibility=args.compatibility
    )

if __name__ == "__main__":
    main() 