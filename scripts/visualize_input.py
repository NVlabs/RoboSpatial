# Create a new file named visualize_simple.py
import os
import json
import cv2
import argparse
import numpy as np
import open3d as o3d
import matplotlib.path as mpath # Needed for face filling in draw_box3d

# --- Utility Functions (Copied and potentially simplified) ---

def _9dof_to_box(box_params, color=None):
    """Convert 9-DoF box from array/tensor to open3d.OrientedBoundingBox.

    Args:
        box_params (numpy.ndarray|list): 9-DoF box [cx, cy, cz, sx, sy, sz, rx, ry, rz].
        color (tuple[int], optional): RGB Color of the box (0-255). Defaults to None.

    Returns:
        open3d.geometry.OrientedBoundingBox: The converted Open3D box.
    """
    if isinstance(box_params, list):
        box_params = np.array(box_params)

    center = box_params[:3].reshape(3, 1)
    scale = box_params[3:6].reshape(3, 1)
    rot = box_params[6:].reshape(3, 1)
    rot_mat = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_zxy(rot)
    geo = o3d.geometry.OrientedBoundingBox(center, rot_mat, scale)

    if color is not None:
        geo.color = [x / 255.0 for x in color] # O3D uses 0-1 range

    return geo

def _inside_box(box, point):
    """Check if any points are inside the box.

    Args:
        box (open3d.geometry.OrientedBoundingBox): Oriented Box.
        point (np.ndarray): N points represented by nx3 array (x, y, z).

    Returns:
        bool: True if any point is inside, False otherwise.
    """
    # Reference logic uses nx4, check if conversion needed
    if point.shape[1] == 4:
        point = point[:, :3]
    point_vec = o3d.utility.Vector3dVector(point)
    inside_idx = box.get_point_indices_within_bounding_box(point_vec)
    return len(inside_idx) > 0

# Replaced with logic from visualization/img_drawer.py:draw_box3d
def draw_box3d_on_image(image, box, color, label, extrinsic, intrinsic):
    """Draw 3D boxes on the image, exactly matching img_drawer.py logic.

    Args:
        image (np.ndarray): The image to draw on.
        box (open3d.geometry.OrientedBoundingBox): Box to be drawn.
        color (tuple): Box color.
        label (str): Box category label.
        extrinsic (np.ndarray): 4x4 extrinsic matrix (axis_align @ cam2global).
        intrinsic (np.ndarray): 4x4 camera intrinsic matrix.

    Returns:
        np.ndarray: Image with the box drawn.
    """
    EPS = 1e-4  # Epsilon from img_drawer
    ALPHA = 0.75  # Alpha from img_drawer (was 0.6)

    extrinsic_w2c = np.linalg.inv(extrinsic)
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x.flatten(), y.flatten()
    pixel_points = np.vstack((x, y)).T

    # Fix 1: Use transpose() as in original code
    camera_pos_in_world = (extrinsic @ np.array([0, 0, 0, 1]).reshape(4, 1)).transpose()
    if _inside_box(box, camera_pos_in_world):
        return image

    corners = np.asarray(box.get_box_points())
    corners = corners[[0, 1, 7, 2, 3, 6, 4, 5]]  # Specific corner order from img_drawer
    corners = np.concatenate([corners, np.ones((corners.shape[0], 1))], axis=1)
    
    # Same projection as img_drawer: intrinsic @ extrinsic_w2c @ corners.transpose()
    corners_img = intrinsic @ extrinsic_w2c @ corners.transpose()
    corners_img = corners_img.transpose()
    
    corners_pixel = np.zeros((corners_img.shape[0], 2))
    
    # Fix 2: Use np.abs() in division exactly as in img_drawer
    for i in range(corners_img.shape[0]):
        corners_pixel[i] = corners_img[i][:2] / np.abs(corners_img[i][2])

    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7],
             [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7],
             [0, 3, 7, 4], [1, 2, 6, 5]]

    image_with_box = image.copy()

    # Fix 3: Use exact depth check from img_drawer for lines
    for line in lines:
        # This is the exact check from img_drawer
        if (corners_img[line][:, 2] < EPS).any():
            continue
        px = corners_pixel[line[0]].astype(np.int32)
        py = corners_pixel[line[1]].astype(np.int32)
        cv2.line(image_with_box, (px[0], px[1]), (py[0], py[1]), color, 2)

    # Fix 4: Use exact mask/face handling from img_drawer
    all_mask = np.zeros((h, w), dtype=bool)
    for face in faces:
        # This is the exact check from img_drawer
        if (corners_img[face][:, 2] < EPS).any():
            continue
        pts = corners_pixel[face]
        p = mpath.Path(pts[:, :2])
        mask = p.contains_points(pixel_points).reshape((h, w))
        all_mask = np.logical_or(all_mask, mask)
    
    # Apply color blend - exact formula from img_drawer
    image_with_box[all_mask] = image_with_box[all_mask] * ALPHA + (1 - ALPHA) * np.array(color)

    # Draw text label if any faces were visible
    if all_mask.any():
        textpos = np.min(corners_pixel, axis=0).astype(np.int32)
        textpos[0] = np.clip(textpos[0], a_min=0, a_max=w)
        textpos[1] = np.clip(textpos[1], a_min=0, a_max=h)
        
        # Simple text drawing to mimic self.draw_text from img_drawer
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        
        # Draw background box and text
        cv2.rectangle(image_with_box, 
                     (textpos[0], textpos[1]), 
                     (textpos[0] + text_w, textpos[1] + text_h),
                     color, -1)
        cv2.putText(image_with_box, label, 
                   (textpos[0], textpos[1] + text_h),
                   font, font_scale, (255, 255, 255), thickness)

    return image_with_box

# --- Main Visualization Logic ---

def visualize_single_image(image_path, annotation_data):
    """Loads image and draws 3D boxes based on annotation data."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Extract camera info
    cam_ann = annotation_data.get("camera_annotations")
    if not cam_ann:
        print("Error: 'camera_annotations' not found in JSON.")
        return
    try:
        # extrinsic is Camera -> World
        extrinsic = np.array(cam_ann['extrinsic'])
        # Intrinsic matrix
        intrinsic = np.array(cam_ann['intrinsic'])

    except KeyError as e:
        print(f"Error: Missing camera parameter key: {e}")
        return
    # Removed LinAlgError check here as inversion happens in drawing function now
    except Exception as e:
         print(f"Error processing camera parameters: {e}")
         return


    # Extract object grounding info
    object_grounding = annotation_data.get("objects", [])
    if not object_grounding:
        print("Warning: 'objects' array is missing or empty.")
        # Display original image if no objects
        cv2.imshow("Image with 3D Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    display_image = image.copy()
    # Ensure matplotlib is imported for colormap and path
    try:
        import matplotlib.pyplot as plt
        # Ensure mpath is imported here as it's needed by draw_box3d
        import matplotlib.path as mpath
        colors = plt.colormaps['tab10'] # Get distinct colors - Updated API
    except ImportError:
        print("Error: Matplotlib required for colormap and face drawing. Please install.")
        # Fallback to manual colors if matplotlib fails
        colors = lambda i: [(255,0,0), (0,255,0), (0,0,255)][i % 3]


    for i, obj_data in enumerate(object_grounding):
        obj_name = obj_data.get("Name", f"Object_{i+1}")
        bbox_3d_list = obj_data.get("bbox_3d")

        if bbox_3d_list:
            # Assuming the first bbox in the list is the one to draw
            bbox_9dof = bbox_3d_list[0]

            # Get color
            if callable(colors): # Check if it's a colormap function or fallback list
                 color_float = colors(i)[:3] # Get RGB, discard alpha
                 color_uint8 = tuple(int(c * 255) for c in color_float)
            else: # Fallback list
                 color_uint8 = colors[i % len(colors)] # Use modulo for safety


            try:
                # Box center/extent/rotation are in Aligned World space from JSON
                o3d_box = _9dof_to_box(bbox_9dof, color=color_uint8)
                # Pass the combined extrinsic and intrinsic to the drawing function
                display_image = draw_box3d_on_image(
                    display_image,
                    o3d_box,
                    color_uint8,
                    obj_name,
                    extrinsic, # Combined matrix (axis_align @ cam2global)
                    intrinsic  # Camera intrinsics (K)
                )
            except Exception as e:
                print(f"Error processing/drawing box for '{obj_name}': {e}")
                import traceback
                traceback.print_exc() # More detailed error for debugging
        else:
            print(f"Warning: No 'bbox_3d' found for object '{obj_name}'.")

    # Display the result
    cv2.imshow("Image with 3D Boxes", display_image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 3D bounding boxes from an annotation file on an image.")
    parser.add_argument('--image_path', type=str, required=True,
                        help='Direct path to the image file.')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='Path to the JSON annotation file.')
    args = parser.parse_args()

    # Load annotation data
    try:
        with open(args.annotation_file, 'r') as f:
            annotation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {args.annotation_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {args.annotation_file}")
        exit(1)
    except Exception as e:
         print(f"An unexpected error occurred loading annotations: {e}")
         exit(1)

    # Use the provided image path directly
    image_path = args.image_path

    # Check if the provided image file exists before proceeding
    if not os.path.isfile(image_path):
         print(f"Error: Image file not found at the provided path: {image_path}")
         exit(1) # Exit if the primary path doesn't exist

    # Import matplotlib here to avoid making it a hard dependency if not needed
    # although draw_box3d currently needs mpath
    try:
         import matplotlib.pyplot as plt
         # Make sure mpath is imported within the try block as well
         import matplotlib.path as mpath
    except ImportError:
         print("Error: Matplotlib is required by the drawing function. Please install it (`pip install matplotlib`).")
         exit(1)


    visualize_single_image(image_path, annotation_data)
    print("Visualization finished.")