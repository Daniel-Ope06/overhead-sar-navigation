import bpy
import bmesh
import bpy_extras
import random
import math
import os

# --- Configuration ---
NUM_IMAGES = 5

# Get the directory of the currently open .blend file
if not bpy.data.filepath:
    raise Exception("ERROR: This script must run in a .blend file")

BLEND_DIR = os.path.dirname(bpy.data.filepath)
OUTPUT_DIR = os.path.join(BLEND_DIR, "synthetic_dataset")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")

# YOLO Class IDs
CLASS_UGV = 0
CLASS_HUMAN = 1


def setup_directories():
    """
    Creates the required output directories if they do not exist.
    Separates images and labels to adhere to standard YOLO training structures.
    """
    for directory in [IMAGE_DIR, LABEL_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)


def create_maze(grid_obj):
    """
    Applies a Depth First Search algorithm to carve a random maze.
    Closes all edges of the grid object and incrementally
    opens a path ensuring all cells remain accessible.
    """
    if grid_obj.type != 'MESH':  # type: ignore
        raise Exception("Must be a mesh object")

    assert (grid_obj is not None)
    assert (grid_obj.modifiers is not None)

    # Verify object has maze nodes modifier
    maze_nodes = bpy.data.node_groups['Maze']
    maze_modifier = None

    for modifier in grid_obj.modifiers:
        if modifier.type == 'NODES' and modifier.node_group == maze_nodes:
            maze_modifier = modifier
            break

    if not maze_modifier:
        maze_modifier = grid_obj.modifiers.new('Maze', 'NODES')  # type: ignore
        maze_modifier.node_group = bpy.data.node_groups['Maze']
        for i in range(len(grid_obj.modifiers) - 1):  # type: ignore
            bpy.ops.object.modifier_move_up(
                modifier=maze_modifier.name)  # type: ignore

    # Verify object mesh has 'closed' attribute
    mesh = grid_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)  # type: ignore
    closed = mesh.attributes.get('closed')  # type: ignore
    if not closed:
        closed = mesh.attributes.new(  # type: ignore
            name='closed', type='BOOLEAN', domain='EDGE'
        )
        closed.name = 'closed'

    # Reset the maze by closing all edges
    for edge in bm.edges:
        setattr(closed.data[edge.index], 'value', True)

    # Randomly chose a cell to start carving the maze from
    current_cell = random.choice(list(bm.faces))
    visited = [current_cell]
    stack = [current_cell]

    # Loop until the entire maze is carved out
    while len(stack) > 0:
        current_cell = stack.pop()

        # Get unvisited neighbors to current cell
        unvisited_neighbors = []
        for edge in current_cell.edges:
            for neighbor in edge.link_faces:
                if neighbor != current_cell and neighbor not in visited:
                    unvisited_neighbors.append(neighbor)

        # Randomly choose an unvisited neighbor
        if len(unvisited_neighbors) > 0:
            stack.append(current_cell)
            chosen_neighbor = random.choice(unvisited_neighbors)
            visited.append(chosen_neighbor)
            stack.append(chosen_neighbor)

            # Open all edges between current cell and chosen neighbor
            for edge in current_cell.edges:
                if edge in chosen_neighbor.edges:
                    setattr(closed.data[edge.index], 'value', False)

    # Update the mesh to show the maze
    mesh.update()  # type: ignore


def position_actors(grid_obj, ugv_obj, human_obj):
    """
    Randomly places the UGV and Human at two distinct cells on the grid,
    sets their Z height, and randomizes their Z-axis rotation
    """
    mesh = grid_obj.data

    # Randomly picks 2 distinct faces to prevent overlap
    chosen_faces = random.sample(list(mesh.polygons), 2)
    ugv_face = chosen_faces[0]
    human_face = chosen_faces[1]

    # Calculate global coordinates
    ugv_global_loc = grid_obj.matrix_world @ ugv_face.center
    human_global_loc = grid_obj.matrix_world @ human_face.center

    # Apply the locations
    ugv_obj.location = (ugv_global_loc.x, ugv_global_loc.y, 0.33)
    human_obj.location = (human_global_loc.x, human_global_loc.y, -0.55)

    # Apply random rotation
    # Human: Continuously random (0 to 360 degrees)
    human_obj.rotation_euler.z = random.uniform(0, 2 * math.pi)

    # UGV: Strictly N, E, S, W (0, 90, 180, 270 degrees)
    cardinal_angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2]
    ugv_obj.rotation_euler.z = random.choice(cardinal_angles)

    # Update the scene
    bpy.context.view_layer.update()  # type: ignore


def get_yolo_bbox(scene, cam, obj):
    """
    Projects the 3D bounding box of an object into the 2D camera space.
    Returns the normalized coordinates in standard YOLO format:
    (center_x, center_y, width, height).
    """
    bbox_corners = [obj.matrix_world @ bpy.mathutils.Vector(  # type: ignore
        corner) for corner in obj.bound_box]
    co_2d = [bpy_extras.object_utils.world_to_camera_view(
        scene, cam, c) for c in bbox_corners]

    x_coords = [c.x for c in co_2d]
    # Invert Y for YOLO image coordinates
    y_coords = [1.0 - c.y for c in co_2d]

    min_x, max_x = max(0.0, min(x_coords)), min(1.0, max(x_coords))
    min_y, max_y = max(0.0, min(y_coords)), min(1.0, max(y_coords))

    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + (width / 2)
    center_y = min_y + (height / 2)

    return center_x, center_y, width, height


def generate_dataset():
    """
    Main execution loop.
    """
    setup_directories()

    scene = bpy.context.scene
    cam = scene.camera  # type: ignore
    assert (scene is not None)

    grid = scene.objects.get('Grid')
    ugv = scene.objects.get('UGV')
    human = scene.objects.get('Human')

    if not all([grid, ugv, human, cam]):
        print("ERROR: Missing Grid, UGV, Human, or Camera in the .blend file.")
        return

    print(f"Starting generation of {NUM_IMAGES} samples...")

    for i in range(NUM_IMAGES):
        # Randomize Environment
        create_maze(grid)
        position_actors(grid, ugv, human)

        bpy.context.view_layer.update()  # type: ignore

        # Render Output
        img_filename = f"maze_env_{i:04d}.png"
        scene.render.filepath = os.path.join(  # type: ignore
            IMAGE_DIR, img_filename)
        bpy.ops.render.render(write_still=True)

        # Calculate Bounding Boxes
        bbox_u = get_yolo_bbox(scene, cam, ugv)
        bbox_h = get_yolo_bbox(scene, cam, human)

        # Save YOLO Labels
        label_filename = f"maze_env_{i:04d}.txt"
        label_path = os.path.join(LABEL_DIR, label_filename)

        with open(label_path, 'w') as f:
            f.write(
                f"{CLASS_UGV} {bbox_u[0]:.6f} {bbox_u[1]:.6f} {bbox_u[2]:.6f} {bbox_u[3]:.6f}\n")
            f.write(
                f"{CLASS_HUMAN} {bbox_h[0]:.6f} {bbox_h[1]:.6f} {bbox_h[2]:.6f} {bbox_h[3]:.6f}\n")

        print(f"Rendered and labeled frame {i+1}/{NUM_IMAGES}")


if __name__ == "__main__":
    generate_dataset()
