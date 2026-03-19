import bpy
import bmesh
import random
import math


def create_maze(grid_obj):
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


# Call the function
assert (bpy.context.scene is not None)
grid = bpy.context.scene.objects['Grid']
ugv = bpy.context.scene.objects['UGV']
human = bpy.context.scene.objects['Human']
create_maze(grid)
position_actors(grid, ugv, human)
