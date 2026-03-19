import bpy
import bmesh
import random


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

            # open all edges between current cell and chosen neighbor
            for edge in current_cell.edges:
                if edge in chosen_neighbor.edges:
                    setattr(closed.data[edge.index], 'value', False)

    # update the mesh to show the maze
    mesh.update()  # type: ignore


# Call the function
assert (bpy.context.scene is not None)
create_maze(bpy.context.scene.objects['Grid'])
