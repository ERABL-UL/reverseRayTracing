import numpy as np
import OSToolBox as ost


def generate_hollow_cube_points(size, density):
    # Define the vertices of the cube
    vertices = [
        np.array([0, 0]),
        np.array([size, 0]),
        np.array([size, size]),
        np.array([0, size])
    ]

    # Generate points only for the bottom square
    points = []
    for i in range(len(vertices)):
        start_vertex = vertices[i]
        end_vertex = vertices[(i + 1)% len(vertices)]
        num_points = int(np.linalg.norm(end_vertex - start_vertex) * density)
        segment_points = [start_vertex + t * (end_vertex - start_vertex) for t in np.linspace(0, 1, num_points)]
        points.extend(segment_points)

    points = np.array(points)
    points = np.c_[points, np.ones(points.shape[0]), np.zeros(points.shape[0])]
    z_0 = points[:, -2]
    cube_slice = points

    for i in np.arange(start=-1, stop=int(size)-1, step=0.1):
        bottom_square_z = z_0 + i
        cube_to_be = np.c_[cube_slice[:, :2], bottom_square_z, cube_slice[:, -1]]
        points = np.vstack((points, cube_to_be))

    bottom = []
    for x in np.linspace(0, size, int(size * density)):
        for y in np.linspace(0, size, int(size * density)):
            bottom.append([x, y, 0, 1])

    bottom = np.array(bottom)
    # points = np.vstack((points, bottom))

    top = np.c_[bottom[:, :2], bottom[:, -1]+size-1, np.ones(bottom.shape[0])*2]

    points = np.vstack((points, top))
    points[:, 0] += 8
    points[:, 1] -= 5

    return points


def generate_hollow_cylinder_points(radius, height, density):
    points = []
    # Generate points on the side surface
    for z in np.linspace(0, height, int(height * density)):
        for theta in np.linspace(0, 2 * np.pi, int(2 * np.pi * density)):
            x_inner = radius * np.cos(theta)
            y_inner = radius * np.sin(theta)

            points.append([x_inner, y_inner, z, 0])  # Inner surface

    # Generate points on the top and bottom faces
    for z in [0, height]:
        for theta in np.linspace(0, 2 * np.pi, int(2 * np.pi * density)):
            for r in np.arange(0, radius, 0.01):
                x = r * np.cos(theta)  # Adjusted for the inner radius
                y = r * np.sin(theta)  # Adjusted for the inner radius
                points.append([x, y, z, 1])

    return np.array(points)

# Hollow Cube: 3x3x3 meters with a 1-meter thickness
cube = generate_hollow_cube_points(10., 10.)  # Adjust density as needed

# Hollow Cylinder: 1m inner radius, 1.5m outer radius, 2m height
cylinder = generate_hollow_cylinder_points(2, 6, 20)  # Adjust density as needed

# Save the points to a file or process as needed
ost.write_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/DEMO/DEMO_cube.ply", cube, ["x", "y", "z", "lbl"])
ost.write_ply("/home/willalbert/Documents/GitHub/reverseRayTracing/OUT/DEMO/DEMO_cylinder.ply", cylinder, ["x", "y", "z", "lbl"])

