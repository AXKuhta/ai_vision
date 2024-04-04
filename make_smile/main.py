import numpy as np
from PIL import Image
import matplotlib.pyplot as pp

def quad_as_rect(quad):
    if quad[0] != quad[2]: return False
    if quad[1] != quad[7]: return False
    if quad[4] != quad[6]: return False
    if quad[3] != quad[5]: return False
    return True

def quad_to_rect(quad):
    assert(len(quad) == 8)
    assert(quad_as_rect(quad))
    return (quad[0], quad[1], quad[4], quad[3])

def rect_to_quad(rect):
    assert(len(rect) == 4)
    return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])

def shape_to_rect(shape):
    assert(len(shape) == 2)
    return (0, 0, shape[0], shape[1])

def griddify(rect, w_div, h_div):
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    x_step = w / float(w_div)
    y_step = h / float(h_div)
    y = rect[1]
    grid_vertex_matrix = []
    for _ in range(h_div + 1):
        grid_vertex_matrix.append([])
        x = rect[0]
        for _ in range(w_div + 1):
            grid_vertex_matrix[-1].append([int(x), int(y)])
            x += x_step
        y += y_step
    grid = np.array(grid_vertex_matrix)
    return grid

def distort_grid(org_grid, max_shift):
    new_grid = np.copy(org_grid)
    x_min = np.min(new_grid[:, :, 0])
    y_min = np.min(new_grid[:, :, 1])
    x_max = np.max(new_grid[:, :, 0])
    y_max = np.max(new_grid[:, :, 1])
    new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
    new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
    new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
    new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
    return new_grid

def grid_to_mesh(src_grid, dst_grid):
    assert(src_grid.shape == dst_grid.shape)
    mesh = []
    for i in range(src_grid.shape[0] - 1):
        for j in range(src_grid.shape[1] - 1):
            src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                        src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                        src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                        src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
            dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                        dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                        dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                        dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
            dst_rect = quad_to_rect(dst_quad)
            mesh.append([dst_rect, src_quad])
    return mesh

import matplotlib.pyplot as plt

img = Image.open("small.jpeg")

dst_grid = griddify(shape_to_rect(img.size), 34, 34)
src_grid = dst_grid.copy() # distort_grid(dst_grid, 3)

# Y X
# V1
#src_grid[14][8] += np.array([5, 5])
#src_grid[14][9] += np.array([0, -5])
#src_grid[14][10] += np.array([-5, 5])

# V2
src_grid[14][8] += np.array([-7, 15])
src_grid[14][9] += np.array([0, 3])
src_grid[14][10] += np.array([-5, 5])
src_grid[13][11] += np.array([-5, 15])
src_grid[14][11] += np.array([-5, 15])

src_grid[11][22] += np.array([0, 10])
src_grid[12][22] += np.array([0, 10])
src_grid[11][23] += np.array([0, 5])
src_grid[12][23] += np.array([0, 5])

src_grid[11][24] += np.array([0, 3])
src_grid[12][24] += np.array([0, 3])

src_grid[11][25] += np.array([-5, 5])
src_grid[12][25] += np.array([-5, 5])
src_grid[11][26] += np.array([-5, 10])
src_grid[12][26] += np.array([-5, 10])


mesh = grid_to_mesh(src_grid, dst_grid)

distorted = img.transform(img.size, Image.MESH, mesh)


fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)

axes[0].imshow(img)
axes[0].scatter(dst_grid[:, :, 0], dst_grid[:, :, 1])

for r, a in enumerate(dst_grid):
	for c, b in enumerate(a):
		axes[0].text(*b, f"{r} {c}")

axes[1].imshow(distorted)
# axes[1].scatter(src_grid[:, :, 0], src_grid[:, :, 1])

plt.show()
