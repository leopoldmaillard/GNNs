import numpy as np
from PIL import Image

"""
This method returns the adjacency matrix [num_pixels, num_pixels] and a node embedding [num_pixels, 1]
of a input image.
author : Guillaume Renton (Normandy University)
"""
def from_image_to_graph(path_image):
    image = Image.open(path_image)
    im = np.asarray(image)
    row, col = np.shape(im)
    adjacency = np.zeros((row*col, row*col))
    for i in range(row-1):
        for j in range(col):
            adjacency[i*row+j, i*row+j+1] = 1
            adjacency[i*row+j, (i+1)*row+j] = 1

            adjacency[i*row+j+1, i*row+j] = 1
            adjacency[(i+1)*row+j, i*row+j] = 1
    nodes = []
    for i in range(row):
        nodes = np.concatenate((nodes, im[i,:]))
    nodes = nodes/255

    return adjacency, nodes[:,np.newaxis]