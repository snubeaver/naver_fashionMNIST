import numpy as np
import h5py

def concat_bias(kernel, bias):
    return np.concatenate((bias.flatten(), kernel.flatten()), axis=0)

def concat_layer_params(layer_name, file):
    kernel = file[ layer_name + '/kernel:0'].value
    bias =  file[ layer_name + '/bias:0'].value
    return concat_bias(kernel, bias)


def pca(dir, layer, weights, epochs):
    for i in range(1, epochs + 1):
        with h5py.File(f'{dir}/{i}.hdf5', 'r') as f:
            weight = concat_layer_params(layer, f)
            weights.append(weight)

    weights = np.stack(weights, axis=0)
    u, s, vh = np.linalg.svd(weights, full_matrices=False)
    coordinates = np.dot(weights, vh[0:10].T)
    return coordinates