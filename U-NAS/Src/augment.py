import numpy as np
import nibabel as nib
from nilearn.image import new_img_like, resample_to_img
import random
import itertools
import pdb

def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)


def flip_image(image, axis):
    try:
        new_data = np.copy(image.get_data())
        for axis_index in axis:
            new_data = np.flip(new_data, axis=axis_index)
    except TypeError:
        new_data = np.flip(image.get_data(), axis=axis)
    return new_img_like(image, data=new_data)


def random_flip_dimensions(n_dimensions):
    axis = list()
    for dim in range(n_dimensions):
        if random_boolean():
            axis.append(dim)
    return axis


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_boolean():
    return np.random.choice([True, False])


def distort_image(image, flip_axis=None, scale_factor=None):
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    return image


def do_augment(data, truth, affine, scale_deviation=None, flip=True):
    '''
    data.shape = (4,_,_,_)
    truth.shape = (1,_,_,_)
    '''
    n_dim = len(truth.shape) - 1
    scale_factor = random_scale_factor(n_dim, std=scale_deviation) if scale_deviation is not None else None
    flip_axis = random_flip_dimensions(n_dim) if flip else None
    data = augment_data(data, affine, scale_factor, flip_axis)
    truth = augment_data(truth, affine, scale_factor, flip_axis)
    return data, truth

def augment_data(data, affine, scale_factor, flip_axis):
    data_list = []
    for data_index in range(data.shape[0]):
        image = get_image(data[data_index], affine)
        data_list.append(resample_to_img(distort_image(image, flip_axis=flip_axis,
                                                       scale_factor=scale_factor), image,
                                         interpolation="nearest").get_data())
#                                          interpolation="continuous").get_data())
    return np.asarray(data_list)

def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)


# -------------------------------------- permutation: -----------------------------------

def generate_permutation_keys():
    """
    This function returns a set of "keys" that represent the 48 unique rotations &
    reflections of a 3D matrix.

    Each item of the set is a tuple:
    ((rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.

    48 unique rotations & reflections:
    https://en.wikipedia.org/wiki/Octahedral_symmetry#The_isometries_of_the_cube
    """
    return set(itertools.product(
        itertools.combinations_with_replacement(range(2), 2), range(2), range(2), range(2), range(2)))


def random_permutation_key():
    """
    Generates and randomly selects a permutation key. See the documentation for the
    "generate_permutation_keys" function.
    """
    return random.choice(list(generate_permutation_keys()))


def permute_data(data, key):
    """
    Permutes the given data according to the specification of the given key. Input data
    must be of shape (n_modalities, x, y, z).

    Input key is a tuple: (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose)

    As an example, ((0, 1), 0, 1, 0, 1) represents a permutation in which the data is
    rotated 90 degrees around the z-axis, then reversed on the y-axis, and then
    transposed.
    """
    data = np.copy(data)
    (rotate_y, rotate_z), flip_x, flip_y, flip_z, transpose = key

    if rotate_y != 0:
        data = np.rot90(data, rotate_y, axes=(1, 3))
    if rotate_z != 0:
        data = np.rot90(data, rotate_z, axes=(2, 3))
    if flip_x:
        data = data[:, ::-1]
    if flip_y:
        data = data[:, :, ::-1]
    if flip_z:
        data = data[:, :, :, ::-1]
    if transpose:
        for i in range(data.shape[0]):
            data[i] = data[i].T
    return data


def random_permutation_x_y(x_data, y_data):
    """
    Performs random permutation on the data.
    :param x_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :param y_data: numpy array containing the data. Data must be of shape (n_modalities, x, y, z).
    :return: the permuted data
    """
    key = random_permutation_key()
    return permute_data(x_data, key), permute_data(y_data, key)