"""
A collection of utility and helper functions to avoid boilerplate code
"""

import argparse
import itertools
import os
import re

import nibabel as nib
import numpy as np
import scipy as sp
import sklearn.metrics as metrics
import tensorflow as tf
import tensorflow.contrib.slim as slim
import yaml
import importlib

from . import config_wrapper
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_X_y
from ml_project.data import DataLoader

class Connector(BaseEstimator, TransformerMixin):
    """docstring for Connector"""
    def __init__(self):
        super(Connector, self).__init__()

    def fit(self, X, y):
        check_X_y(X, y, force_all_finite=True,
                        allow_nd=True,
                        multi_output=True)
        return self

    def transform(self, X, y=None):
        return DataLoader(X, y)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
        

def logits2labels(logits):
    logits = np.squeeze(logits)
    labels = np.argmax(logits, axis=-1)
    return labels

def logits2proba(logits):
    logits = np.squeeze(logits)
    num_classes = logits.shape[-1]
    sig = sp.special.expit(logits)
    Z = np.sum(sig, axis=-1, keepdims=True)
    proba = sig / np.tile(Z, (1, num_classes))
    return proba

def get_object(module_string, class_string):
    module = importlib.import_module(module_string)
    object_ = getattr(module, class_string)
    return object_

def parse_arguments(parser=None):
    """
    Provides a function for an unified parameter interface that is compatible with Sumatra Sumatra_.
    
    The function expects the Sumatra parameter file Parameter_ a first unnamed argument. If the 
    Sumatra experiment label (with the fixed key :code:`sumatra_label`) is passed in the
    parameter file, the folder :code:`out/sumatra_label` is created.
    
    Additionally it adds a default program argument :code:`--out` which is a mandatory argument
    which specifies the output directory where all produced data is stored.
    
    .. _Sumatra: https://pythonhosted.org/Sumatra/
    .. _Parameter: https://pythonhosted.org/Sumatra/parameter_files.html 
    
    Examples:
        
        In an experiment, add this to the beginning of the file:
        
        .. code-block:: python
        
            # Experiment arguments
            args, parameter = utils.parse_arguments()
            
        The Sumatra experiment needs then to be called as follows:
        
        .. code-block:: shell
        
            smt configure -m path/to/main.py
            smt run path/to/parameter_file.yaml --out path/to/out_directory
            
    Args:
        parser (argparse object): Optional, overrides the provided argparser 
        
    Raises:
        RuntimeError: If the parameter file is missing

    Returns:
        (dict, dict, dict): A tuple (args, parameter) where args are the program arguments and
        parameter is the Sumatra parameter file. The last tuple entry is the raw content of
        the Sumatra parameter file, without any classes names replaced with instances.
    """
    # Default experiment arguments
    if parser is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-o', '--out', help='Output directory', required=True)

    args, parameter_file = parser.parse_known_args()
    if len(parameter_file) > 0:
        parameter_file = parameter_file[0]
    else:
        raise RuntimeError("Parameter file missing!")

    # Experiment arguments
    args = vars(args)
    param = config_wrapper.Config()
    param.parse_config_file(parameter_file)
    parameter = param.config

    # Also return the raw parameter file (only strings, no substitutions)
    with open(os.path.expanduser(parameter_file)) as config_file:
        raw_parameter_file = yaml.load(config_file)

    # Create Sumatra subdirectory
    if 'sumatra_label' in parameter:
        args['out'] = os.path.join(args['out'], parameter['sumatra_label'])
        if not os.path.exists(args['out']):
            os.mkdir(args['out'])

    # Append trailing slash, if not present
    if 'out' in args:
        args['out'] = append_slash(args['out'])

    # Args = Command line args, parameter = Sumatra parameter file
    return args, parameter, raw_parameter_file


def dump_yaml(input, prepend=">>  "):
    """
    Pretty format a YAML / dict and prepend the string :code:`prepend` to each line / key.
    
    Args:
        input (dict): Dictionary / YAML to be pretty printed 
        prepend (string): This string is prepended to each line of the input 

    Returns:
        Pretty formatted YAML / dictionary
    """
    output = yaml.dump(input, default_flow_style=False, )
    output = output.replace("\n", "\n" + prepend)
    output = prepend + output
    return output[0:(len(output) - len(prepend) - 1)]


def nrows(X):
    """
    Returns the number of rows of :code:`X`. This is equivalent to :code:`X.shape[0]`, 
    but performs additional type checks.

    Args:
        X (np.ndarray): Input data

    Returns:
        int: The number of rows, i.e. the dimension of the first axis.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('Argument `X` is not of type `np.ndarray`.')

    return X.shape[0]


def ncols(X):
    """
    Returns the number of columns of :code:`X`. This is equivalent to :code:`X.shape[1]`, 
    but performs additional type checks.

    Args:
        X (np.ndarray): Input data

    Returns:
        int: The number of columns, i.e. the dimension of the second axis.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError('Argument `X` is not of type `np.ndarray`.')

    if len(X.shape) <= 1:
        raise ValueError('Argument `X` has dimension:' + str(len(X.shape)) + ', required: 1')

    return X.shape[1]


def append_slash(path):
    """
    Appends a slash to the input :code:`path`, only of the last character is not already
    a slash. This is helpful correctly unify folder paths etc.
    
    >>> append_slash("some/path")
    some/path/
    >>> append_slash("some/path/")
    some/path/
    
    Args:
        path (string): Path to append 

    Returns:
        string: Correctly unify folder path
    """
    if not path[-1] == "/":
        path += "/"

    return path


def labels_to_indices(labels, columns):
    """
    Maps Pandas column labels to numpy column indices
    
    >>> labels_to_indices(['label_1', 'label_2', 'label_3'], ['label_1', 'label_3'])
    [0, 2]
    >>> labels_to_indices(['label_1', 'label_2', 'label_3'], ['label_1'])
    [0]
    >>> labels_to_indices(['label_1', 'label_2', 'label_3'], [])
    []
    
    Args:
        labels (list of strings): List of all column labels
        columns (list of strings): List of columns labels which should be mapped to numbers

    Returns:
        list: A list of integers containing the numpy column indices
    """
    x = list(map(lambda col: [labels.index(x) for x in labels if col in x], columns))
    return [item for sublist in x for item in sublist]


def tf_variable(name, scope, shape=None,
                initializer=tf.random_normal_initializer(), dtype=tf.float32,
                collections=None, **kwargs):
    """
    Creates a Tensorflow variables of shape :code:`shape` in the scope :code:`scope`. If the
    variable already exists in the scope, then a reference to it is returned.
    
    
    Examples:
        This avoids the common Tensorflow boilerplate code 
    
        .. code-block:: python
        
            # Define Tensorflow variable x in scope scope
            with tf.variable_scope("scope"):
                x = tf.get_variable("name", [10, 5], initializer=tf.random_normal_initializer(), 
                    dtype=tf.float32)
                
            # Some intermediate code
            ...
            
            # Read / use x
            with tf.variable_scope("scope", reuse=True):
                return tf.get_variable("name")
                
        This function minimizes the code to
            
        .. code-block:: python
        
            # Define Tensorflow variable x in scope scope
            x = tf_variable("name", "scope", [10, 5])
                
            # Some intermediate code
            ...
            
            # Read / use x
            x = tf_variable("name", "scope")
            
    Args:
        collections (list of strings): List of collection names to which the variable should
            be added. Default is none.
        name (string): Tensorflow variable name 
        scope (string): Tensorflow scope name 
        shape (list): Shape
        initializer (calable): Tensorflow initializer 
        dtype: Tensorflow data type 
        kwargs: Key word args which are further passed to tensorflow

    Returns:
        (tf_variable): A reference to a newly initialized variable called :code:`name` in the
        scope :code:`scope` if there is no variable in the scope yet, or a reference to an already
        existing variable :code:`name` in the scope :code:`scope`.
    """
    try:
        with tf.variable_scope(scope):
            return slim.variable(name=name, shape=shape, initializer=initializer, dtype=dtype,
                                 collections=collections, **kwargs)
    except ValueError:
        with tf.variable_scope(scope, reuse=True):
            return slim.variable(name=name)


def swap(df):
    """
    Swap the cluster labels such that a cluster label permutation is considered (Note: The heart
    data set has only two clusters)

    Args:
        df (pd.DataFrame): Assignment matrix holding the assignments for the 
            cluster as columns and the samples as rows

    Returns:
        Permuted cluster labels
    """
    df[df == 0] = -1
    df[df == 1] = 0
    df[df == -1] = 1
    return df


def compute_scores(y_pred, y_true):
    """
    Computes different scores such as purity, F1, accuary and NMI

    Args:
        y_pred (np.ndarray): Matrix holding the predicted assignments for the 
            cluster as columns and the samples as rows 
        y_true (np.ndarray): Column vector holding the ground truth
        
    Returns:
        A dictionary which contains the computed scores
    """
    scores = {}

    # Compute the confusion matrix needed for the purity score
    N = y_pred.shape[0]
    C = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Scores
    scores['purity'] = "%0.2f" % float(np.sum(np.max(C, axis=1)) / N)
    scores['f1'] = "%0.2f" % metrics.f1_score(y_true=y_true, y_pred=y_pred)
    scores['accuracy'] = "%0.2f" % (metrics.accuracy_score(y_true=y_true, y_pred=y_pred)*100.0)
    scores['nmi'] = "%0.2f" % metrics.normalized_mutual_info_score(
        labels_true=y_true, labels_pred=y_pred)

    return scores


def make_mri_index(base, pattern="_t1.nii.gz"):
    """
    Search for files containing the substring :code:`pattern` ind the file name. In the default
    settings it will search for T1 files. Possible patterns might be :code:`chunking` to detect
    corrupted MRI files or :code:`data.nii.gz` to search for DTI data etc.

    Args:
        base (string): Folder which gets recursively traversed
        pattern (string): String to search for in file names
    Returns:
        CSV formatted output
    """
    exclude = ['avg_brain.nii.gz', 'std_brain.nii.gz', 'union_brain_mask.nii.gz']
    out = "id,patient_id,patient_id_scan,folder,file,path,path_general,dim_1,dim_2,dim_3,dim_4\n"
    for root, dirs, files in os.walk(base):
        path = root.split(os.sep)
        patient = "/".join(path).split(base)[1]

        for file in files:
            if pattern in file:
                if file not in exclude:
                    full_path = "/".join(path) + "/" + file
                    folder = "/".join(path)
                    id = patient + "_" + file
                    patient_id = patient.split("/")[0]
                    patient_id = re.sub("[^0-9]", "", patient_id)
                    if patient_id == "":
                        patient_id = file.split("_")[0]

                    # Get file dimensions
                    if ".nii.gz" in full_path:
                        # Nifti files
                        header = nib.load(full_path).header
                        dims = ["%d" % i for i in header['dim'][1:5]]
                    else:
                        # npy files
                        dims = [str(i) for i in np.load(full_path).shape]
                        dims += ["1"] * (4 - len(dims))

                    out += id + "," + patient_id + "," + patient + "," + folder + "," + \
                           file + "," + full_path + "," + \
                           full_path.split(".nii.gz")[0] + "," + ",".join(dims) + "\n"

    return out


def get_brain_window_indices(stop, stride, start=(0, 0, 0)):
    """
    Generate an index mask to loop over the entire brain volume form start to stop with
    stride :code:`stride`.

    Args:
        start (tuple): Start coordinates. Default (0, 0, 0)
        stop (tuple): Volume / Brain of shape (d1, d2, d3) 
        stride (tuple): Stride of shape (w1, w2, w3)

    Returns:
        A list of tuples containing the indices of a sliding window with the stide equal to the
        window size
    """
    index = lambda i: list(range(start[i], stop[i], stride[i]))
    for i in itertools.product(index(0), index(1), index(2)):
        yield tuple(map(np.minimum, stop, i))


def is_perfect_root(value, exponent):
    """
    Checks if an integer has a perfect nth root
    
    Taken from https://stackoverflow.com/questions/27607711/
    check-if-an-integer-has-perfect-nth-root-python`
    
    Args:
        value (int): Interger to check
        exponent (int): Root

    Returns:
        True, if value has a perfect nth-exponent
    """
    root = value ** ( 1.0 / exponent)
    root = int(round(root))
    return root ** exponent == value


def make_list(l):
    """
    Ensures that the output is a list
    
    Args:
        l: Make a list of l, i.e. [l] 

    Returns:
        Input wrapped into a list
    """
    if not isinstance(l, list):
        return [l]

    return list(itertools.chain(l))


def get_n_parameter():
    """
    Calculates the number of paramters of the model.
    
    Found on https://stackoverflow.com/questions/38160940/how-to-count-
    total-number-of-trainable-parameters-in-a-tensorflow-model
    
    Returns:
        Number of parameters.
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    return total_parameters
