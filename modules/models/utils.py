import os
import numpy as np
import tensorflow as tf
import builtins
import nibabel as nib

from modules.models.example_loader import PointExamples, aff_to_rot
from sklearn.externals import joblib
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.ops.variable_scope import variable_scope as var_scope
from tensorflow.python.summary import summary


def zero_fraction(name, value):
    summary.scalar("ZeroFraction_{}".format(name), tf.nn.zero_fraction(value))
        

def parse_layers(inputs, layers, mode=ModeKeys.TRAIN, default_summaries=None):
    """Parse layer config to tensors

    Args:
        layers (list): List containing the layer config.

    Returns:
        tensor: Final output of layers applied to inputs.
    """
    for idx, layer in enumerate(layers):
        assert len(list(layer)) == 1
        key = list(layer)[0]
        params = layer[key]
        if "name" not in params:
            params["name"] = key + str(idx)
        name = str(params["name"])
        params.pop("name")
        with var_scope(name, values=(inputs,)) as scope:
            if key == "dropout" or key == "batchnorm":
                if mode in [ModeKeys.EVAL, ModeKeys.PREDICT]:
                    inputs = getattr(tf.layers, key)(
                        inputs, training=False, **params, name=scope)
                elif mode == ModeKeys.TRAIN:
                    inputs = getattr(tf.layers, key)(
                        inputs, training=True, **params, name=scope)
            else:
                inputs = getattr(tf.layers, key)(
                    inputs, **params, name=scope)

        if default_summaries is not None:
            for summary in default_summaries:
                summary["sum_op"](name, inputs)

    return inputs


def np_placeholder(X):
    assert isinstance(X, np.ndarray)
    return tf.placeholder(
                shape=[None] + list(X.shape[1:]),
                dtype=X.dtype.name)


def make_rand_data(save_path, mode="train_and_test"):
    """Create and save small random train and test files for tracking

    Args:
        save_path (str): Folder where to save data.
        mode (str): One of "train", "test", "train_and_test" (default).

    Returns:
        dict: Dictionary containing the paths to the saved files.

    """
    assert mode in ["train", "test", "train_and_test"]
    
    dwi_train_path = None
    fiber_path = None
    dwi_test_path = None
    mask_path = None

    affine = np.eye(4)

    if mode == "train" or mode == "train_and_test":
        dwi_train = np.random.rand(10, 10, 10, 15)
        dwi_train = nib.nifti2.Nifti2Image(dwi_train, affine=affine)

        pts0 = np.random.uniform(size=(50,3))
        pts1 = np.random.uniform(size=(50,3))
        streamlines = ([(pts0, None, None), (pts1, None, None)])
        hdr = nib.trackvis.empty_header()
        nib.trackvis.aff_to_hdr(affine, hdr)

        dwi_train_path = os.path.join(save_path, "dwi_train.nii.gz")
        fiber_path = os.path.join(save_path, "fibers.trk")
        
        nib.save(dwi_train, dwi_train_path)
        nib.trackvis.write(
            fiber_path,
            streamlines,
            hdr)

    if mode == "test" or mode == "train_and_test":
        dwi_test = np.random.rand(10, 10, 10, 15)
        mask = np.random.binomial(n=1, p=0.1, size=(10, 10, 10))

        dwi_test = nib.nifti2.Nifti2Image(dwi_test, affine=affine)
        mask = nib.nifti2.Nifti2Image(mask, affine=affine)

        dwi_test_path = os.path.join(save_path, "dwi_test.nii.gz")
        mask_path = os.path.join(save_path, "mask.nii.gz")

        nib.save(dwi_test, dwi_test_path)
        nib.save(mask, mask_path)

    return {
        "train": {
            "dwi_file": dwi_train_path,
            "trk_file": fiber_path
        },
        "test": {
            "dwi_file": dwi_test_path,
            "mask_file": mask_path
        }
    }


def make_rand_sets(save_path, mode="train_and_test"):
    """Create and save small random train and test set for tracking"""

    paths = make_rand_data(save_path, mode)    

    if mode == "test" or mode == "train_and_test":
        make_test_set(
            save_path=os.path.join(save_path, "test.pkl"),
            **paths["test"])

        for _, file in paths["test"].items():
            os.remove(file)

    if mode == "train" or mode == "train_and_test":
        make_train_set(
            save_path=os.path.join(save_path, "train.pkl"),
            n_samples=100,
            n_incoming=3,
            **paths["train"])

        for _, file in paths["train"].items():
            os.remove(file)

    return {
        "train": {
            "X": os.path.join(save_path, "train_X.pkl"),
            "y": os.path.join(save_path, "train_y.pkl")
        },
        "test": {
            "X": os.path.join(save_path, "test_X.pkl")
        }
    }

def make_test_set(
    dwi_file=None,
    mask_file=None,
    save_path=None):
    """Convert diffusion data and white matter mask to pickle.

    Args:
        dwi_file (str): Path to nifti file containing diffusion data.
        mask_file (str): Path to nifti file containing white matter mask.

    Returns:
        None: Saves pickle to save_path
    """

    dwi = nib.load(dwi_file)
    mask = nib.load(mask_file)

    header = nib.trackvis.empty_header()
    nib.trackvis.aff_to_hdr(dwi.affine, header, True, True)
    header["dim"] = dwi.header.structarr["dim"][1:4]

    features = {
        "dwi": dwi.get_data(),
        "mask": mask.get_data(),
        "header": header
    }

    joblib.dump(features, os.path.join(save_path, "test_X.pkl"))


def make_train_set(
        dwi_file=None,
        trk_file=None,
        save_path=None,
        block_size=3,
        samples_percent=1.0,
        n_samples=None,
        min_fiber_length=0,
        n_incoming=1):
    """Save training set as pickle"""

    if samples_percent < 1 and n_samples is not None:
        raise RuntimeError("n_samples must be None, if samples_percent < 1.")

    # The labels are the real vectors.
    label_type = "point"

    example_loader = PointExamples(
        nii_file=dwi_file,
        trk_file=trk_file,
        block_size=block_size,
        example_percent=samples_percent,
        num_eval_examples=0,
        min_fiber_length=min_fiber_length,
        last_incoming=n_incoming)

    X = {
        'blocks': [],
        'incoming': [],
        'centers': [],
    }
    y = []

    if n_samples is None:
        n_samples = len(example_loader.train_labels)

    nii_aff = example_loader.brain_file.affine
    trk_aff = nib.trackvis.aff_from_hdr(example_loader.fiber_header)

    assert np.allclose(nii_aff, trk_aff)

    for idx, label in enumerate(example_loader.train_labels):
        if idx >= n_samples:
            break
        block = PointExamples.build_datablock(
            example_loader.brain_data,
            example_loader.block_size,
            label['center'],
            label['incoming'],
            label['outgoing'],
            label_type,
            nii_aff)
        X['blocks'].append(block['data_block'])
        X['incoming'].append(block['incoming'])
        X['centers'].append(block['center'])
        y.append(block['outgoing'])

    for key in X.keys():
        X[key] = np.array(X[key])
    y = np.array(y)

    joblib.dump(X, os.path.join(save_path, "train_X.pkl"))
    joblib.dump(y, os.path.join(save_path, "train_y.pkl"))

def parse_hooks(hooks, locals, save_path):
    training_hooks = []
    for hook in hooks:
        if "SummarySaverHook" in hook:
            params = hook["SummarySaverHook"]

            tensor_name = params["tensor"]
            sum_op = params["sum_op"](tensor_name, locals[tensor_name])

            hook_class = getattr(tf.train, "SummarySaverHook")
            hook_instance = hook_class(
                summary_op=sum_op,
                output_dir=save_path,
                save_steps=params["save_steps"])
        else:
            assert len(list(hook)) == 1
            hook_name = list(hook)[0]
            hook_class = getattr(tf.train, hook_name)
            hook_instance = hook_class(**hook[hook_name])

        training_hooks.append(hook_instance)

    return training_hooks


def save_fibers(fiber_list, header, out_name="fibers.trk"):
    """Save fibers form a list.

    Args:
    fiber_list: The list of fibers (lists of lists of points) to be saved.
    header: Original header of the fibers.
    out_name: Name with which to save the '.trk' file.
    """
    streamline = []
    for fiber_idx, _ in enumerate(fiber_list):
        cur_fiber = np.asarray(fiber_list[fiber_idx])
        streamline.append([cur_fiber, None, None])
        # Save new tractography using the header of the predicted fibers
        nib.trackvis.write(out_name, streamline, points_space='voxel',
        hdr_mapping=header)


def print(*args, **kwargs):
    builtins.print(*args, flush=True, **kwargs)
