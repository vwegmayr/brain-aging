import os
import numpy as np
import tensorflow as tf
import builtins
import nibabel as nib
from modules.models.example_loader import PointExamples
from sklearn.externals import joblib


def np_placeholder(X):
    assert isinstance(X, np.ndarray)
    return tf.placeholder(
                shape=[None] + list(X.shape[1:]),
                dtype=X.dtype.name)


def convert_dwi_and_mask_to_pkl(
    dwi_file,
    mask_file,
    save_path):
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

    joblib.dump(features, save_path)


def convert_nii_and_trk_to_pkl(
        nii_file,
        trk_file,
        block_size,
        path,
        samples_percent=1.0,
        n_samples=None,
        min_fiber_length=0,
        last_incoming=1):
    """Save the samples to numpy binary format."""
    # The labels are the real vectors.
    label_type = "point"

    example_loader = PointExamples(
        nii_file=nii_file,
        trk_file=trk_file,
        block_size=block_size,
        example_percent=samples_percent,
        num_eval_examples=0,
        min_fiber_length=min_fiber_length,
        last_incoming=last_incoming)

    X = {
        'blocks': [],
        'incoming': [],
        'centers': [],
    }
    y = []

    if n_samples is None:
        n_samples = len(example_loader.train_labels)

    for idx, label in enumerate(example_loader.train_labels):
        if idx >= n_samples:
            break
        block = PointExamples.build_datablock(
            example_loader.brain_data,
            example_loader.block_size,
            label['center'],
            label['incoming'],
            label['outgoing'],
            label_type)
        X['blocks'].append(block['data_block'])
        X['incoming'].append(block['incoming'])
        X['centers'].append(block['center'])
        y.append(block['outgoing'])


    for key in X.keys():
        X[key] = np.array(X[key])
    y = np.array(y)

    joblib.dump(X, os.path.join(path, "X.pkl"))
    joblib.dump(y, os.path.join(path, "y.pkl"))

def parse_hooks(hooks, locals, outdir):
    training_hooks = []
    for hook in hooks:
        if hook["type"] == "SummarySaverHook":
            name = hook["params"]["name"]
            summary_op = getattr(tf.summary, hook["params"]["op"])
            summary_op = summary_op(name, locals[name])
            hook_class = getattr(tf.train, "SummarySaverHook")
            hook_instance = hook_class(
                summary_op=summary_op,
                output_dir=outdir,
                save_steps=hook["params"]["save_steps"])
        else:
            hook_class = getattr(tf.train, hook["type"])
            hook_instance = hook_class(**hook["params"])

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
