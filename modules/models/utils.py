import os
import numpy as np
import tensorflow as tf
import builtins
from modules.models.example_loader import PointExamples


def convert_nii_and_trk_to_npy(
        nii_file,
        trk_file,
        block_size,
        path):
    """Save the samples to numpy binary format."""
    # The labels are the real vectors.
    label_type = "point"

    example_loader = PointExamples(
        nii_file=nii_file,
        trk_file=trk_file,
        block_size=block_size,
        num_eval_examples=0)

    X = {
        'blocks': [],
        'incoming': [],
        'centers': [],
    }
    y = []

    for idx, label in enumerate(example_loader.train_labels):
        print("{:3.2f} % Loaded").format(idx / len(example_loader.train_labels * 100), end='\r')
        block = PointExamples.build_datablock(
            example_loader.brain_data,
            example_loader.block_size,
            label['center'],
            label['incoming'], label['outgoing'],
            label_type)
        X['blocks'].append(block['data_block'])
        X['incoming'].append(block['incoming'])
        X['centers'].append(block['center'])
        y.append(block['outgoing'])

    np.save(os.path.join(path, "X.npy"), X)
    np.save(os.path.join(path, "y.npy"), y)

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


def print(string):
    builtins.print(string, flush=True)
