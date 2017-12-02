import tensorflow as tf
import builtins


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
