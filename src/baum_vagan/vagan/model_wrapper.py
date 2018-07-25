import importlib
import pydoc
import numpy as np
import matplotlib.pyplot as plt

from src.baum_vagan.vagan.model_vagan import vagan


def import_string(s):
    mod_name = ".".join(s.split(".")[:-1])
    mod = importlib.import_module(mod_name)
    f = getattr(mod, s.split(".")[-1])
    return f


class Dict2Obj(object):
    def __init__(self, dic):
        for key in dic:
            v = dic[key]
            # Special case
            if v == 'None':
                v = None
            setattr(self, key, v)


class VAGanWrapper(object):
    """
    Wraps Baumgartner implementaion of vagan model. In particular,
    it transforms the configuration into a python object where parameters
    are accessible through fields ('.' access).
    """
    def __init__(self, **kwargs):
        """
        Arg:
            - kwargs: params dictionary as specified in yaml config 
        """
        # Transform to object
        exp_config = Dict2Obj(kwargs)

        # Import network functions
        exp_config.critic_net = import_string(exp_config.critic_net)
        exp_config.generator_net = import_string(exp_config.generator_net)

        # Import optimizer
        exp_config.optimizer_handle = pydoc.locate(
            exp_config.optimizer_handle
        )

        # Should be done in the end, import data loader and create
        data_loader = import_string(exp_config.data_loader)
        data = data_loader(exp_config)

        self.config = exp_config
        self.data = data
        self.vagan = vagan(exp_config=exp_config, data=data)

    def fit(self, X, y=None):
        self.vagan.train()

    def set_save_path(self, save_path):
        self.vagan.set_save_path(save_path)

    def show_morphed_images(self):
        # Run predictions in an endless loop
        exp_config = self.config
        sampler_AD = lambda bs: self.data.testAD.next_batch(bs)[0]
        while True:

            ad_in = sampler_AD(1)
            mask = self.vagan.predict_mask(ad_in)
            # print(logits)

            morphed = ad_in + mask
            if exp_config.use_tanh:
                morphed = np.tanh(morphed)

            plt.imshow(np.squeeze(morphed[0]), cmap='gray')
            plt.show()

    def transform(self, X=None, y=None):
        # Load model
        path = self.config.trained_model_dir
        self.vagan.load_weights(log_dir=path)

        self.show_morphed_images()
        