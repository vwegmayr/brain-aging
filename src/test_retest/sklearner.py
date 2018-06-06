from sklearn.metrics import accuracy_score
import importlib


class Sklearner(object):
    def __init__(self, config, estimator_config, streamer_config):
        self.config = config
        self.est = self.create_object(estimator_config)
        self.streamer = self.create_object(streamer_config)

    def create_object(self, config):
        _class = config["class"]
        _params = config["params"]
        return _class(**_params)

    def fit(self, X, y):
        # Read true data based on streamer
        X, y = self.streamer.get_data_matrices(train=True)
        self.est.fit(X, y)

        # Compute score on test set
        X_test, y_test = self.streamer.get_data_matrices(train=False)
        pred = self.est.predict(X_test)

        sc_funcs = self.get_score_funcs()
        for sc_func in sc_funcs:
            sc = sc_func(y_test, pred)
            print("{}: {}".format(sc_func.__name__, sc))

        self.streamer = None

    def get_score_funcs(self):
        func_names = self.config["score_funcs"]
        return [self.str_to_func(s) for s in func_names]

    def str_to_func(self, s):
        mod_name = ".".join(s.split(".")[:-1])
        mod = importlib.import_module(mod_name)
        f = getattr(mod, s.split(".")[-1])
        return f
