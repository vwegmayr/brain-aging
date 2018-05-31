import nibabel as nib
import os
import abc
import json
from sklearn.externals import joblib

from modules.models.data_transform import DataTransformer


class MriFeatureDumper(DataTransformer):
    def __init__(self, streamer, out_dir):
        # Initialize streamer
        _class = streamer["class"]
        self.streamer = _class(**streamer["params"])
        self.out_dir = out_dir

    @abc.abstractmethod
    def compute_features(self, image):
        """
        Returns:
            - feature_names: a list of names
            - feature_values: a list of values
        """
        pass

    def get_output_file_name(self, file_id):
        return self.streamer.get_image_label(file_id)

    def transform(self, X, y=None):
        out_path = self.out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Stream image one by one
        batches = self.streamer.get_batches()
        for batch in batches:
            file_ids = [fid for group in batch for fid in group.file_ids]
            for fid in file_ids:
                path = self.streamer.get_file_path(fid)
                im = nib.load(path).get_data()

                feature_names, feature_values = self.compute_features(im)
                dic = {
                    f_name: feature_values[i]
                    for i, f_name in enumerate(feature_names)
                }

                file_name = self.get_output_file_name(fid)
                with open(os.path.join(out_path, file_name + ".json"), "w") \
                        as f:
                    json.dump(dic, f, indent=2)

        self.streamer = None


class MriIncrementalPCAFeatures(MriFeatureDumper):
    def __init__(self, model_path, *args, **kwargs):
        super(MriIncrementalPCAFeatures, self).__init__(
            *args,
            **kwargs
        )

        self.model = joblib.load(model_path)

    def compute_features(self, image):
        projection = self.model.pca.transform([image.ravel()])[0]
        n = len(projection)
        feature_names = [str(i) for i in range(n)]
        feature_values = [float(v) for v in projection]
        return feature_names, feature_values
