import sys

from .unsupervised_features import PyRadiomicsSingleFileTransformer


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    # print("in_path is {}".format(in_path))
    trafo = PyRadiomicsSingleFileTransformer(in_path, out_path)
    trafo.transform(None)
