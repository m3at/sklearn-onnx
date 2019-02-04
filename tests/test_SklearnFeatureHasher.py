"""
Tests scikit-learn's binarizer converter.
"""

import unittest
import numpy
from collections import defaultdict
from sklearn.feature_extraction import FeatureHasher
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


def tokens(doc):
    return doc.split()


def token_freqs(doc):
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq


class TestSklearnFeatureHasher(unittest.TestCase):

    def test_feature_hasher(self):
        raw_data = ["one two", "three four", "one three", "two four"]
        hasher = FeatureHasher(n_features=2**3, input_type='string', dtype=numpy.float32)
        X = hasher.transform(raw_data)
        model = convert_sklearn(hasher, initial_types=[('input', StringTensorType([1, 1]))])
        
        dump_data_and_model(numpy.array([[1, 1]], dtype=numpy.float32),
                            model, model_onnx, basename="SklearnBinarizer-SkipDim1")


if __name__ == "__main__":
    unittest.main()
