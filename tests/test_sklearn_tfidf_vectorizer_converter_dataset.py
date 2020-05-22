"""
Tests scikit-learn's tfidf converter using downloaded data.
"""
import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from test_utils import dump_data_and_model


class TestSklearnTfidfVectorizerDataSet(unittest.TestCase):

    def test_tfidf_20newsgroups(self):
        data = fetch_20newsgroups()
        X, y = np.array(data.data)[:100], np.array(data.target)[:100]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        model = TfidfVectorizer().fit(X_train)
        onnx_model = convert_sklearn(
            model, 'cv', [('input', StringTensorType(X_test.shape))])
        dump_data_and_model(
            X_test, model, onnx_model,
            basename="SklearnTfidfVectorizer20newsgroups",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')")

    def test_tfidf_20newsgroups_nolowercase(self):
        data = fetch_20newsgroups()
        X, y = np.array(data.data)[:100], np.array(data.target)[:100]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        model = TfidfVectorizer(lowercase=False).fit(X_train)
        onnx_model = convert_sklearn(
            model, 'cv', [('input', StringTensorType(X_test.shape))])
        dump_data_and_model(
            X_test, model, onnx_model,
            basename="SklearnTfidfVectorizer20newsgroupsNOLower",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')")

    def test_tfidf_20newsgroups_char(self):
        data = fetch_20newsgroups()
        X, y = np.array(data.data)[:100], np.array(data.target)[:100]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)

        model = TfidfVectorizer(analyzer='char').fit(X_train)
        onnx_model = convert_sklearn(
            model, 'cv', [('input', StringTensorType(X_test.shape))])
        dump_data_and_model(
            X_test, model, onnx_model,
            basename="SklearnTfidfVectorizer20newsgroupsChar",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.4.0')")


if __name__ == "__main__":
    unittest.main()
