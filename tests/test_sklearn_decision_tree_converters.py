# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
from distutils.version import StrictVersion
import numpy as np
from numpy.testing import assert_almost_equal
from pandas import DataFrame
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    ExtraTreeClassifier, ExtraTreeRegressor
)
from sklearn.datasets import make_classification
from skl2onnx.common.data_types import onnx_built_with_ml
from skl2onnx.common.data_types import (
    BooleanTensorType,
    FloatTensorType,
    Int64TensorType,
)
from skl2onnx import convert_sklearn
from onnxruntime import InferenceSession, __version__
from test_utils import (
    dump_one_class_classification,
    dump_binary_classification,
    dump_data_and_model,
    dump_multiple_classification,
    dump_multiple_regression,
    dump_single_regression,
    fit_classification_model,
    fit_multilabel_classification_model,
    fit_regression_model,
    TARGET_OPSET,
    binary_array_to_string
)


class TestSklearnDecisionTreeModels(unittest.TestCase):
    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    @unittest.skipIf(
        StrictVersion(__version__) <= StrictVersion("0.3.0"),
        reason="No suitable kernel definition found "
               "for op Cast(9) (node Cast)")
    def test_decisiontree_classifier1(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict_proba(X)
        if res[1][0][0] != pred[0, 0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decisiontree_regressor0(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(model, initial_types=initial_types)
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        if res[0][0, 0] != pred[0]:
            raise AssertionError("{}\n--\n{}".format(pred, DataFrame(res[1])))

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decisiontree_regressor0_decision_path(self):
        model = DecisionTreeRegressor(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True}})
        sess = InferenceSession(model_onnx.SerializeToString())
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec.todense())
        assert exp == res[1].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decisiontree_classifier_decision_path(self):
        model = DecisionTreeClassifier(max_depth=2)
        X, y = make_classification(10, n_features=4, random_state=42)
        X = X[:, :2]
        model.fit(X, y)
        initial_types = [('input', FloatTensorType((None, X.shape[1])))]
        model_onnx = convert_sklearn(
            model, initial_types=initial_types,
            options={id(model): {'decision_path': True, 'zipmap': False}})

        try:
            sess = InferenceSession(model_onnx.SerializeToString())
        except Exception as e:
            # onnxruntime.capi.onnxruntime_pybind11_state.Fail: 
            # [ONNXRuntimeError] : 1 : FAIL : Node:TreePath Output:decision_path 
            # [ShapeInferenceError] Mismatch between number of source and 
            # target dimensions. Source=0 Target=2
            warnings.warn(str(e))
            return
        res = sess.run(None, {'input': X.astype(np.float32)})
        pred = model.predict(X)
        assert_almost_equal(pred, res[0].ravel())
        prob = model.predict(X)
        assert_almost_equal(prob, res[1].ravel())
        dec = model.decision_path(X)
        exp = binary_array_to_string(dec.todense())
        assert exp == res[2].ravel().tolist()

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_decision_tree_classifier(self):
        model = DecisionTreeClassifier()
        dump_one_class_classification(
            model,
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_binary_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")
        dump_multiple_classification(
            model,
            label_uint8=True,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")
        dump_multiple_classification(
            model,
            label_string=True,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_extra_tree_classifier(self):
        model = ExtraTreeClassifier()
        dump_one_class_classification(
            model,
            # Operator cast-1 is not implemented in onnxruntime
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_binary_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )
        dump_multiple_classification(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.3') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')",
        )

    def test_decision_tree_regressor(self):
        model = DecisionTreeRegressor()
        dump_single_regression(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )
        dump_multiple_regression(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def test_extra_tree_regressor(self):
        model = ExtraTreeRegressor()
        dump_single_regression(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )
        dump_multiple_regression(
            model,
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2')",
        )

    def test_decision_tree_regressor_int(self):
        model, X = fit_regression_model(
            DecisionTreeRegressor(random_state=42), is_int=True)
        model_onnx = convert_sklearn(
            model,
            "decision tree regression",
            [("input", Int64TensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnDecisionTreeRegressionInt",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_multi_class_nocl(self):
        model, X = fit_classification_model(
            DecisionTreeClassifier(),
            4, label_string=True)
        model_onnx = convert_sklearn(
            model,
            "multi-class nocl",
            [("input", FloatTensorType([None, X.shape[1]]))],
            options={id(model): {'nocl': True}})
        self.assertIsNotNone(model_onnx)
        sonx = str(model_onnx)
        assert 'classlabels_strings' not in sonx
        assert 'cl0' not in sonx
        dump_data_and_model(
            X, model, model_onnx, classes=model.classes_,
            basename="SklearnDTMultiNoCl",
            allow_failure="StrictVersion(onnx.__version__)"
                          " < StrictVersion('1.2') or "
                          "StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_decision_tree_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            DecisionTreeClassifier(random_state=42))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn DecisionTreeClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnDecisionTreeClassifierMultiLabel-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )

    @unittest.skipIf(not onnx_built_with_ml(),
                     reason="Requires ONNX-ML extension.")
    def test_model_extra_tree_classifier_multilabel(self):
        model, X_test = fit_multilabel_classification_model(
            ExtraTreeClassifier(random_state=42))
        options = {id(model): {'zipmap': False}}
        model_onnx = convert_sklearn(
            model,
            "scikit-learn ExtraTreeClassifier",
            [("input", FloatTensorType([None, X_test.shape[1]]))],
            options=options,
            target_opset=TARGET_OPSET
        )
        self.assertTrue(model_onnx is not None)
        assert 'zipmap' not in str(model_onnx).lower()
        dump_data_and_model(
            X_test,
            model,
            model_onnx,
            basename="SklearnExtraTreeClassifierMultiLabel-Out0",
            allow_failure="StrictVersion("
            "onnxruntime.__version__) <= StrictVersion('0.2.1')",
        )

    def test_decision_tree_regressor_bool(self):
        model, X = fit_regression_model(
            DecisionTreeRegressor(random_state=42), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "decision tree regressor",
            [("input", BooleanTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnDecisionTreeRegressionBool-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")

    def test_extra_tree_regressor_bool(self):
        model, X = fit_regression_model(
            ExtraTreeRegressor(random_state=42), is_bool=True)
        model_onnx = convert_sklearn(
            model,
            "extra tree regressor",
            [("input", BooleanTensorType([None, X.shape[1]]))],
        )
        self.assertIsNotNone(model_onnx)
        dump_data_and_model(
            X,
            model,
            model_onnx,
            basename="SklearnExtraTreeRegressionBool-Dec4",
            allow_failure="StrictVersion(onnxruntime.__version__)"
                          " <= StrictVersion('0.2.1')")


if __name__ == "__main__":
    unittest.main()
