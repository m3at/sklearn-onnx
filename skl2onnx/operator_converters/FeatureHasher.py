# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from ..common._registration import register_converter
from ..common._apply_operation import apply_concat, apply_identity


def convert_sklearn_feature_hasher(scope, operator, container):
    op = operator.raw_operator
              
    # https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/feature_extraction/_hashing.pyx
    # X.sum_duplicates()
    # if op.non_negative:
    #    np.abs(X.data, X.data)


register_converter('SklearnFeatureHasher', convert_sklearn_feature_hasher)
