# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy
from ..common._registration import register_shape_calculator
from ..common.data_types import FloatTensorType, Int64TensorType


def calculate_sklearn_feature_hasher_output_shapes(operator):
    '''
    This operator is used only to merge columns in a pipeline.
    Only id function is supported.
    '''
    op = operator.raw_operator
    N = operator.inputs[0].type.shape[0]
    C = op.n_features
            
    if op.dtype == numpy.float32:
        operator.outputs[0].type = FloatTensorType([N, C])
    elif op.dtype == numpy.int64:
        operator.outputs[0].type = Int64TensorType([N, C])
    else:
        raise TypeError("Unsupported type '{}'".format(op.dtype))

register_shape_calculator('SklearnFeatureHasher', calculate_sklearn_feature_hasher_output_shapes)
