# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy

from onnxconverter_common import apply_identity
from onnx import TensorProto

from ..common._registration import register_converter
from ..algebra.onnx_ops import (
    OnnxAdd, OnnxSub, OnnxPow, OnnxDiv, OnnxMul,
    OnnxCast, OnnxNot, OnnxLess, OnnxLog, OnnxNeg, OnnxImputer
)


def convert_powertransformer(scope, operator, container):
    """Converter for PowerTransformer"""
    op_in = operator.inputs[0]
    op_out = operator.outputs[0].full_name
    op = operator.raw_operator
    opv = container.target_opset
    lambdas = op.lambdas_

    # tensors of units and zeros
    ones_ = OnnxDiv(op_in, op_in, op_version=opv)
    zeros_ = OnnxSub(op_in, op_in, op_version=opv)

    # logical masks for input
    less_than_zero = OnnxLess(op_in, zeros_, op_version=opv)
    less_mask = OnnxCast(less_than_zero,
                         to=getattr(TensorProto, 'FLOAT'),
                         op_version=opv)

    greater_than_zero = OnnxNot(less_than_zero, op_version=opv)
    greater_mask = OnnxCast(greater_than_zero,
                            to=getattr(TensorProto, 'FLOAT'),
                            op_version=opv)

    # logical masks for lambdas
    lambda_zero_mask = numpy.float32(lambdas == 0)
    lambda_nonzero_mask = numpy.float32(lambdas != 0)
    lambda_two_mask = numpy.float32(lambdas == 2)
    lambda_nontwo_mask = numpy.float32(lambdas != 2)

    if 'yeo-johnson' in op.method:
        y0 = OnnxAdd(op_in, ones_, op_version=opv)  # For positive input
        y1 = OnnxSub(ones_, op_in, op_version=opv)  # For negative input

        # positive input, lambda != 0
        y_gr0_l_ne0 = OnnxPow(y0, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxSub(y_gr0_l_ne0, ones_, op_version=opv)
        y_gr0_l_ne0 = OnnxDiv(y_gr0_l_ne0, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxImputer(y_gr0_l_ne0,
                                  imputed_value_floats=[0.0],
                                  replaced_value_float=numpy.inf,
                                  op_version=opv)
        y_gr0_l_ne0 = OnnxMul(y_gr0_l_ne0, lambda_nonzero_mask,
                              op_version=opv)

        # positive input, lambda == 0
        y_gr0_l_eq0 = OnnxLog(y0, op_version=opv)
        y_gr0_l_eq0 = OnnxMul(y_gr0_l_eq0, lambda_zero_mask,
                              op_version=opv)

        # positive input, an arbitrary lambda
        y_gr0 = OnnxAdd(y_gr0_l_ne0, y_gr0_l_eq0, op_version=opv)
        y_gr0 = OnnxImputer(y_gr0, imputed_value_floats=[0.0],
                            replaced_value_float=numpy.NAN,
                            op_version=opv)
        y_gr0 = OnnxMul(y_gr0, greater_mask, op_version=opv)

        # negative input, lambda != 2
        y_le0_l_ne2 = OnnxPow(y1, 2-lambdas, op_version=opv)
        y_le0_l_ne2 = OnnxSub(ones_, y_le0_l_ne2, op_version=opv)
        y_le0_l_ne2 = OnnxDiv(y_le0_l_ne2, 2-lambdas, op_version=opv)
        y_le0_l_ne2 = OnnxImputer(y_le0_l_ne2,
                                  imputed_value_floats=[0.0],
                                  replaced_value_float=numpy.inf,
                                  op_version=opv)
        y_le0_l_ne2 = OnnxMul(y_le0_l_ne2, lambda_nontwo_mask, op_version=opv)

        # negative input, lambda == 2
        y_le0_l_eq2 = OnnxNeg(OnnxLog(y1, op_version=opv), op_version=opv)
        y_le0_l_eq2 = OnnxMul(y_le0_l_eq2, lambda_two_mask)

        # negative input, an arbitrary lambda
        y_le0 = OnnxAdd(y_le0_l_ne2, y_le0_l_eq2, op_version=opv)
        y_le0 = OnnxImputer(y_le0, imputed_value_floats=[0.0],
                            replaced_value_float=numpy.NAN,
                            op_version=opv)
        y_le0 = OnnxMul(y_le0, less_mask)

        # Arbitrary input and lambda
        y = OnnxAdd(y_gr0, y_le0, output_names='tmp', op_version=opv)

    elif 'box-cox' in op.method:
        # positive input, lambda != 0
        y_gr0_l_ne0 = OnnxPow(op_in, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxSub(y_gr0_l_ne0, ones_, op_version=opv)
        y_gr0_l_ne0 = OnnxDiv(y_gr0_l_ne0, lambdas, op_version=opv)
        y_gr0_l_ne0 = OnnxImputer(y_gr0_l_ne0,
                                  imputed_value_floats=[0.0],
                                  replaced_value_float=numpy.inf,
                                  op_version=opv)
        y_gr0_l_ne0 = OnnxMul(y_gr0_l_ne0, lambda_nonzero_mask,
                              op_version=opv)

        # positive input, lambda == 0
        y_gr0_l_eq0 = OnnxLog(op_in)
        y_gr0_l_eq0 = OnnxImputer(y_gr0_l_eq0,
                                  imputed_value_floats=[0.0],
                                  replaced_value_float=numpy.NAN,
                                  op_version=opv)
        y_gr0_l_eq0 = OnnxMul(y_gr0_l_eq0, lambda_zero_mask, op_version=opv)

        # positive input, arbitrary lambda
        y = OnnxAdd(y_gr0_l_ne0, y_gr0_l_eq0,
                    output_names='tmp',
                    op_version=opv)

        # negative input
        # PowerTransformer(method='box-cox').fit(negative_data)
        # raises ValueError.
        # Therefore we cannot use convert_sklearn() for that model
    else:
        raise NotImplementedError(
            'Method {} is not supported'.format(op.method))

    y.set_onnx_name_prefix('pref')
    y.add_to(scope, container)

    if op.standardize:
        name = scope.get_unique_operator_name('Scaler')
        attrs = dict(name=name,
                     offset=op._scaler.mean_,
                     scale=1.0 / op._scaler.scale_)
        container.add_node('Scaler', 'tmp', op_out,
                           op_domain='ai.onnx.ml', **attrs)
    else:
        apply_identity(scope, 'tmp', op_out, container)


register_converter('SklearnPowerTransformer', convert_powertransformer)
