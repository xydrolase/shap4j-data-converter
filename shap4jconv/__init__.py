#!/usr/bin/env python

import pickle
import ctypes
import os.path

import numpy as np
from shap.explainers._tree import TreeEnsemble


INT_ARRAYS = ('children_left', 'children_right', 'children_default', 'features')
DOUBLE_ARRAYS = ('thresholds', 'values', 'node_sample_weight')


class Shap4jHeader(ctypes.Structure):
    """
    The header of the shap4j data file (without the actual data arrays), which is used by
    the shap4j library.
    """
    _fields_ = [
        ('magic', ctypes.c_byte * 4),
        ('version', ctypes.c_int32),
        ('num_trees', ctypes.c_int32),
        ('max_depth', ctypes.c_int32),
        ('max_nodes', ctypes.c_int32),
        ('num_outputs', ctypes.c_int32),
        ('offset_int_arrays', ctypes.c_int32),
        ('offset_double_arrays', ctypes.c_int32),
        ('base_offset', ctypes.c_double)
    ]


class Shap4jDataConverter:
    MAGIC = ctypes.c_byte * 4
    MAGIC_BYTES = MAGIC(ord('S'), ord('H'), ord('A'), ord('P'))
    VERSION = 1

    def __init__(self):
        pass

    @staticmethod
    def validate_data_shapes(model, array_name, length):
        assert hasattr(model, array_name), f"Tree ensemble model does not have attribute: {array_name}"
        num_elements = np.product(getattr(model, array_name).shape)
        assert num_elements == length, \
            f"Data array {array_name} is expected to have {length} elements, found {num_elements} instead."

    @staticmethod
    def align_by(x, by):
        if (x / by) == (x // by):
            return x

        return (x // by + 1) * by

    def convert(self, model_file, output_file=None, input_type="pickle", overwrite=False):
        """
        Convert the input model_file of a given input_type to the specified output_file.
        If output_file is omitted, the output filename is generated based on the input
        model_file name (e.g. foo.pkl becomes foo.shap4j).
        """
        
        with open(model_file, "rb") as f:
            if input_type == "pickle":
                model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")

        ensemble_model = TreeEnsemble(model)
        # max number of nodes in any given tree of the ensemble model.
        max_nodes = np.max([len(t.values) for t in ensemble_model.trees])

        # FIXME: heuristic; verify this on all kinds of inputs
        num_trees = len(ensemble_model.trees)

        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'children_left', num_trees * max_nodes)
        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'children_right', num_trees * max_nodes)
        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'children_default', num_trees * max_nodes)
        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'features', num_trees * max_nodes)

        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'thresholds', num_trees * max_nodes)
        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'values', num_trees * max_nodes * ensemble_model.num_outputs)
        Shap4jDataConverter.validate_data_shapes(ensemble_model, 'node_sample_weight', num_trees * max_nodes)

        assert ensemble_model.children_left.shape

        if output_file is None:
            base, ext = os.path.splitext(model_file)
            output_file = base + '.shap4j'

        if not overwrite and os.path.exists(output_file):
            raise IOError(f"Output file {output_file} already exists.")

        pad_header = b''
        pad_int_arrays = b''

        with open(output_file, 'wb+') as f:
            header = Shap4jHeader(
                magic=self.MAGIC_BYTES,
                version=self.VERSION,
                num_trees=num_trees,
                max_depth=ensemble_model.max_depth,
                max_nodes=max_nodes,
                base_offset=ensemble_model.base_offset,
                num_outputs=ensemble_model.num_outputs,
                offset_int_arrays=0,
                offset_double_arrays=0
            )
            header_size = len(bytes(header))

            offset = header_size

            # align the starting address of the integer arrays by multiples of 16
            if offset % 16 != 0:
                aligned_offset = Shap4jDataConverter.align_by(offset, 16)
                pad_header = b'\x00' * (aligned_offset - offset)
                header.offset_int_arrays = aligned_offset
                offset = aligned_offset
            else:
                header.offset_int_arrays = offset

            for int_array in INT_ARRAYS:
                ndarray = getattr(ensemble_model, int_array)
                offset += ndarray.size * ndarray.dtype.itemsize

            # similarly, align the starting address of the double arrays by 16
            if offset % 16 != 0:
                aligned_offset = Shap4jDataConverter.align_by(offset, 16)
                pad_int_arrays = b'\x00' * (aligned_offset - offset)
                header.offset_double_arrays = aligned_offset
            else:
                header.offset_double_arrays = offset

            f.write(bytes(header))
            f.write(pad_header)

            for int_array in INT_ARRAYS:
                f.write(getattr(ensemble_model, int_array).tobytes())

            f.write(pad_int_arrays)

            for double_array in DOUBLE_ARRAYS:
                f.write(getattr(ensemble_model, double_array).tobytes())


