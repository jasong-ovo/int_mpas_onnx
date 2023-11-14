import os
import numpy as np
import onnx
import onnxruntime as ort
import argparse
from .utils import GC_processor_onnx



parser = argparse.ArgumentParser()
parser.add_argument('--inp_node_data_path', type=str, default=None, help='path to inp node data')
parser.add_argument('--inp_edge_data_path', type=str, default=None, help='path to inp edge data')
parser.add_argument('--output_data_dir', type=str, default='output_data_dir', help='path to save output data')


if __name__ == "__main__":
    args = parser.parse_args()
    model_parts_12min = ['onnx_models/12min/xxx.onnx', 'onnx_models/12min/xxx.onnx',
                         'onnx_models/12min/xxx.onnx', 'onnx_models/12min/xxx.onnx',
                         'onnx_models/12min/xxx.onnx'] ## change xxx.onnx to your model parts 
    model_12min = GC_processor_onnx(model_parts_12min)
    node_data = np.load(args.inp_node_data_path) ## shape: (1, 256002, 221)
    edge_data = np.load(args.inp_edge_data_path) ## shape: (1, 768000, 55)
    node_data, edge_data = model_12min.predict(node_data, edge_data)

    np.save(os.path.join(args.output_data_dir, 'node_data.npy'), node_data)
    np.save(os.path.join(args.output_data_dir, 'edge_data.npy'), edge_data)



