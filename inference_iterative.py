import os
import numpy as np
import onnx
import onnxruntime as ort
import argparse
from .utils import GC_processor_onnx



parser = argparse.ArgumentParser()
parser.add_argument('--inp_node_data_path', type=str, default=None, help='path to inp node data')
parser.add_argument('--inp_edge_data_path', type=str, default=None, help='path to inp edge data')
parser.add_argument('--output_data_dir',    type=str, default=None, help='path to save output data')
parser.add_argument('--infer_type',         type=str, default='12min1h', help='choose which type to use')

def min12h1_predict(inp_node_data, inp_edge_data, model_parts_12min, model_parts_1h):
    model_12min = GC_processor_onnx(model_parts_12min)
    model_1h = GC_processor_onnx(model_parts_1h)

    inp1h_node_data, inp1h_edge_data = inp_node_data, inp_edge_data
    node_outputs = []
    edge_outputs = []
    for i in range(30): ## 6h
        if (i+1) % 5 == 0:
            output_node_data, output_edge_data = model_1h.predict(inp1h_node_data, inp1h_edge_data)
            inp1h_node_data, inp1h_edge_data = output_node_data, output_edge_data
        else:
            output_node_data, output_edge_data = model_12min.predict(inp_node_data, inp_edge_data)
        inp_node_data, inp_edge_data = output_node_data, output_edge_data
        node_outputs.append(output_node_data)
        edge_outputs.append(output_edge_data)
    node_outputs = np.stack(node_outputs, axis=0)
    edge_outputs = np.stack(edge_outputs, axis=0)
    return node_outputs, edge_outputs



if __name__ == "__main__":
    args = parser.parse_args()
    node_data = np.load(args.inp_node_data_path) ## shape: (1, 256002, 221)
    edge_data = np.load(args.inp_edge_data_path) ## shape: (1, 768000, 55)

    model_parts_12min = ['onnx_models/12min/xxx.onnx', 'onnx_models/12min/xxx.onnx',
                         'onnx_models/12min/xxx.onnx', 'onnx_models/12min/xxx.onnx',
                         'onnx_models/12min/xxx.onnx'] ## change xxx.onnx to your model parts 
    model_parts_1h = ['onnx_models/1h/xxx.onnx', 'onnx_models/1h/xxx.onnx',
                      'onnx_models/1h/xxx.onnx', 'onnx_models/1h/xxx.onnx',
                      'onnx_models/1h/xxx.onnx']
    
    node_outputs, edge_outputs = min12h1_predict(inp_node_data=node_data, inp_edge_data=edge_data,
                                                  model_parts_12min=model_parts_12min, model_parts_1h=model_parts_1h)
    np.save(os.path.join(args.output_data_dir, 'node_data.npy'), node_outputs)
    np.save(os.path.join(args.output_data_dir, 'edge_data.npy'), edge_outputs)
    