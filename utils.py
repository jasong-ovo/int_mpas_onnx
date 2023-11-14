import onnxruntime as ort

class GC_processor_onnx(object):
    def __init__(self, model_parts) -> None:
        # Set the behavier of onnxruntime
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena=False
        options.enable_mem_pattern = False
        options.enable_mem_reuse = False
        # Increase the number for faster inference and more memory consumption
        options.intra_op_num_threads = 1
        # Set the behavier of cuda provider
        cuda_provider_options = {'arena_extend_strategy':'kSameAsRequested',}

        self.ort_sessions = []
        for sub_model in model_parts:
            ort_session = ort.InferenceSession(sub_model, sess_options=options, providers=['CPUExecutionProvider'])
            self.ort_sessions.append(ort_session)

    def predict(self, node_data, edge_data):
        for ort_session in self.ort_sessions:
            output = ort_session.run(None, {'input_node': node_data, 'input_edge': edge_data})
            node_data = output[0]
            edge_data = output[1]
        return node_data, edge_data