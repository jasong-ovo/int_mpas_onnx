import onnxruntime as ort
import xarray as xr
import numpy as np
import yaml


################################################## onnx model as parts ##################################################
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


################################################## data format convertor ##################################################
class data_format_convertor(object):
    def __init__(self, cfg_path) -> None:
        ## open yaml config ##
        #cfg_path = './data_configs/mpas_dataset.yaml'
        with open(cfg_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)

        kwargs = cfg_params.get('mpas_48km_raw', None)
        self.single_level_cell_vnames = kwargs.get('single_level_cell_vnames', None)
        self.multi_level_cell_vnames = kwargs.get('multi_level_cell_vnames', None)
        self.multi_level_P1_cell_vnames = kwargs.get('multi_level_P1_cell_vnames', None)
        self.multi_level_edge_vnames = kwargs.get('multi_level_edge_vnames', None)
        self.level_list = kwargs.get('level_list', None)
        self.P1_level_list = kwargs.get('P1_level_list', None)

        #### graph structure ####
        graph_structure_dir = './data_configs/mpas_graph_structure'
        self.indexToCellID = np.load(f'{graph_structure_dir}/indexToCellID.npy')
        self.indexToEdgeID = np.load(f'{graph_structure_dir}/indexToEdgeID.npy')

        ### normalization ###
        self.norm_type = kwargs.get('norm_type', 'Z-score')
        self.eps = 1e-16
        ## load mean and var ##
        statics_dir = './data_configs/mpas_statics'
        self.mean_all_node_vars = np.load(f'{statics_dir}/node_mean.npy') 
        self.var_all_node_vars = np.load(f'{statics_dir}/node_var.npy')
        self.mean_all_edge_vars = np.load(f'{statics_dir}/edge_mean.npy') 
        self.var_all_edge_vars = np.load(f'{statics_dir}/edge_var.npy')
        ####################################################################################################################
        self.node_var2id_statics, self.edge_var2id_statics = self._init_var2id_statics(f'{statics_dir}/cal_statics.yaml')
        self.node_id2var_statics = {v:k for k,v in self.node_var2id_statics.items()}
        self.edge_id2var_statics = {v:k for k,v in self.edge_var2id_statics.items()}
        node_idx, edge_idx = self._find_varIdx()
        self.node_idx, self.edge_idx = node_idx, edge_idx
        self.node_mean, self.node_std = self.mean_all_node_vars[node_idx], np.sqrt(self.var_all_node_vars[node_idx])
        self.edge_mean, self.edge_std = self.mean_all_edge_vars[edge_idx], np.sqrt(self.var_all_edge_vars[edge_idx])


    
    def nc_2_npy(self, nc_path):
        nc_data = xr.open_dataset(nc_path)
        node_data, edge_data = self._get_data(nc_data)
        normal_node_data, normal_edge_data = self.normalization(node_data, type=self.norm_type, mean=self.node_mean, std=self.node_std), self.normalization(edge_data, type=self.norm_type, mean=self.edge_mean, std=self.edge_std)
        return normal_node_data, normal_edge_data
    
    def _get_data(self, data):
        """
        choose variables from xarray data.
        ---------------------------------
        single_level_cell_data: (1, 256002)
        multi_level_cell_data: (1, 256002, 55)
        multi_level_P1_cell_data: (1, 256002, 56)
        multi_level_edge_data: (1, 768000, 55)
        """
        cell_data = []
        edge_data = []
        
        for vname in self.single_level_cell_vnames:
            if vname == "isltyp" or vname == "ter":
                cell_data.append(data[vname].values[np.newaxis, ..., np.newaxis])
            else:
                cell_data.append(data[vname].values[..., np.newaxis])
        for vname in self.multi_level_cell_vnames:
            cell_data.append(data[vname].values[..., self.level_list])
        for vname in self.multi_level_P1_cell_vnames:
            if vname == "zgrid":
                data_val = data[vname].values[..., self.P1_level_list]
                cell_data.append(data_val[np.newaxis, ...])
            else:
                cell_data.append(data[vname].values[..., self.P1_level_list])
        cell_data = np.concatenate(cell_data, axis=-1)
        cell_data = cell_data[:, self.indexToCellID, :]
        
        if self.multi_level_edge_vnames is not None:
            for vname in self.multi_level_edge_vnames:
                edge_data.append(data[vname].values[..., self.level_list])
            edge_data = np.concatenate(edge_data, axis=-1)
            edge_data = edge_data[:, self.indexToEdgeID, :]
        return cell_data, edge_data
    
    def normalization(self, data, type='Z-score', **kwargs):
        """
        normalization data.
        """
        if type == 'Z-score':
            mean = kwargs.get('mean', None)
            std = kwargs.get('std', None)
            return (data-mean)/(std + self.eps)
        elif type == 'orig':
            return data
        else:
            raise NotImplementedError(f'normalization type: {type} is not implemented.')
    
    def _init_var2id_statics(self, cfg_path):
        """
        init var2id dict in mean_all and var_all.
        """
        import yaml
        with open(cfg_path, 'r') as cfg_file:
            cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
        single_level_cell_vnames = cfg_params['mpas_48km_raw']['single_level_cell_vnames']
        multi_level_cell_vnames = cfg_params['mpas_48km_raw']['multi_level_cell_vnames']
        multi_level_P1_cell_vnames = cfg_params['mpas_48km_raw']['multi_level_P1_cell_vnames']
        multi_level_edge_vnames = cfg_params['mpas_48km_raw']['multi_level_edge_vnames']
        level_list = cfg_params['mpas_48km_raw']['level_list']
        P1_level_list = cfg_params['mpas_48km_raw']['P1_level_list']

        node_idx = 0
        node_var2id_dict = {}
        for vname in single_level_cell_vnames:
            node_var2id_dict.update({vname: node_idx})
            node_idx += 1
        for vname in multi_level_cell_vnames:
            for level in level_list:
                node_var2id_dict.update({f'{vname}-{level}': node_idx})
                node_idx += 1
        for vname in multi_level_P1_cell_vnames:
            for level in P1_level_list:
                node_var2id_dict.update({f'{vname}-{level}': node_idx})
                node_idx += 1
        
        edge_idx = 0 
        edge_var2id_dict = {}
        for vname in multi_level_edge_vnames:
            for level in level_list:
                edge_var2id_dict.update({f'{vname}-{level}': edge_idx})
                edge_idx += 1
        
        return node_var2id_dict, edge_var2id_dict
    

    def _find_varIdx(self):
        """
        find idx of var used in training from statics dict.
        """
        node_varIdxes = []
        for vname in self.single_level_cell_vnames:
            node_varIdxes.append(self.node_var2id_statics[vname])
        for vname in self.multi_level_cell_vnames:
            for level in self.level_list:
                node_varIdxes.append(self.node_var2id_statics[f'{vname}-{level}'])
        for vname in self.multi_level_P1_cell_vnames:
            for level in self.P1_level_list:
                node_varIdxes.append(self.node_var2id_statics[f'{vname}-{level}'])
        
        edge_varIdxes = []
        for vname in self.multi_level_edge_vnames:
            for level in self.level_list:
                edge_varIdxes.append(self.edge_var2id_statics[f'{vname}-{level}'])
        
        return node_varIdxes, edge_varIdxes

if __name__ == "__main__":
    ## test dataformat_convertor ##
    cfg_path = './data_configs/mpas_dataset.yaml'
    cvtor = data_format_convertor('./data_configs/mpas_dataset.yaml')
    import pdb; pdb.set_trace()



