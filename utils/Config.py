class Config():
    def __init__(self):
        self.Config = {
        'is_training' : 1 ,
        'model_id' : 'test',
        'model' : 'Informer',
        'data'  : 'custom', # dataset type
        'root_path' : './data/ETT/',
        'data_path' : '상주_경영형_딸기_ver1128_3.csv',
        'features'  : 'MS',
        'target'  : ['sky_window_1_L','sky_window_1_R','side_window_1_L','side_window_1_R','flow_fan','hmdfc','light_beam','co2_occur','crc_pump_1','crc_pump_2','3way_valve_1','3way_valve_2','heat_cooler','fumi_gator'],
        'freq' :'h',
        'checkpoints' :'./checkpoints/', 
        'seq_len' : 48 ,
        'label_len' : 48 ,
        'pred_len'  : 96 ,
        'individual' : False , # DLinear
        'embed_type' : 0,
        'enc_in' :7,
        'dec_in' :7,
        'c_out' :7,
        'd_model' : 512 ,
        'n_heads' :8, 
        'e_layers' :2,
        'd_layers' :1,
        'd_ff'  : 2048,
        'moving_avg' :25,
        'factor'  :1,
        'distil' :True,
        'dropout':0.05,
        'embed' :'fixed',
        'activation' :'gelu', 
        'do_predict' : True,
        'num_workers' :10, 
        'itr' :1,
        'train_epochs':50,
        'batch_size'  : 64,
        'patience' : 3,
        'learning_rate':0.0001,
        'des' :'Exp',
        'loss' :'mse',
        'lradj' :'type1' ,
        'use_amp':False,
        'use_gpu':True,
        'gpu' :1,
        'devices' : '0123',
        'test_flop' :False 
        }
        return self.Config

class total_target:
    def __init__(self, target):
            
        self.total_target = {
            'sky_window': ['hd_x', 'co2', 'tp_y', 'sr', 'tp_x', 'pc', 'wd', 'ws'],
            'flow_fan': ['hd_x', 'tp_y', 'tp_x', 'sr', 'co2', 'pc', 'ws', 'wd'],
            'hmdfc': ['hd_x', 'tp_y', 'sr', 'co2', 'ws', 'wd', 'tp_x'],
            # 'light_beam': ['hd_x', 'co2', 'tp_y', 'sr', 'tp_x', 'pc', 'wd', 'ws'],
            # 'co2_occur': ['hd_x', 'co2', 'tp_y', 'sr', 'tp_x', 'pc', 'wd', 'ws'], #['sr', 'tp_y', 'tp_x', 'pc', 'co2'],
            'crc_pump': ['co2', 'tp_y', 'sr', 'hd_x', 'tp_x', 'wd', 'ws', 'pc'],
            '3way_valve_1': ['co2', 'tp_y', 'sr', 'hd_x', 'tp_x', 'pc', 'ws'],
            '3way_valve_2': ['co2', 'tp_y', 'hd_x', 'sr', 'tp_x', 'ws', 'pc', 'wd'],
            'heat_cooler': ['hd_x', 'co2', 'sr', 'tp_x', 'tp_y', 'pc'],
            # 'fumi_gator': ['hd_x', 'co2', 'tp_y', 'sr', 'tp_x', 'pc', 'wd', 'ws'],
            # 'side_window' : ['tp_x','hd_x','co2','tp_y','wd','ws','pc'],
            'cg_curtain' : ['tp_x','hd_x','co2','tp_y','wd','ws','pc','sr']
            }
        result = self.total_target[target]
        return result + 1 