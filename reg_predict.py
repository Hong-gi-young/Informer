from exp.exp_basic import Exp_Basic
from models import Informer
# from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import torch
import torch.nn as nn
from models import Informer
import torch.nn.functional as F
from torch import optim
from utils.timefeatures import time_features
from torch.utils.data import Dataset, DataLoader
import os
import warnings
from sklearn.preprocessing import StandardScaler
from pickle import dump
from pickle import load
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from Config import total_target 
import random
import argparse

class Dataset_Pred(Dataset):
    def __init__(self, df_raw, target,
                data_path, scaler_path, 
                features='MS', scale=True, inverse=False, timeenc=0, freq='15min',size=[48,48,1]):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.data_path = data_path
        self.df_raw = df_raw
        self.scaler_path = scaler_path
        self.__read_data__()

    def __read_data__(self):
        cols = total_target[self.target]
        # df_raw = pd.read_csv(self.data_path)
        df_raw = self.df_raw[['reg_date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len 
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            #불러오기 
            scaler = scaler = load(open(self.scaler_path, 'rb'))
            data = scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        tmp_stamp = df_raw[['reg_date']][border1:border2] 
        tmp_stamp['reg_date'] = pd.to_datetime(tmp_stamp.reg_date)
        pred_dates = pd.date_range(tmp_stamp.reg_date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns=['reg_date'])
        df_stamp.reg_date = list(tmp_stamp.reg_date.values) + list(pred_dates[1:])
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.reg_date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.reg_date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.reg_date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.reg_date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.reg_date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['reg_date'], 1).values
            
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['reg_date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2] #스케일링 안된거
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def _acquire_device(use_gpu=True):
    if use_gpu:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
        print('Use CPU')
    return device

def predict(args):
    seq_len = args.seq_len
    label_len = args.label_len
    pred_len = args.pred_len
    
    df_raw = pd.read_csv(args.data_path, encoding='utf8')
    predict_values = []
    for t in range(len(df_raw)- int(seq_len) -1):
        # print('restart',t,":",t + seq_len)
        df = df_raw.iloc[t:t + seq_len]
        # print(df)
        pred_loader = Dataset_Pred(df,target,args.data_path,scaler_path)
        pred_loader = DataLoader(pred_loader, batch_size=1, shuffle=False)
        device=_acquire_device()
        model = Informer.Model(args).float().to(device)
        model.load_state_dict(torch.load(best_model_path))

        # preds = []
        model.eval()
        number = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device) #[1,48,9],[1,1,9] => [1,49,9]?
                print('dec_inp shape',dec_inp.shape)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  #.squeeze()
                number += 1
                
        #regression
        pred = np.array(pred)
        
        #스케일러 불러오기
        scaler = joblib.load(f'/home/seoul/홍기영/스마트팜test/{target}_pred_scaler.pkl')
        pred = scaler.inverse_transform(pred[0])
        print('저장 후',pred)
        predict_values.append(pred)
        
    # result save
    folder_path = './predicts/' + target + '_' +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cols = total_target[target]
    predict_values=np.array(predict_values)
    predict_values = predict_values.reshape(-1,len(cols)+1)
    print('predict_values',predict_values)
    pd.DataFrame(predict_values, columns=[cols+[target]]).to_excel(f'{target}prediction.xlsx')
        
    # 예측 성능 
    for i in range(len(predict_values)):
        df_raw.loc[args.seq_len+i,'pred']  = float(predict_values[i][-1])
    df_eval =df_raw[df_raw['pred'].notnull()]
    print('df_eval',df_eval)
    mae, mse = metric(df_eval[target],df_eval['pred'])
    print(f'MSE : {mse}, MAE : {mae}')
    
def define_argparser(targets,in_out,best_model_path,scaler_path):
    parser = argparse.ArgumentParser(description='Informer & Transformer family for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Informer',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='/home/seoul/홍기영/스마트팜test/dataset/상주_딸기_보온시기.csv', help='data file')
    parser.add_argument('--best_model_path', type=str, default=best_model_path, help='data file')
    parser.add_argument('--scaler_path', type=str, default=scaler_path, help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    parser.add_argument('--target', type=str,  nargs='+', default=targets, help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # DLinear
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=in_out, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=in_out, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=in_out, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation') 
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=False)
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data', default=True)
    # parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()
    
    return args


if __name__=='__main__':
    target = 'cg_curtain'
    in_out = len(total_target[target])+1
    best_model_path = '/home/seoul/홍기영/스마트팜test/checkpoints/cg_curtain/test_Informer_custom_ftMS_sl48_ll48_pl1__dm512_nh8_el2_dl1_df2048_fc1_ebfixed_dtTrue_test_0/0.0407627_0.0574807_checkpoint.pth'
    scaler_path = f'/home/seoul/홍기영/스마트팜test/{target}_pred_scaler.pkl'
    args = define_argparser(target,in_out,best_model_path,scaler_path)
    predict(args)