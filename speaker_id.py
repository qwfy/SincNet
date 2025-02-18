# speaker_id.py
# Mirco Ravanelli
# Mila - University of Montreal

# July 2018

# Description:
# This code performs a speaker_id experiments with SincNet.

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf

from data_io import ReadList
from data_io import read_conf
from data_io import str_to_bool
from dnn_models import MLP
from dnn_models import SincNet as CNN

IS_DATA_PARALLEL = False
DEVICE_IDS = [0]
# IS_DATA_PARALLEL = True
# DEVICE_IDS = list(range(8))
device = torch.device(f"cuda:{DEVICE_IDS[0]}")

def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
  # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])
  sig_batch = np.zeros([batch_size, wlen])
  lab_batch = np.zeros(batch_size)

  snt_id_arr = np.random.randint(N_snt, size=batch_size)

  rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)

  for i in range(batch_size):
    signal, _ = sf.read(data_folder + wav_lst[snt_id_arr[i]])

    # accessing to a random chunk
    snt_len = signal.shape[0]
    snt_beg = np.random.randint(snt_len - wlen - 1)
    snt_end = snt_beg + wlen

    channels = len(signal.shape)
    if channels >= 2:
      assert False

    sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]
    lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]]]

  inp = torch.from_numpy(sig_batch).float().cuda(device).contiguous()
  lab = torch.from_numpy(lab_batch).float().cuda(device).contiguous()

  return inp, lab


# Reading cfg file
options = read_conf()
print(options)
# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder + '/'
output_folder = options.output_folder

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
wav_lst_tr = ReadList(tr_lst)
snt_tr = len(wav_lst_tr)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
  os.stat(output_folder)
except:
  os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
cost = nn.NLLLoss()

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Feature extractor CNN
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net_out_dim = CNN_net.out_dim
if IS_DATA_PARALLEL:
  CNN_net = nn.DataParallel(CNN_net, device_ids=DEVICE_IDS)
CNN_net.cuda(device)

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()

DNN1_arch = {'input_dim': CNN_net_out_dim,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }

DNN1_net = MLP(DNN1_arch)
if IS_DATA_PARALLEL:
  DNN1_net = nn.DataParallel(DNN1_net, device_ids=DEVICE_IDS)
DNN1_net.cuda(device)

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,
             }

DNN2_net = MLP(DNN2_arch)
if IS_DATA_PARALLEL:
  DNN2_net = nn.DataParallel(DNN2_net, device_ids=DEVICE_IDS)
DNN2_net.cuda(device)

if pt_file != 'none':
  checkpoint_load = torch.load(pt_file)
  CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
  DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
  DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

print('localtime when starting:', time.localtime())
for epoch in range(N_epochs):
  epoch_start = time.monotonic()
  test_flag = 0
  CNN_net.train()
  DNN1_net.train()
  DNN2_net.train()

  loss_sum = 0
  err_sum = 0

  for i in range(N_batches):
    [inp, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_tr, snt_tr, wlen, lab_dict, 0.2)
    pout = DNN2_net(DNN1_net(CNN_net(inp)))

    pred = torch.max(pout, dim=1)[1]
    loss = cost(pout, lab.long())
    err = torch.mean((pred != lab.long()).float())

    optimizer_CNN.zero_grad()
    optimizer_DNN1.zero_grad()
    optimizer_DNN2.zero_grad()

    loss.backward()
    optimizer_CNN.step()
    optimizer_DNN1.step()
    optimizer_DNN2.step()

    loss_sum = loss_sum + loss.detach()
    err_sum = err_sum + err.detach()

  loss_tot = loss_sum / N_batches
  err_tot = err_sum / N_batches

  # Full Validation  new
  if epoch % N_eval_epoch == 0:

    CNN_net.eval()
    DNN1_net.eval()
    DNN2_net.eval()
    test_flag = 1
    loss_sum = 0
    err_sum = 0
    err_sum_snt = 0

    with torch.no_grad():
      for i in range(snt_te):
        signal, _ = sf.read(data_folder + wav_lst_te[i])
        lab_batch = lab_dict[wav_lst_te[i]]

        # 1 2 3 4 5 6 7 8 9 10 11 12 13
        # stride = 3
        # window = 5
        # we should have (13 - 5) // 3 + 1 = 3 examples
        # for we have (13 - 5) segments, each segment corresponds to a left end point,
        # and each left point corresponds to one example,
        # plus the one at the end
        # 1 2 3 4 5
        # 4 5 6 7 8
        # 7 8 9 10 11

        N_fr_actual = 0
        N_fr = (signal.shape[0] - wlen) // wshift + 1
        sig_arr = np.zeros((N_fr, wlen), dtype=np.float32)
        lab = (torch.zeros(N_fr) + lab_batch).cuda(device).contiguous().long()
        for i_sig, beg_samp in enumerate(range(0, signal.shape[0], wshift)):
          end_samp = beg_samp + wlen
          if end_samp > signal.shape[0]:
            break
          else:
            sig_arr[i_sig] = signal[beg_samp:end_samp]
            N_fr_actual += 1
        assert N_fr_actual == N_fr
        sig_arr = torch.from_numpy(sig_arr).cuda(device).contiguous()
        pout = DNN2_net(DNN1_net(CNN_net(sig_arr)))

        pred = torch.max(pout, dim=1)[1]
        loss = cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())

        [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
        err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

        loss_sum = loss_sum + loss.detach()
        err_sum = err_sum + err.detach()

      err_tot_dev_snt = err_sum_snt / snt_te
      loss_tot_dev = loss_sum / snt_te
      err_tot_dev = err_sum / snt_te

    print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f time=%f" % (
      epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt, time.monotonic() - epoch_start))

    with open(output_folder + "/res.res", "a") as res_file:
      res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f time=%f\n" % (
        epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt, time.monotonic() - epoch_start))

    checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                  'DNN1_model_par': DNN1_net.state_dict(),
                  'DNN2_model_par': DNN2_net.state_dict()}
    torch.save(checkpoint, output_folder + f'/model_raw_{epoch}.pkl')

  else:
    print("epoch %i, loss_tr=%f err_tr=%f time=%f" % (
      epoch, loss_tot, err_tot, time.monotonic() - epoch_start))
