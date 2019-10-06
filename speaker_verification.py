# python speaker_verification.py --cfg=cfg/SincNet_TIMIT.cfg

import collections
import os
import random
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader

from data_io import read_conf
from data_io import str_to_bool
from dnn_models import MLP
from dnn_models import SincNet as CNN

IS_DATA_PARALLEL = False
DEVICE_IDS = [0]
# IS_DATA_PARALLEL = True
# DEVICE_IDS = list(range(8))
device = torch.device(f'cuda:{DEVICE_IDS[0]}')

LABEL_SAME_CLASS = 0
LABEL_DIFF_CLASS = 1

M = 0.1

def read_list(file_path):
  lines = []
  with open(file_path, 'r') as f:
    for line in f.read().splitlines():
      line = line.strip()
      if line:
        lines.append(line)
  print(f'{file_path}: {len(lines)} lines')
  return lines


def timit_by_speaker(audio_file_list):
  # train/dr1/mpgh0/si675.wav
  by_speaker = collections.defaultdict(lambda: [])
  for audio_file in audio_file_list:
    speaker = audio_file.split('/')[2]
    by_speaker[speaker].append(audio_file)
  by_speaker = dict(by_speaker)
  return by_speaker


class Dataset(IterableDataset):

  def __init__(self, data_folder, by_speaker, wlen, fact_amp, max_samples):
    self.data_folder = data_folder
    self.wlen = wlen
    self.fact_amp = fact_amp
    self.by_speaker = by_speaker
    self.all_speakers = list(by_speaker.keys())
    self.max_samples = max_samples
    self.num_sampled = 0

  def __iter__(self):
    return self

  def load_random_chunk(self, file_path):
    signal, _ = sf.read(os.path.join(self.data_folder, file_path))
    channels = len(signal.shape)
    if channels >= 2:
      assert False
    # accessing to a random chunk
    snt_len = signal.shape[0]
    snt_beg = np.random.randint(snt_len - wlen - 1)
    snt_end = snt_beg + wlen

    rand_amp = np.random.uniform(1.0 - self.fact_amp, 1 + self.fact_amp)
    chunk = signal[snt_beg:snt_end] * rand_amp
    chunk = torch.from_numpy(chunk).float().cuda(device)
    return chunk

  def __next__(self):
    # TODO @incomplete: hard negative mining
    if self.max_samples is None or self.num_sampled < self.max_samples:
      self.num_sampled += 1
      label = np.random.choice([LABEL_SAME_CLASS, LABEL_DIFF_CLASS])
      if label == LABEL_SAME_CLASS:
        speaker = random.choice(self.all_speakers)
        file_a, file_b = random.choices(self.by_speaker[speaker], k=2)
      elif label == LABEL_DIFF_CLASS:
        speaker_a = random.choice(self.all_speakers)
        file_a = random.choice(self.by_speaker[speaker_a])
        while True:
          speaker_b = random.choice(self.all_speakers)
          if speaker_b != speaker_a:
            break
        file_b = random.choice(self.by_speaker[speaker_b])
      else:
        assert False

      a = self.load_random_chunk(file_a)
      b = self.load_random_chunk(file_b)
      return a, b, label
    else:
      raise StopIteration()


# Reading cfg file
options = read_conf()
print(options)
# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder
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

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
wav_lst_tr = read_list(tr_lst)
by_speaker_tr = timit_by_speaker(audio_file_list=wav_lst_tr)

# test list
wav_lst_te = read_list(te_lst)
by_speaker_te = timit_by_speaker(audio_file_list=wav_lst_te)

os.makedirs(output_folder, exist_ok=True)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

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
            'cnn_drop': cnn_drop}

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
             'fc_act': fc_act}

DNN1_net = MLP(DNN1_arch)
if IS_DATA_PARALLEL:
  DNN1_net = nn.DataParallel(DNN1_net, device_ids=DEVICE_IDS)
DNN1_net.cuda(device)

if pt_file != 'none':
  checkpoint_load = torch.load(pt_file)
  CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
  DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])

optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

def calc_loss(xs, ys, labels):
  dw = torch.norm((xs - ys), p=2, dim=1)
  ls = dw ** 2 / 2
  ld = torch.max(M - dw, torch.zeros(dw.shape, dtype=torch.float32).cuda(device))
  ld = ld ** 2 / 2
  loss = (1 - labels) * ls + labels * ld
  loss = torch.mean(loss)
  return loss

print('localtime when starting:', time.localtime())
train_data_set = Dataset(
  data_folder=data_folder,
  by_speaker=by_speaker_tr,
  wlen=wlen,
  fact_amp=0.2,
  max_samples=None)
train_data_loader = DataLoader(train_data_set, batch_size=batch_size)
train_data_iter = train_data_loader.__iter__()
for epoch in range(N_epochs):
  epoch_start = time.monotonic()
  test_flag = 0
  CNN_net.train()
  DNN1_net.train()

  loss_sum = 0

  for i in range(N_batches):
    part_a, part_b, labels = next(train_data_iter)
    labels = labels.float().cuda()
    vectors_a = DNN1_net(CNN_net(part_a))
    vectors_b = DNN1_net(CNN_net(part_b))

    loss = calc_loss(vectors_a, vectors_b, labels)

    optimizer_CNN.zero_grad()
    optimizer_DNN1.zero_grad()
    loss.backward()
    optimizer_CNN.step()
    optimizer_DNN1.step()

    loss_sum = loss_sum + loss.detach()

  loss_tr = loss_sum / N_batches

  stat = f'epoch={epoch:03d} time={int(time.monotonic() - epoch_start)} loss_tr={loss_tr:.4f}'

  if epoch % N_eval_epoch == 0:

    CNN_net.eval()
    DNN1_net.eval()
    loss_sum = 0

    with torch.no_grad():
      samples_to_eval = (len(by_speaker_te) * 2 // batch_size + 1) * batch_size
      eval_data_set = Dataset(
        data_folder=data_folder,
        by_speaker=by_speaker_te,
        wlen=wlen,
        fact_amp=0,
        max_samples=samples_to_eval)
      eval_data_loader = DataLoader(eval_data_set, batch_size=batch_size)
      for part_a, part_b, labels in eval_data_loader:
        labels = labels.float().cuda()
        vectors_a = DNN1_net(CNN_net(part_a))
        vectors_b = DNN1_net(CNN_net(part_b))
        loss = calc_loss(vectors_a, vectors_b, labels)
        loss_sum = loss_sum + loss.detach()
      loss_te = loss_sum / (samples_to_eval / batch_size)

    stat += f' loss_te={loss_te:.4f}'

    with open(os.path.join(output_folder, 'res.res'), 'a') as res_file:
      res_file.write(stat + '\n')

    checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                  'DNN1_model_par': DNN1_net.state_dict()}
    torch.save(checkpoint, os.path.join(output_folder, f'model_raw_{epoch:03d}.pkl'))

  print(stat)
