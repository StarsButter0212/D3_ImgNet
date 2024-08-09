#!/usr/bin/env python3
import os
import time
import sys
from pathlib import Path
import pickle
import timeit
import collections
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn.modules.utils import _pair

sys.path.append('../')
import train.train_args as args
from datetime import datetime
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
np.set_printoptions(threshold=1e6)  # Unomitted display
np.set_printoptions(linewidth=300)  # No wrap
torch.set_printoptions(threshold=1e6)  # Unomitted display
torch.set_printoptions(linewidth=300)  # No wrap


# *****************************************************************************************
class SplAtConv2d(nn.Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, radix=2, reduction_factor=4, norm_layer=None,
                 bias=True, **kwargs):
        super(SplAtConv2d, self).__init__()

        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.use_bn = norm_layer is not None

        padding = _pair(padding)
        inter_channels = max(in_channels * radix // reduction_factor, 32)

        self.conv = nn.Conv2d(in_channels, channels * radix, kernel_size, stride, padding,
                              dilation, groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(channels * radix)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = norm_layer(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=self.cardinality)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)

        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class BasicBlock(nn.Module):
    """ResNet BasicBlock
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, avd=False, avd_first=False,
                 dilation=1, is_first=False, norm_layer=None):
        super(BasicBlock, self).__init__()

        self.radix = radix
        self.avd = avd and (stride > 1 or is_first)
        self.avd_first = avd_first

        if self.avd:
            self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
            stride = 1

        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)

        if radix >= 1:
            self.conv2 = SplAtConv2d(
                planes, planes, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality,
                radix=radix, norm_layer=norm_layer, bias=False)
        else:
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        if self.avd and self.avd_first:
            out = self.avd_layer(x)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv2(out)
        if self.radix == 0:
            out = self.bn2(out)
            out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


# *****************************************************************************************
class D3_ImgNet(nn.Module):
    """ Main network."""

    def __init__(self, device, basis_file, N_type_orbitals, dim, seq_len,
                 operation, dropout=0):
        super(D3_ImgNet, self).__init__()

        """All learning parameters of the model."""
        self.basis_file = '../basissets/' + basis_file + '.gbs'
        self.coefficient = nn.Embedding(N_type_orbitals, dim)
        self.zeta = nn.Embedding(N_type_orbitals, 1)  # Orbital exponent.
        nn.init.ones_(self.zeta.weight)

        # Network architecture and related params.
        self.dilated = True
        self.drop = dropout
        self.seq_len = seq_len

        block = BasicBlock
        layers = [1, 1, 1, 1]
        radix = 2
        groups = 4
        avg_down = True
        avd = True
        avd_first = False
        norm_layer = nn.BatchNorm2d

        dilated = False
        dilation = 1

        self.inplanes = 32
        self.radix = radix
        self.cardinality = groups
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 32, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2, norm_layer=norm_layer)

        if dilated or dilation == 4:
            self.layer3 = self._make_layer(block, 128, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilation == 2:
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                           dilation=1, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)
        else:
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 256, layers[3], stride=2,
                                           norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.res_output = nn.Linear(256, 1)

        self._initialize_weights()

        # Other parameters
        self.device = device
        self.dim = dim
        self.operation = operation

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None,
                    is_first=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                if dilation == 1:
                    down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                else:
                    down_layers.append(nn.AvgPool2d(kernel_size=1, stride=1,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))
            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=1, is_first=is_first, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                                radix=self.radix, cardinality=self.cardinality,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=2, is_first=is_first, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                radix=self.radix, cardinality=self.cardinality,
                                avd=self.avd, avd_first=self.avd_first,
                                dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def list_to_batch(self, xs, dtype=torch.FloatTensor, set=None, cat=None, axis=None):
        """Transform the list of numpy data into the batch of tensor data."""
        if set:
            xs = np.concatenate([x for x in xs])
        else:
            xs = [dtype(x).to(self.device) for x in xs]

        if cat:
            return torch.cat(xs, axis)
        else:
            return xs

    def basis_matrix(self, atomic_orbitals, distance_matrices, n_quantum_numbers, l_quantum_numbers):
        """Transform the distance matrix into a basis matrix_1."""
        zetas = torch.squeeze(self.zeta(atomic_orbitals))
        GTOs = (distance_matrices ** (l_quantum_numbers) *
                distance_matrices ** (2 * (n_quantum_numbers - l_quantum_numbers - 1)) *
                torch.exp(-zetas * distance_matrices ** 2))
        GTOs = F.normalize(GTOs, 2, 0)  # (input, p, dim)
        return GTOs

    def electron_density(self, KS_orbitals):
        """Create the electron density of a molecular."""
        densities = KS_orbitals ** 2
        densities = torch.split(densities, self.seq_len, dim=0)
        atoms = len(densities)
        return torch.stack(densities), atoms

    def train_densities_data(self, inputs):
        """Linear combination of atomic orbitals (LCAO)."""

        """Inputs."""
        (atomic_orbitals, distance_matrices,
         n_quantum_numbers, l_quantum_numbers, N_electrons) = inputs

        """Cat or pad each input data for batch processing."""
        atomic_orbitals = self.list_to_batch(atomic_orbitals, torch.LongTensor)
        distance_matrices = self.list_to_batch(distance_matrices)
        n_quantum_numbers = self.list_to_batch(n_quantum_numbers)
        l_quantum_numbers = self.list_to_batch(l_quantum_numbers)
        N_electrons = self.list_to_batch(N_electrons)

        """Normalize the coefficients in LCAO."""
        densities, atoms = [], []
        for atomic_orbital, distance_matrice, n_quantum_number, l_quantum_number, N_elec in \
                zip(atomic_orbitals, distance_matrices, n_quantum_numbers, l_quantum_numbers, N_electrons):
            basis_sets = self.basis_matrix(atomic_orbital, distance_matrice, n_quantum_number, l_quantum_number)
            orbital_coefs = F.normalize(self.coefficient(atomic_orbital), p=2, dim=0)

            KS_orbitals = F.normalize(torch.matmul(basis_sets, orbital_coefs), p=2, dim=0)
            KS_orbitals = torch.sqrt(N_elec / orbital_coefs.size()[1]) * KS_orbitals
            density, atom = self.electron_density(KS_orbitals)

            densities.append(density)
            atoms.append(atom)

        return torch.cat(densities), atoms

    def img2point(self, input, operation='sum', axis=None):
        x = input.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if operation == 'sum':
            vectors = [torch.sum(vs, 0) for vs in torch.split(x, axis, dim=0)]
        if operation == 'mean':
            vectors = [torch.mean(vs, 0) for vs in torch.split(x, axis, dim=0)]
        return torch.stack(vectors)

    def forward(self, data, train=False, target=None, predict=False):

        idx, inputs, energies = data[0], data[2:7], data[7]
        N_atoms, molecular_formula = data[1], data[8]

        if predict:
            with torch.no_grad():
                densities, atoms = self.train_densities_data(inputs)
                final_layer = self.img2point(densities,
                                             self.operation, atoms)
                E_ = self.res_output(final_layer)
            return idx, E_

        elif train:
            densities, atoms = self.train_densities_data(inputs)

            if target == 'E':  # Supervised learning for energy.
                E = self.list_to_batch(energies, cat=True, axis=0)  # Correct E.
                final_layer = self.img2point(densities,
                                             self.operation, atoms)
                E_ = self.res_output(final_layer)  # Predicted E.
                loss = F.mse_loss(E, E_)
                loss_ = F.mse_loss(E, E_, reduction='sum')
            return loss, loss_

        else:  # Test.
            with torch.no_grad():
                E = self.list_to_batch(energies, cat=True, axis=0)
                densities, atoms = self.train_densities_data(inputs)
                final_layer = self.img2point(densities,
                                             self.operation, atoms)
                E_ = self.res_output(final_layer)
                return idx, N_atoms, E, E_, molecular_formula


class Trainer(object):
    def __init__(self, model, lr):
        self.model = model

        self.optimizer = optim.AdamW(self.model.parameters(), lr, weight_decay=1e-3)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, 5 * lr,
                                                       total_steps=270, pct_start=0.2)

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader, epoch):
        self.model.train()

        """Minimize two loss functions in terms of E."""
        losses_E, losses_E_sum = 0, 0
        loop_data = tqdm(enumerate(dataloader), total=len(dataloader))
        start_time = datetime.now()

        for index, data in loop_data:
            loss_E, loss_E_sum = self.model.forward(data, train=True, target='E')
            self.optimize(loss_E, self.optimizer)
            losses_E += loss_E.item()
            losses_E_sum += loss_E_sum.item()

            delta_time = datetime.now() - start_time
            loop_data.set_description('\33[36m【Epoch {0:04d}】'.format(epoch))
            loop_data.set_postfix({'loss_E_sum': '{0:.6f}'.format(losses_E_sum),
                                   'loss_E': '{0:.6f}'.format(losses_E),
                                   'cost_time': '{0}'.format(delta_time)}, '\33[0m')

        self.scheduler.step()
        return losses_E / len(dataloader), losses_E_sum


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader):
        self.model.eval()
        N = sum([len(data[0]) for data in dataloader])
        IDs, N_atoms, Es, Es_, molecular_formula = [], [], [], [], []
        SAE, loss_val, losses_val = 0, 0, 0  # Sum absolute error.

        for i, data in enumerate(dataloader):
            idx, AOs, E, E_, MOL = self.model.forward(data)
            SAE_batch = torch.sum(torch.abs(E - E_), 0)
            SAE += SAE_batch
            IDs += list(idx)
            N_atoms += list(AOs)
            Es += E.tolist()
            Es_ += E_.tolist()
            molecular_formula += list(MOL)
            loss_val += F.mse_loss(E, E_).item()  # Mean squared error.
            losses_val += F.l1_loss(E, E_,
                                    reduction='sum').item()  # Sum of mean absolute error.

        MAE = (SAE / N).tolist()  # Mean absolute error.
        MAE = ','.join([str(m) for m in MAE])  # For homo and lumo.

        prediction = 'ID\tN_atoms\tMolecular\tCorrect\tPredict\tError\n'
        for idx, AOs, E, E_, MOL in zip(IDs, N_atoms, Es, Es_, molecular_formula):
            error = np.abs(np.array(E) - np.array(E_))
            error = ','.join([str(e) for e in error])
            E = ','.join([str(e) for e in E])
            E_ = ','.join([str(e) for e in E_])
            prediction += '\t'.join([idx, str(AOs), MOL, E, E_, error]) + '\n'

        return MAE, prediction, loss_val / len(dataloader), losses_val

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, directory, index=None):
        self.directory = directory
        self.paths = sorted(Path(self.directory).iterdir(),
                            key=os.path.getmtime)

        if index is not None:
            self.files = [str(p).strip().split('\\')[-1]
                          for p in np.asarray(self.paths)[index]]
        else:
            self.files = [str(p).strip().split('\\')[-1]
                          for p in self.paths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return np.load(self.directory + self.files[idx], allow_pickle=True)


def lambda_xs(xs):
    return list(zip(*xs))


def mydataloader(dataset, batch_size, num_workers, shuffle=False):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=lambda_xs, pin_memory=True)
    return dataloader


if __name__ == "__main__":

    """Args."""
    dataset = args.dataset
    unit = '(' + dataset.split('_')[-1] + ')'  # eV or Hartree
    basis_set = args.basis_set
    dim = args.dim
    seq_len = args.seq_len

    batch_size = args.batch_size
    operation = args.operation
    lr = args.lr
    iteration = args.iteration
    dropout = args.dropout
    num_workers = args.num_workers

    """Fix the random seed."""
    torch.manual_seed(1729)

    """GPU or CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU.')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU.')
    print('-' * 50)

    """Create the dataloaders of training, val, and test set."""
    dir_dataset = '../dataset/' + dataset + '/' + 'create_data' + '_' + basis_set + '/'
    field = '_'.join([basis_set + '/'])

    dataset_train = MyDataset(dir_dataset + 'train_' + field)
    dataset_val = MyDataset(dir_dataset + 'val_' + field)
    dataset_test = MyDataset(dir_dataset + 'test_' + field)

    dataloader_train = mydataloader(dataset_train, batch_size, num_workers, shuffle=True)
    dataloader_val = mydataloader(dataset_val, batch_size, num_workers)
    dataloader_test = mydataloader(dataset_test, batch_size, num_workers)

    print('# of training samples: ', len(dataset_train))
    print('# of validation samples: ', len(dataset_val))
    print('# of test samples: ', len(dataset_test))
    print('-' * 50)

    """Load orbital_dict generated in preprocessing."""
    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)

    print('Set a D3-ResNet model.')
    print('# of n_iterations:', iteration)

    N_orbitals = len(orbital_dict)
    print('# of n_orbitals:', N_orbitals)
    print('# of n_seq_len:', seq_len)

    """The output dimension in regression."""
    N_output = len(dataset_test[0][-2][0])

    model = D3_ImgNet(device, basis_set, N_orbitals, dim,
                      seq_len, operation, dropout=dropout).to(device)

    trainer = Trainer(model, lr)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 50)

    """Output files."""
    file_result = '../output/result--' + dataset + '.txt'
    result = ('Epoch\tTime(sec)\tloss_E_sum\tlosses_val\tLoss_E\t'
              'Loss_val\tMAE_val' + unit + '\tMAE_test' + unit)

    with open(file_result, 'w') as f:
        f.write(result + '\n')

    file_prediction = '../output/prediction--' + dataset + '.txt'
    file_model = '../output/model--' + dataset + '.pth'

    print('Start training of the D3_ImgNet model...')

    start = timeit.default_timer()
    loss_val, MAE_min = np.inf, np.inf

    for epoch in range(iteration):
        loss_E, loss_E_sum = trainer.train(dataloader_train, epoch)
        MAE_val, _, loss_val, losses_val = tester.test(dataloader_val)
        MAE_test, prediction, _, losses_test = tester.test(dataloader_test)

        tqdm.write('\33[34m【losses_val】: {0}, 【MAE_val】: {1}\33[0m'.format(losses_val, MAE_val))
        tqdm.write('\33[34m【losses_test】: {0}, 【MAE_test】: {1}\33[0m'.format(losses_test, MAE_test))
        time.sleep(0.1)

        cost_time = timeit.default_timer() - start
        result = '\t'.join(map(str, [epoch, cost_time, loss_E_sum, losses_val,
                                     loss_E, loss_val, MAE_val, MAE_test]))
        tester.save_result(result, file_result)

        # Save the model with the best performance at last 10 epochs
        if epoch >= (iteration - 10):
            if float(MAE_test) < MAE_min:
                MAE_min = float(MAE_test)
                tester.save_prediction(prediction, file_prediction)
                tester.save_model(model, file_model)

    print('The training has finished -- MAE_min: {}.'.format(MAE_min))
