import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules import FilterLinear

# choose device
USE_CUDA = torch.cuda.is_available()
if not USE_CUDA:
    DEVICE = torch.device('cpu')
else:
    DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE)

class tconv_encode(nn.Module):
    """
    Performed on batch_size*detector*7 data, to merge the time features of previous days.
    Output is batch_size*detector*1
    """
    def __init__(self, feature_size):
        super(tconv_encode, self).__init__()

        self.feature_size = feature_size # number of detectors

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(2, 3), stride=(1, 1), dilation=(feature_size, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), dilation=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(p=0.2))
        self.block4 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1, 3), stride=(1, 1), dilation=(1, 1), padding=(0, 1)),
            nn.ReLU())

    def forward(self, input):
        x = input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class linear_encode(nn.Module):
    """
    Performed on batch_size*detector*7 data, to merge the time features of previous days.
    Output is batch_size*detector*1
    """
    def __init__(self, feature_size):
        super(linear_encode, self).__init__()

        self.feature_size = feature_size # number of detectors

        self.block1 = nn.Sequential(
            nn.Linear(feature_size, feature_size//8),
            nn.Sigmoid(),
            nn.Dropout(p=0.2))
        self.block2 = nn.Sequential(
            nn.Linear(feature_size//8, feature_size//8),
            nn.Sigmoid(),
            nn.Dropout(p=0.2))
        self.block3 = nn.Sequential(
            nn.Linear(feature_size//8, feature_size*2),
            nn.Sigmoid())

    def forward(self, input):
        x = input
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class tDKFN(nn.Module):

    def __init__(self, K, A, feature_size, pred_size=1, ext_feature_size = 4, Clamp_A=True):
        # GC-LSTM
        '''
        Args:
            K: K-hop graph
            A: adjacency matrix
            feature_size: number of nodes
            pred_size: the length of output
            ext_feature_size: number of extra features. Only include periodical features by default
            Clamp_A: Boolean value, clamping all elements of A between 0. to 1.
        '''
        super(tDKFN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = feature_size
        self.ext_feature_size =  ext_feature_size
        self.pred_size = pred_size
        self.K = K
        self.A_list = []  # Adjacency Matrix List

        # normalization
        D_inverse = torch.diag(1 / torch.sum(A, 0))
        norm_A = torch.matmul(D_inverse, A)
        A = norm_A

        A_temp = torch.eye(feature_size, feature_size)

        # assert not torch.isnan(A_temp).any()

        for i in range(K):
            A_temp = torch.matmul(A_temp, A)
            if Clamp_A:
                # confine elements of A
                A_temp = torch.clamp(A_temp, max=1.)

            assert not torch.isnan(A_temp).any(), str(i)

            self.A_list.append(A_temp)

        # a length adjustable Module List for hosting all graph convolutions
        self.gc_list = nn.ModuleList([FilterLinear(feature_size, feature_size, self.A_list[i], bias=False) for i in range(K)])

        hidden_size = self.feature_size
        gc_input_size = self.feature_size * K

        self.fl = nn.Linear(gc_input_size + hidden_size*2, hidden_size) # input: gcn output, pred and residual
        self.il = nn.Linear(gc_input_size + hidden_size*2, hidden_size)
        self.ol = nn.Linear(gc_input_size + hidden_size*2, hidden_size)
        self.Cl = nn.Linear(gc_input_size + hidden_size*2, hidden_size)

        # initialize the neighbor weight for the cell state
        self.Neighbor_weight = Parameter(torch.FloatTensor(feature_size))
        stdv = 1. / math.sqrt(feature_size)
        self.Neighbor_weight.data.uniform_(-stdv, stdv)

        # RNN
        input_size = self.feature_size

        self.tconv = tconv_encode(feature_size=self.feature_size) ################### timestep convolution

        self.rfl = nn.Linear(input_size + hidden_size*2, hidden_size) # input: tcn output, seasonal, pred and residual
        self.ril = nn.Linear(input_size + hidden_size*2, hidden_size)
        self.rol = nn.Linear(input_size + hidden_size*2, hidden_size)
        self.rCl = nn.Linear(input_size + hidden_size*2, hidden_size)

        # addtional vars
        self.c = torch.nn.Parameter(torch.Tensor([1]))
        # self.cf = torch.nn.Parameter(torch.ones(8))

        # multiple output linear layer
        self.fc_mo = linear_encode(self.feature_size)

    def forward(self, input, input_t, Hidden_State, Cell_State, rHidden_State, rCell_State, pred):

        # GC-LSTM
        x = input

        assert not torch.isnan(x).any(), 'Input is nan!'

        gc = self.gc_list[0](x)
        for i in range(1, self.K):
            assert not torch.isnan(gc).any(), str(i-1)
            gc = torch.cat((gc, self.gc_list[i](x)), 1)

        assert not torch.isnan(gc).any()

        combined = torch.cat((gc, Hidden_State, pred), 1)
        f = torch.sigmoid(self.fl(combined))
        i = torch.sigmoid(self.il(combined))
        o = torch.sigmoid(self.ol(combined))
        C = torch.tanh(self.Cl(combined))

        NC = torch.mul(Cell_State,
                       torch.mv(Variable(self.A_list[-1], requires_grad=False).to(DEVICE), self.Neighbor_weight))
        Cell_State = f * NC + i * C
        Hidden_State = o * torch.tanh(Cell_State)

        # LSTM
        rcombined = torch.cat((input_t, rHidden_State, pred), 1)
        rf = torch.sigmoid(self.rfl(rcombined))
        ri = torch.sigmoid(self.ril(rcombined))
        ro = torch.sigmoid(self.rol(rcombined))
        rC = torch.tanh(self.rCl(rcombined))
        rCell_State = rf * rCell_State + ri * rC
        rHidden_State = ro * torch.tanh(rCell_State)

        # Kalman Filtering
        var1, var2 = torch.var(input_t[:, :self.hidden_size]), torch.var(gc) # var of original input and input after GCN

        # weight GCN by var2, weight LSTM by var1 to remove the affect of vars
        pred = (Hidden_State * var1 * self.c + rHidden_State[:, :self.hidden_size] * var2) / (var1 + var2 * self.c)

        Hidden_State, rHidden_State = Hidden_State - pred, torch.cat([rHidden_State[:, :self.hidden_size] - pred, rHidden_State[:, self.hidden_size:]], dim=1)

        return Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred

    def Bi_torch(self, a):
        a[a < 0] = 0
        a[a > 0] = 1
        return a

    def loop(self, inputs):
        # input: torch.Size([64, 10, 207])
        batch_size = inputs.size(0)

        time_step = inputs.size(1)
        Hidden_State, Cell_State, rHidden_State, rCell_State = self.initHidden(batch_size)

        inputs_t = self.tconv(inputs.permute(0, 2, 1).unsqueeze(1)) # batch_size*time*detector -> batch_size*detector*time
        inputs_t = inputs_t.squeeze(1).permute(0, 2, 1)

        pred = Hidden_State.clone() # init Kalman filtering result
        for i in range(time_step):
            Hidden_State, Cell_State, gc, rHidden_State, rCell_State, pred = self.forward(
                torch.squeeze(inputs[:, i:i + 1, :self.hidden_size]), torch.squeeze(inputs_t[:, i:i + 1, :]),
                Hidden_State, Cell_State, rHidden_State, rCell_State, pred)

        pred = torch.cat([pred, rHidden_State[:, self.hidden_size:]], dim=1)
        pred = self.fc_mo(pred).unsqueeze(1)

        return pred

    def initHidden(self, batch_size):
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        rHidden_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        rCell_State = Variable(torch.zeros(batch_size, self.hidden_size).to(DEVICE))
        return Hidden_State, Cell_State, rHidden_State, rCell_State

    def reinitHidden(self, batch_size, Hidden_State_data, Cell_State_data):
        Hidden_State = Variable(Hidden_State_data.to(DEVICE), requires_grad=True)
        Cell_State = Variable(Cell_State_data.to(DEVICE), requires_grad=True)
        rHidden_State = Variable(Hidden_State_data.to(DEVICE), requires_grad=True)
        rCell_State = Variable(Cell_State_data.to(DEVICE), requires_grad=True)
        return Hidden_State, Cell_State, rHidden_State, rCell_State
