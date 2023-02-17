import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from DDGCN.model_import import parse_cfg
from DDGCN.gcn import ConvGraphical, Graph
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np



class DDGCN(nn.Module):
       
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()


        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)


        sk_size = A.size(0)
        tk_size = 9
        kernel_size = (tk_size, sk_size)
        self.data_bn = nn.BatchNorm1d(
            in_channels * A.size(1)) if data_bn else lambda x: x
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(
                in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 2, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 2, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))


        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)


        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        
        
    def forward(self, x):



        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            

        x = F.avg_pool2d(x, x.size()[2:])

        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = self.fcn(x)

        x = x.view(x.size(0), -1)


        return x

    def extract_feature(self, x):


        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

 
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)


        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature
    
    def load_graph(graph_cfg, self):
    
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(
            self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        

class st_gcn_block(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1), 
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.count = 0
        self.W_F = []
   
    def forward(self, x, A):
        #with torch.no_grad():     
          #self.w = self.gcn.conv.weight.cpu().detach().numpy() #remove self
          #x = x.cpu().detach().numpy()
            
    ### Dynamic Nodes Update for DCS ALGORTHM ########################
          #A = DCS(A, self)
      
    # ###### DCW ALGORTHM ########################  
          #x = DCW(x, self.w, self)
    
    # ###### DSTG ALGORTHM ########################  
          #x = DSTG(x, A, self)
                
          #A = torch.from_numpy(A).cuda()
          #x = torch.from_numpy(x).cuda()
           
          res = self.residual(x)
          x, A = self.gcn(x, A)
          x = self.tcn(x) + res
          
          #np.save('A4DD', A.cpu().detach().numpy())
             
          self.count += 1
          return self.relu(x), A
          #torch.cuda.empty_cache()

def DCS(A, self):
    
    
    
    A = A.cpu().detach().numpy()
    T_f = A[1].copy()
    #print (self.count)
    
    if (self.count == 0):
        self.W_F.append(np.random.rand(T_f.shape[0], T_f.shape[1]))
    else:
        W_F = self.W_F
        w_f = W_F[-1]

        L_f = []
        for i in range (0, T_f.shape[0]):
            L_f.append(np.nonzero(T_f[i]))
        L_f = np.array(L_f).squeeze()
        score = np.load('score.npy')  
        #print (score)
        grad = T_f.T.dot(1.0/score) / (T_f.shape[0])
        w_f = w_f - 0.00001 * grad
        self.W_F.append(w_f)

        O_f = 1 / (1 + np.exp(-T_f.dot(w_f)))
        
        for i in range (0, T_f.shape[0]):
              node_ind = np.argpartition(np.abs(O_f[i]), -L_f[i].shape[0])[-L_f[i].shape[0]:]
              if (L_f[i].shape[0] != 0):
                   A[1][i][np.sort(L_f[i])] =  0
                   A[1][i][np.sort(node_ind)] = T_f[i][np.sort(L_f[i])]
    return A
        

def DCW(x, w, self):
    
   
    for i in range (0, self.w.shape[2]):
        for j in range (0,  self.w.shape[3]):
                distance, path = fastdtw(self.w[:,:, i, j], x[:,:, i, j], dist=euclidean)
                path1 = np.array(path)[:,1]
        x[path1] = x[i,:, :, :]
    return x



def DSTG(x, A):
    #print (x.shape)
    e_ntu_parents = [1, 20, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0, 12, 13, 14, 0, 16, 17, 18, 20, 22, 7, 24, 11]
    
    x_b = x.permute(0, 2, 3, 1).contiguous() #BxTxJxF
    #print (x_b.shape)
    
    # e_kin = [(4, 3), (3, 2), (7, 6), (6, 5),
    #          (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
    #          (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
    #          (17, 15), (16, 14)]

    
    if (A.shape[1] == 25):
        for i in range(0, x_b.shape[0]):
            #print (i)
            for j in range(0, x_b.shape[1]):
                for k in range(0, x_b.shape[2]):
                    x_b[i, j, k,:] = x_b[i, j, k,:] - x_b[i, j, e_ntu_parents[k],:]

    # else:
    #     for i in range(0, 2):
    #         for j in range(0, 3):
    #             x[:,:,i,j] = x[:,:,i,j] - x[:,:,i,(e_kin[j][1]-1)]
    #         for j in range(3, 17):
    #             x[:,:,i,j+1] = x[:,:,i,j+1] - x[:,:,i,(e_kin[j][1]-1)]
    return x_b.permute(0, 3, 1, 2).contiguous()

        


   
