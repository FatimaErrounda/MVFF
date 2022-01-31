import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
import torch.optim as optim
import math
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.input_size = args.input_length
    self.output_size= args.output_length
    self.hidden_layer_size = 2*args.input_length
    self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size)

    self.fc = nn.Linear(self.hidden_layer_size, self.output_size)
    self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name).requires_grad_()

    self.hidden_state = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name).requires_grad_()

    if(args.local_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.local_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')
    for param in self.parameters():
      param.grad = None

  def fwd_pass(self, input_Sequence, optimizer, train=False):

    for seq, labels in input_Sequence:
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      if train:
          optimizer.zero_grad()
          self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name)

          self.hidden_state = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name)

      y_pred = self.forward(seq)

      single_loss = self.loss_function(y_pred, labels)
      if train:
        single_loss.backward()
        optimizer.step()

    # return acc_floats,loss
  
  def predict(self,LocalPredictionSamples,args):
    input_Sequence = LocalPredictionSamples.sequence
    predictions = []
    n = len(input_Sequence)
    for seq, labels in input_Sequence:
      seq = seq.to(self.args.device_name)
      y_pred = self.forward(seq)
      predictions.append(y_pred)
    return predictions
  
  def forward(self, input_seq):
    input_seq = input_seq.unsqueeze(0)
    input_seq = input_seq.unsqueeze(0)
    lstm_out, (self.hidden_cell,self.hidden_state) = self.lstm(input_seq, (self.hidden_state.detach(),self.hidden_cell.detach()))
    lstm_out = lstm_out.to(self.args.device_name)
    predictions = self.fc(lstm_out.view(len(lstm_out), -1))
    return predictions[-1]

class CNN(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    outChannels = self.args.input_length
    self.main = nn.Sequential(
       nn.Conv3d(1, 1, kernel_size=(self.args.input_length,3,3), stride=1, padding=(0,1,1))
    if(args.global_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.global_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')
    for param in self.parameters():
      param.grad = None

  def fwd_pass(self, input_Sequence, adj, optimizer, train=False):
    for seq, labels in input_Sequence:
      seq = seq.unsqueeze(0)
      seq = seq.to(self.args.device_name)
      labels = labels.squeeze()
      labels = labels.to(self.args.device_name)
      
      if train:
        optimizer.zero_grad()

      y_pred = self.forward(seq)
      
      y_pred = y_pred.squeeze()
      single_loss = self.loss_function(y_pred, labels)
      losses.append(single_loss.item())
    
    # return output, acc_floats, loss

  def forward(self, input_seq):
    out = self.main(input_seq)
    return out

class GRU(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.input_size = args.input_length
    self.output_size= args.output_length
    self.hidden_layer_size = 2*args.input_length
    self.gru = nn.GRU(self.input_size, self.hidden_layer_size, batch_first=False)
    self.linear = nn.Linear(self.hidden_layer_size, self.output_size)
    self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size)
    if(args.local_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.local_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')
    for param in self.parameters():
      param.grad = None
        
  
  def fwd_pass(self, input_Sequence, optimizer, train=False):
    for seq, labels in input_Sequence:
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      if train:
        optimizer.zero_grad()
        self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size).to(self.args.device_name)
      
      y_pred = self.forward(seq)
      
      single_loss = self.loss_function(y_pred, labels)
      if train:
        single_loss.backward()
        optimizer.step()
    # return acc_floats, loss

  def forward(self, input_seq):
    input_seq = input_seq.unsqueeze(0)
    gru_out, self.hidden_cell = self.gru(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
    predictions = self.linear(gru_out)
    predictions = predictions.squeeze(0)
    return predictions[-1]

  def predict(self,LocalPredictionSamples,args):
    input_Sequence = LocalPredictionSamples.sequence
    predictions = []
    n = len(input_Sequence)
    for seq, labels in input_Sequence:
      seq = seq.to(self.args.device_name)
      y_pred = self.forward(seq)
      predictions.append(y_pred)
    return predictions

class FedGRU(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.input_size = args.input_length
    self.output_size= args.output_length
    self.hidden_layer_size = 2* args.input_length
    self.gru = nn.GRU(self.input_size, self.hidden_layer_size, batch_first=False)
    self.linear = nn.Linear(self.hidden_layer_size, self.output_size)
    self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size)

    if(args.local_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.local_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')
    for param in self.parameters():
      param.grad = None
  
  def fwd_pass(self, input_Sequence,optimizer, train=False):
    for seq, labels in input_Sequence:
      seq = seq.unsqueeze(0)
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)

      if train:
        optimizer.zero_grad()
        self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size).to(self.args.device_name)

      y_pred = self.forward(seq)
      y_pred2 = y_pred.detach().clone()

      single_loss = self.loss_function(y_pred, labels)

      if train:
        single_loss.backward()
        optimizer.step()
    # return acc_floats, loss

  def forward(self, input_seq):
    gru_out, self.hidden_cell = self.gru(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
    predictions = self.linear(gru_out)
    predictions = predictions.squeeze(0)
    return predictions[-1]

class FedLSTM(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.input_size = args.input_length
    self.output_size= args.output_length
    self.hidden_layer_size = 2*args.input_length
    self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size)

    self.fc = nn.Linear(self.hidden_layer_size, self.output_size)
    self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name).requires_grad_()

    self.hidden_state = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name).requires_grad_()

    if(args.local_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.local_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')
    for param in self.parameters():
      param.grad = None

  def fwd_pass(self, input_Sequence,optimizer, train=False):
    for seq, labels in input_Sequence:
      seq = seq.to(self.args.device_name)
      labels = labels.to(self.args.device_name)
      if train:
        optimizer.zero_grad()
        self.hidden_cell = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name)

        self.hidden_state = torch.zeros(1,1,self.hidden_layer_size).to(self.args.device_name)
      y_pred = self.forward(seq)
      
      single_loss = self.loss_function(y_pred, labels)

      if train:
        single_loss.backward()
        optimizer.step()
    # return acc_floats, loss

  def forward(self, input_seq):
    input_seq = input_seq.unsqueeze(0)
    input_seq = input_seq.unsqueeze(0)
    lstm_out, (self.hidden_cell,self.hidden_state) = self.lstm(input_seq, (self.hidden_state.detach(),self.hidden_cell.detach()))
    lstm_out = lstm_out.to(self.args.device_name)
    predictions = self.fc(lstm_out.view(len(lstm_out), -1))
    return predictions[-1]

class MyGraphOutputLayer(nn.Module):
  def __init__(self,args, in_features):
    super(MyGraphOutputLayer, self).__init__()
    self.args = args
    self.W = nn.Parameter(torch.randn(in_features,args.x_axis, args.y_axis)).to(args.device_name)
    self.sigmoid = nn.Sigmoid()
    for param in self.parameters():
      param.grad = None
  
  def forward(self, h, att):
    adj_dimension = self.args.x_axis*self.args.y_axis
    e = torch.mul(self.W,h)
    e = torch.sum(e, dim=0)
    e = torch.reshape(e,(1,adj_dimension))
    e = e.expand(adj_dimension,adj_dimension)
    e = torch.mul(e,att)
    e = torch.sum(e, dim=0)
    return e.reshape(self.args.x_axis,self.args.y_axis)

class MyGraphAttentionLayer(nn.Module):
  def __init__(self, args, in_features,out_features, x_index, y_index, i_1,j_1,i_2,j_2, neighbour, dropout, alpha, concat=True):
    super(MyGraphAttentionLayer, self).__init__()
    self.dropout = dropout
    self.in_features = in_features
    self.x_index = x_index
    self.y_index = y_index
    self.i_1 = i_1
    self.j_1 = j_1
    self.i_2 = i_2
    self.j_2 = j_2
    self.neighbour = neighbour
    self.args = args
    if(i_1>=args.x_axis or i_2>=args.x_axis or j_1>=args.y_axis or j_2>=args.y_axis):
      print("a problem in creating the attention layer in ",i_1,j_1)
      print("a problem in creating the attention layer in ",i_2,j_2)
    self.alpha = alpha

    self.W = nn.Parameter(torch.randn(in_features).to(args.device_name))
    self.a = nn.Parameter(torch.randn(2*out_features).to(args.device_name))

    self.leakyrelu = nn.LeakyReLU(self.alpha)
    for param in self.parameters():
      param.grad = None

  def forward(self, h, adj):
    if self.neighbour:
      x1 = h[:,self.i_1,self.j_1]
      x2 = h[:,self.i_2,self.j_2]
      Wh1 = torch.matmul(x1, self.W) 
      Wh2 = torch.matmul(x2, self.W) 
      WH = torch.stack((Wh1,Wh2), dim=0)
      WH = torch.matmul(WH, self.a)
      return self.leakyrelu(WH)
    else:
      return torch.tensor(0.).to(self.args.device_name)

class MyGAT(nn.Module):
  def __init__(self,args, adj, dropout=0.6, alpha=0.2):
    super(MyGAT, self).__init__()
    self.dropout = dropout
    self.args = args
    
    if(args.global_model_loss == 'MSE'):
      self.loss_function = nn.MSELoss().to(self.args.device_name)
    else:
      if(args.global_model_loss == 'MAE'):
        self.loss_function = nn.L1Loss().to(self.args.device_name)
      else:
        raise SystemError('Need to specify a loss function')

    self.attentions = []
    self.totalattentions = args.x_axis * args.y_axis
    for x in range(self.totalattentions):
      for y in range(self.totalattentions):
        j_1 = x // self.args.x_axis
        i_1 = x % self.args.x_axis
        j_2 = y // self.args.x_axis
        i_2 = y % self.args.x_axis  
        self.attentions.append(MyGraphAttentionLayer(args, args.input_length,args.output_length, x,y, i_1,j_1, i_2, j_2, adj[x][y]==1, dropout=dropout, alpha=alpha))
    
    for i, attention in enumerate(self.attentions):
        self.add_module('attention_{}'.format(i), attention)

    self.out_layers = MyGraphOutputLayer(args, args.input_length)
    for param in self.parameters():
      param.grad = None

  def fwd_pass(self, input_Sequence, adj, optimizer, train=False):
    for seq, labels in input_Sequence:
      seq = seq.unsqueeze(0)
      seq = seq.to(self.args.device_name)
      labels = labels.squeeze()
      labels = labels.to(self.args.device_name)
      
      if train:
        optimizer.zero_grad()

      y_pred = self.forward(seq,adj)
      y_pred = y_pred.squeeze()
      single_loss = self.loss_function(y_pred, labels)
      if train:
        single_loss.backward()
        optimizer.step()
    # return output, acc_floats, loss

  def forward(self, x, adj):
    x = F.dropout(x, self.dropout, training=self.training)
    x=x.squeeze()
    attx = torch.stack([att(x, adj) for att in self.attentions], dim=0)
    attx = torch.reshape(attx,(self.totalattentions,self.totalattentions))
    zero_vec = 0.0*torch.ones_like(attx)
    #keep only the neighbours
    attx = torch.where(adj > 0, attx, zero_vec)
    #readjust the coefficient per neighbour
    attx = F.softmax(attx, dim=0)
    #calculate the next timestamp
    return self.out_layers(x,attx)
