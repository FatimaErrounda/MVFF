import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
import copy


class Pair(object):
  def __init__(self,x_axis, y_axis,model):
    self.x_axis = x_axis
    self.y_axis = y_axis
    self.model = model

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def update_global_weights(model, GlobalSequences, GlobalLabels,args,adj):
  if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                       momentum=0.5)
  elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  #build the sequence with labels
  input_Sequence = []
  i = 0
  for item in GlobalSequences.sequence:
    seq, label1 = item
    input_Sequence.append((seq,GlobalLabels[i]))
  model.to(args.device_name)
  _, acc, loss = model.fwd_pass(input_Sequence,adj,optimizer,True) 
  return acc, loss 

def update_weights(timestamp, dataset, model, args):
  if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.5)
  elif args.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  
  found_timestamp = False
  index_timestamp = -1
  n = len(dataset.sequence_timestamps)
  first = dataset.sequence_timestamps[0]
  last = dataset.sequence_timestamps[n-1]
  
  index_timestamp = (timestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_timestamp = int(index_timestamp.item())

  endtimestamp = timestamp+args.trainingInterval
  index_end = (endtimestamp-first[0][0])/(last[0][0]-first[0][0])*n
  index_end = int(index_end.item())

  if index_timestamp == -1:
    print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)
  model.to(args.device_name)
  acc, loss = model.fwd_pass(dataset.sequence[index_timestamp:index_end],optimizer,True)
  return acc, loss


def test_model(model, adj, dataset,args):
  return model.fwd_pass(dataset.sequence,adj,args.optimizer, False)

def fed_test_model(model, dataset,args):
  acc = {}
  losses = []
  n = len(dataset.sequence)
  output = [[[]]]
  output = np.full((n,args.x_axis,args.y_axis), 0.) 
  with torch.no_grad():
    k = 0
    for seq, label in dataset.sequence:
      dims = seq.shape
      seq = seq.squeeze(0)

      prediction = [[]]
      prediction = np.full((args.x_axis,args.y_axis), 0.) 
      for i in range(dims[1]):
        for j in range(dims[2]):
          pred = model.forward(seq[:,i,j])
          prediction[i][j] = pred
          output[k][i][j] = pred
      if "MAE" in acc:
          acc["MAE"] += abs(label - prediction)
      else: 
        acc["MAE"] = abs(label - prediction)
      if "MSE" in acc:
        acc["MSE"] += (label - prediction) * (label - prediction) 
      else:
        acc["MSE"] = (label - prediction) * (label - prediction) 
      if "AE" in acc:
        acc["AE"] += label - prediction
      else:
        acc["AE"] = label - prediction
      if "WMAPE" in acc:
        acc["WMAPE"] += abs(label)
      else:
        acc["WMAPE"] = abs(label)
      k+=1

  acc_WMAPE = float((acc["MAE"]/acc["WMAPE"]).nanmean())
    
  acc["MAE"] /= n
  acc["MSE"] /= n
  acc["AE"] /= n

  acc_MAE = float(acc["MAE"].mean())
  acc_MSE = float(acc["MSE"].mean())
  acc_AE = float(acc["AE"].mean())
  acc_RMSE = math.sqrt(acc_MSE)

  acc_floats = {}
  acc_floats["MAE"] = acc_MAE
  acc_floats["MSE"] = acc_MSE
  acc_floats["RMSE"] = acc_RMSE
  acc_floats["AE"] = acc_AE
  acc_floats["WMAPE"] = acc_WMAPE
  loss = 0.0
  return output, acc_floats, loss