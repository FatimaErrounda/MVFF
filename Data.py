#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math 
import random
from sys import exit
from Models import LSTM, GRU, CNN
from collections import defaultdict
import datetime

class LocalSequentialDataset:
  def __init__(self, timestamps, x_axis, y_axis, begin, end, args):
    self.timestamps = timestamps
    self.x_axis = x_axis
    self.y_axis = y_axis
    self.begin = begin
    self.end = end
    self.args = args
    self.sequence = []
    self.sequence_timestamps = []
    self.overall_timestamps = []

  def make_data(self):  
    total_sample_length = self.args.input_length + self.args.output_length
    endpartitionIndex = int(self.end*(len(self.timestamps)-1))
    endpartitionIndex = int(endpartitionIndex/total_sample_length)
    endpartitionIndex *= total_sample_length
    beginpartitionIndex = int(self.begin*(len(self.timestamps)-1))
    beginpartitionIndex = int(beginpartitionIndex/total_sample_length)
    beginpartitionIndex *= total_sample_length
    
    data = self.timestamps["stats"][beginpartitionIndex:endpartitionIndex]
    self.args.start_training_time = self.timestamps["timestamp"][beginpartitionIndex]
    self.args.start_ending_time = self.timestamps["timestamp"][endpartitionIndex]
    self.overall_timestamps = self.timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values
    data_timestamps = torch.LongTensor(self.timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values)
    data_stats = torch.Tensor(data.values)
    #print("train_data_stats[:2]",train_data_stats[:2])
    if self.args.train_normalization == True:
      # scaler = MinMaxScaler(feature_range=(-5, 5))
      scaler = MinMaxScaler(feature_range=(-1, 1))
      data_normalized = scaler.fit_transform(data_stats.reshape(-1, 1))
      data_normalized = torch.FloatTensor(data_normalized).view(-1)
    else:
      data_normalized = torch.FloatTensor(data_stats).view(-1)
    L = len(data_normalized)
    for i in range(L-total_sample_length):
        seq = data_normalized[i:i+self.args.input_length]
        label = data_normalized[i+self.args.input_length:i+total_sample_length]
        seq_timestamp = data_timestamps[i:i+self.args.input_length]
        label_timestamp = data_timestamps[i+self.args.input_length:i+total_sample_length]
        self.sequence.append((seq,label))
        self.sequence_timestamps.append((seq_timestamp,label_timestamp))
        #self.train_sequence.append(torch.cat(train_seq ,train_label))                          
        
class LocalSampledDataset:
  def __init__(self, timestamps,x_axis, y_axis, args):
    self.timestamps = timestamps
    self.args = args
    self.x_axis = x_axis
    self.y_axis = y_axis
    self.sequence = []
    self.sequence_timestamps = []

  def make_data(self, sampled_list):
    #print("making data for the cell ",self.x_axis,",",self.y_axis)  
    total_sample_length = self.args.input_length + self.args.output_length
    datas = self.timestamps["stats"]
    #print(" test_timestamps length is ",len(datas))
    # if(len(datas) != 1485):
    #   print(" a problem occured when reading data for (",self.x_axis,",",self.y_axis)
    datas = torch.Tensor(datas.values)
    if self.args.train_normalization == True:
      # scaler = MinMaxScaler(feature_range=(-5, 5))
      scaler = MinMaxScaler(feature_range=(-1, 1))
      datas = scaler.fit_transform(datas.reshape(-1, 1))
      datas = torch.FloatTensor(datas).view(-1)

    #print("an example of statistics ", datas[0])
    data_timestamps = torch.LongTensor(self.timestamps["timestamp"].values)
    #print("an example of timestamps ", data_timestamps[0])
    #print(" test_data length is ",len(test_data))
    for i in sampled_list:
      seq = datas[i:i+self.args.input_length]
      label = datas[i+self.args.input_length:i+total_sample_length]
      seq_timestamp = data_timestamps[i:i+self.args.input_length]
      label_timestamp = data_timestamps[i+self.args.input_length:i+total_sample_length]
      self.sequence.append((seq,label))
      self.sequence_timestamps.append((seq_timestamp,label_timestamp))   
    #print("the total length of sample sequences is ",len(self.sequence))
    # print("an example of data is ",self.sequence[0])  
    # print("an example of timestamp is is ",self.sequence_timestamps[0])                                      

class GlobalSequentialDataset:
  def __init__(self, args):
    self.args = args
    self.sequence = []
    self.sequence_timestamps = []
    self.seqDict = {}
    self.labelDict = {}
    self.raw_stats_dict = defaultdict(dict)
    self.raw_timestamps_dict= defaultdict(dict)
    
    
  def add_data(self, grid_id, timestamps): 
    grid = grid_id.split(".")
    axis = grid[0].split("X")
    x_axis = int(axis[0])
    y_axis = int(axis[1])

    total_sample_length = self.args.input_length + self.args.output_length

    endpartitionIndex = int(self.args.testRatioEnd *(len(timestamps)-1))
    endpartitionIndex = int(endpartitionIndex/total_sample_length)
    endpartitionIndex *= total_sample_length
    beginpartitionIndex = int(self.args.testRatioBegin*(len(timestamps)-1))
    beginpartitionIndex = int(beginpartitionIndex/total_sample_length)
    beginpartitionIndex *= total_sample_length
    
    data = timestamps["stats"][beginpartitionIndex:endpartitionIndex]
    data_timestamps = torch.LongTensor(timestamps["timestamp"][beginpartitionIndex:endpartitionIndex].values)
    
    starting_timestamp = timestamps["timestamp"][beginpartitionIndex]
    self.args.Start_test_timestamp = starting_timestamp
    self.args.Start_test_timestamp /= 1000
    self.args.End_test_timestamp = timestamps["timestamp"][endpartitionIndex]
    self.args.End_test_timestamp /= 1000

    self.raw_stats_dict[x_axis][y_axis] = []
    self.raw_stats_dict[x_axis][y_axis] = np.full(endpartitionIndex-beginpartitionIndex, 0.) 
    self.raw_stats_dict[x_axis][y_axis] = data
    
    self.raw_timestamps_dict[x_axis][y_axis] = []
    self.raw_timestamps_dict[x_axis][y_axis] = np.full(endpartitionIndex-beginpartitionIndex, 0.) 
    self.raw_timestamps_dict[x_axis][y_axis] = timestamps["timestamp"][beginpartitionIndex:endpartitionIndex]
    self.raw_timestamps_dict[x_axis][y_axis][:] = [x - starting_timestamp for x in self.raw_timestamps_dict[x_axis][y_axis]]

    data_stats = torch.Tensor(data.values)
    #print("train_data_stats[:2]",train_data_stats[:2])
    if self.args.test_normalization == True:
      # scaler = MinMaxScaler(feature_range=(-5, 5))
      scaler = MinMaxScaler(feature_range=(-1, 1))
      data_normalized = scaler.fit_transform(data_stats.reshape(-1, 1))
      data_normalized = torch.FloatTensor(data_normalized).view(-1)
    else:
      data_normalized = torch.FloatTensor(data_stats).view(-1)
    L = len(data_normalized)
    for i in range(L-total_sample_length):
      seq = data_normalized[i:i+self.args.input_length]
      label = data_normalized[i+self.args.input_length:i+total_sample_length]
      seq_timestamp = data_timestamps[i:i+self.args.input_length]
      # label_timestamp = data_timestamps[i+self.args.input_length:i+total_sample_length]

      if seq_timestamp[0].item() in self.seqDict:
        seq_maps = self.seqDict.get(seq_timestamp[0].item())
        k=0
        for s in seq:
          seq_maps[k][x_axis][y_axis]= s
          k += 1
        item = {seq_timestamp[0].item(): seq_maps}
        self.seqDict.update(item)
        # print("adding sequence maps of size ", len(seq_maps))
        label_maps = self.labelDict.get(seq_timestamp[0].item())
        
        #just because for now I have outputs of 1 only
        j=0 
        for l in label:
          label_maps[j][x_axis][y_axis]= l
        item = {seq_timestamp[0].item(): label_maps}
        self.labelDict.update(item)
      else:
        seq_maps = [[[]]]
        seq_maps = np.full((self.args.input_length,self.args.x_axis,self.args.y_axis), 0.) 
        i=0
        for s in seq:
          seq_maps[i][x_axis][y_axis]= s
          i += 1
        # print("adding timestamp ",seq_timestamp[0], " to sequence dictionary")  
        self.seqDict[seq_timestamp[0].item()] = seq_maps
        seq_timestamps = [[[]]]
        seq_timestamps = np.full((self.args.output_length,self.args.x_axis,self.args.y_axis), 0.) 
        j=0
        for l in label:
          seq_timestamps[j][x_axis][y_axis]= l
        self.labelDict[seq_timestamp[0].item()] = seq_timestamps   
  
  def make_data(self):
    # print("size of seqDict", len(self.seqDict))
    # print("size of labelDict", len(self.labelDict))
    for t,seq in self.seqDict.items():
      # print("t is", t)
      # print("seq is", seq)
      seq = torch.FloatTensor(seq)
      seq = seq.unsqueeze(0) 
      # seq = torch.FloatTensor(seq)
      label = self.labelDict.get(t)
      label = torch.FloatTensor(label)
      #print(seq.shape)
      # print("label is", label)
      self.sequence.append((seq,label)) 
    
  def check_data(self):
    print("size of seqDict", len(self.seqDict))
    print("size of labelDict", len(self.labelDict))
    if(len(self.seqDict) != len(self.labelDict)):
      print("error in building the data")
    # print("start timestamp of global sequential",self.args.Start_test_timestamp)
    # print("end timestamp of global sequential",self.args.End_test_timestamp)
    print("start timestamp of global sequential",datetime.datetime.fromtimestamp(self.args.Start_test_timestamp).strftime('%Y-%m-%d %H:%M:%S'))
    print("end timestamp of global sequential",datetime.datetime.fromtimestamp(self.args.End_test_timestamp).strftime('%Y-%m-%d %H:%M:%S'))
    
    # if(len(self.raw_stats[:][:]) != self.args.x_axis  * self.args.y_axis):
    #   print("error in building the raw data")
    # print(self.raw_stats.shape)
    #print("size of labelDict", len(self.sequence))
    #print(self.sequence[:2])

class GlobalSampledDataset:
  def __init__(self, args):
    self.args = args
    self.sequence = []
    #self.sequence = np.zeros((args.bike_x,args.bike_y))
    #self.sequence_timestamps = []
    self.seqDict = {}
    self.labelDict = {}
    self.total_sample_length = self.args.input_length + self.args.output_length

  def add_data(self, grid_id, grid_dataset):
    grid = grid_id.split(".")
    axis = grid[0].split("X")
    x_axis = int(axis[0])
    y_axis = int(axis[1])

    #print("adding data from x", x_axis, " and y ", y_axis)
    #print("the total length of sample sequences is ",len(grid_dataset.sequence))
    for data, timestamps in zip(grid_dataset.sequence, grid_dataset.sequence_timestamps):
      seq,label = data
      seq_timestamp, label_timestamp = timestamps  
      if seq_timestamp[0].item() in self.seqDict:
        seq_maps = self.seqDict.get(seq_timestamp[0].item())
        # for s,t in zip(seq,seq_timestamp):
        #   seq_maps[t][x_axis][y_axis]= s
        i=0
        for s in seq:
          seq_maps[i][x_axis][y_axis]= s
          i += 1
        item = {seq_timestamp[0].item(): seq_maps}
        self.seqDict.update(item)
        #print("adding sequence maps of size ", len(seq_maps))
        label_maps = self.labelDict.get(seq_timestamp[0].item())
        j=0
        for l in label:
          label_maps[j][x_axis][y_axis]= l
        item = {seq_timestamp[0].item(): label_maps}
        self.labelDict.update(item)
        #print("adding label maps of size ", len(label_maps))
      else:
        seq_maps = [[[]]]
        seq_maps = np.full((self.args.input_length,self.args.x_axis,self.args.y_axis), 0.) 
        i=0
        for s in seq:
          seq_maps[i][x_axis][y_axis]= s
          i += 1
        #print("adding timestamp ",seq_timestamp[0], " to sequence dictionary")  
        self.seqDict[seq_timestamp[0].item()] = seq_maps
        seq_timestamps = [[[]]]
        seq_timestamps = np.full((self.args.output_length,self.args.x_axis,self.args.y_axis), 0.) 
        j=0
        for l in label:
          seq_timestamps[j][x_axis][y_axis]= l.item()
        self.labelDict[seq_timestamp[0].item()] = seq_timestamps                    
    # self.sequence[x_axis][y_axis] = grid_dataset.sequence
    # self.sequence_timestamps[x_axis][y_axis] = grid_dataset.sequence_timestamps
  
  def make_data(self):
    # print("size of seqDict", len(self.seqDict))
    # print("size of labelDict", len(self.labelDict))
    for t,seq in self.seqDict.items():
      # print("t is", t)
      # print("seq is", seq)
      seq = torch.FloatTensor(seq)
      seq = seq.unsqueeze(0)
      # print(seq.shape) 
      label = self.labelDict.get(t)
      seq = torch.FloatTensor(seq)
      label = torch.FloatTensor(label)
      #print(seq.shape)
      # print("label is", label)
      self.sequence.append((seq,label))

  def check_data(self):
    print("GlobalSampledDataset")
    print("size of seqDict", len(self.sequence))
    if(len(self.seqDict) != len(self.labelDict)):
      print("error in building the data")
    # for t,seq in self.seqDict.items():

    # print("size of labelDict", len(self.sequence))
    #print(self.sequence[:2])

def GenerateRandomSamples(timestamps, args):
  total_sample_length = args.input_length + args.output_length
  samplenumbers = int(args.predictionSampleRatio*len(timestamps["stats"]))
  maxlength = len(timestamps["stats"])-total_sample_length
  sampled_list = random.sample(range(maxlength), samplenumbers)
  sampled_list= np.sort(sampled_list)
  #print(sampled_list)
  return sampled_list

def MakeTrainingTimes(LocalTrainData, args):
  seq_timestamp, label_timestamps = LocalTrainData[0].sequence_timestamps[0]
  args.beginTrainingTimestamp = seq_timestamp[0].item()
  seq_timestamp, label_timestamps = LocalTrainData[0].sequence_timestamps[len(LocalTrainData[0].sequence_timestamps)-1]
  args.endTrainingTimestamp = seq_timestamp[0].item()

def CheckLocalTestData(LocalTrainData, args):
  print("CheckLocalTestData")
  beginningTime = args.beginTrainingTimestamp
  endingTime = args.endTrainingTimestamp
  print("beginningTime=",beginningTime)
  print("endingTime=",endingTime)
  print("args.trainingInterval=",args.trainingInterval)
  for dataset in LocalTrainData:
    for timestamp in range(beginningTime, endingTime, args.trainingInterval):
      found_timestamp = False
      index_timestamp = -1
      i = 0
      for seq_timestamp, label_timestamps in dataset.sequence_timestamps:
        if timestamp >= seq_timestamp[0].item() and timestamp <= seq_timestamp[len(seq_timestamp)-1].item():
          found_timestamp = True
          index_timestamp = i
        i += 1

      if index_timestamp == -1:
        print("a problem finding the timestamp in the training data for dataset index ",dataset.x_axis,",", dataset.y_axis)

def CheckLocalPredictionData(datasets,samples):
  print("CheckLocalPredictionData")
  
  if "0X0" not in datasets:
    print("a problem in the dictionary of prediction datasets")
    exit("CheckLocalPredictionData: datasets[0X0] is not in the datasets")
  
  print("size of seqDict", len(datasets["0X0"].sequence))

  reference_timestamps = []
  for t in datasets["0X0"].sequence_timestamps:
    seq_t, label_t = t
    reference_timestamps.append(seq_t[0])
  totalsamples = len(datasets["0X0"].sequence_timestamps)
  for data in datasets.values():
    if totalsamples != len(data.sequence_timestamps):
      print("a problem in the timestamps length of the local prediction data (", data.x_axis,",",data.y_axis)
    for t in data.sequence_timestamps: 
      seq_t,label_t = t
      if not seq_t[0] in reference_timestamps:
        print("a problem in the timestamps of the local prediction data(", data.x_axis,",",data.y_axis)
        return False
  return True

def generate_local_prediction(weight_pair, LocalPredictionSample, args):
  if args.scratch_prediction == True:
    if args.local_model == 'LSTM':
      weight_pair.model.to(args.device_name)  
      predictionModel = LSTM(args)
      predictionModel.load_state_dict(weight_pair.model.state_dict())
      predictionModel.to(args.device_name)
      localpredictionresult = predictionModel.predict(LocalPredictionSample,args)
      return localpredictionresult
    else:
      if args.local_model == 'GRU':
        weight_pair.model.to(args.device_name)
        predictionModel = GRU(args)
        predictionModel.load_state_dict(weight_pair.model.state_dict())
        predictionModel.to(args.device_name)
        localpredictionresult = predictionModel.predict(LocalPredictionSample,args)
        return localpredictionresult
      else:
        raise SystemError('Unrecognized local model')
  else:
    weight_pair.model.to(args.device_name)
    localpredictionresult = weight_pair.model.predict(LocalPredictionSample,args)
  return localpredictionresult

def build_adjMatrix(args):
  dimension = args.x_axis*args.y_axis
  adj = torch.zeros(dimension,dimension)

  for i,j in zip(range(args.x_axis),range(args.y_axis)):
    #build matrix
    ij_index = j * args.y_axis + i
    i = i+1
    j = j+1
    matrix_adj = torch.zeros(args.x_axis+2,args.y_axis+2)
    matrix_adj[i-1][j-1] = 1
    matrix_adj[i-1][j] = 1
    matrix_adj[i-1][j+1] = 1
    matrix_adj[i][j-1] = 1
    matrix_adj[i][j+1] = 1
    matrix_adj[i+1][j-1] = 1
    matrix_adj[i+1][j] = 1
    matrix_adj[i+1][j+1] = 1
    matrix_adj = matrix_adj[1:,:]
    matrix_adj = matrix_adj[:-1,:]
    matrix_adj = matrix_adj[:,1:]
    matrix_adj = matrix_adj[:,:-1]
    indeces = (matrix_adj == 1.0).nonzero()
    for index in indeces:
      x,y = index
      index = y * args.x_axis + x
      adj[ij_index][index] = 1
      # adj[index][ij_index] = 1
  return adj

def build_map(localpredictionresults, args):
  totalNumberOfSequences = len(localpredictionresults["0X0"])
  seq_predictions = [[[[]]]]
  seq_predictions = np.full((totalNumberOfSequences,args.output_length,args.x_axis,args.y_axis), 0.) 
  for index, localpredictions in localpredictionresults.items():
    axis = index.split("X")
    x_axis = int(axis[0])
    y_axis = int(axis[1])
    # seq_predictions = [[[]]]
    # seq_predictions = np.full((args.output_length,args.x_axis,args.y_axis), 0.) 
    i = 0;
    for prediction in localpredictions:
      seq_predictions[i][0][x_axis][y_axis]= prediction
      i+=1
  seq_predictions = torch.Tensor(seq_predictions)
  # print(seq_predictions.shape)
  return seq_predictions
