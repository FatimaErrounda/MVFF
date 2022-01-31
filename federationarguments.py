from datetime import datetime
class arguments():
  def __init__(self, dataset, x, y):
    self.dataset = "bikeNYC"
    # self.dataset = "Yelp"
    self.input_length = 6
    self.output_length = 1
    self.local_model_loss = "MAE"
    self.global_model_loss = "MAE"
    
    self.trainRatioEnd = 0.80
    self.trainRatioBegin = 0
    
    self.predictionSampleRatio = 0.1
    
    self.testRatioBegin = 0.89
    self.testRatioEnd = 0.99
    
    self.x_axis = x
    self.y_axis = y
    self.local_model = 'GRU'
    self.global_model = 'CNN'
    # self.local_model = 'FedLSTM'
    # self.global_model = 'FedLSTM'

    self.epochs = 1
    self.scratch_prediction = True
    self.test_normalization = False
    self.train_normalization = True
    self.optimizer = 'adam'
    self.trainingInterval = 24*60*60*1000
    self.batch_size = 3

  def setDevice(self,devicename):
    self.device_name = devicename
    