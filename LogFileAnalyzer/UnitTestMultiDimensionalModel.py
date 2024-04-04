import unittest
import numpy as np
import MultiDimensionalModel
import tensorflow as tf

class UnitTestMultiDimensionalModel(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.datatype        = tf.float64
        self.sequence_length = 16
        self.rnn_units       = 64
        self.num_samples     = 10
        self.numberOfInferences = 201
        self.vocab_size         = 1001

    def test_Model(self):
        print("TestModel test")
        self.featureList = []

        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100, "layer_name": "Categorical_layer"})
        self.featureList.append({"embedded": False, "layer_name": "Non embedding"})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples))
        return_sequences = False
        dense_layer      = False


        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_sequences = return_sequences, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInferences)
        print("TestModel test DONE")
        self.assertEqual(prediction.shape, (self.numberOfInferences, self.rnn_units))  

    def test_ModelOutputSequences(self):
        print("TestModelOutputSequences test")
        self.featureList = []
        

        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100, "layer_name":  "Categorical_layer"})
        self.featureList.append({"embedded": False, "layer_name": "Non embedding"})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples, self.sequence_length, self.rnn_units))
        return_sequences = True
        dense_layer      = False
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_sequences = return_sequences, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInferences)
        print("TestModelOutputSequences test DONE")
        self.assertEqual(prediction.shape, (self.numberOfInferences, self.sequence_length, self.rnn_units))  

    def test_ModelDense(self):
        print("TestModelDense test")
        self.featureList = []
        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
        self.featureList.append({"embedded": False, "layer_name": "Non embedding"})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples, self.featureList[0]["vocab_size"] ))
        return_sequences = False
        dense_layer      = True
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_sequences = return_sequences, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInferences)
        print("TestModelDense test DONE")
        self.assertEqual(prediction.shape, (self.numberOfInferences, self.featureList[0]["vocab_size"]))  

    def test_ModelDenseAndSequences(self):
        print("TestModelDenseAndSequences test")
        self.featureList = []
        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
        self.featureList.append({"embedded": False, "layer_name": "Non embedding"})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples, self.sequence_length, self.featureList[0]["vocab_size"] ))
        return_sequences = True
        dense_layer      = True
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_sequences = return_sequences, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInferences)
        
        self.assertEqual(prediction.shape, (self.numberOfInferences, self.sequence_length, self.featureList[0]["vocab_size"]))  
        print("TestModelDenseAndSequences test DONE")

    def Inference(self, model, numberOfInterences):
        embedded_input = tf.ones((numberOfInterences, self.sequence_length))
        direct_lstm_input = tf.ones((numberOfInterences, self.sequence_length))
        datasetInputData_tensor = tf.stack((embedded_input, direct_lstm_input), axis = -1)
        datasetInputData_tensor = tf.cast(datasetInputData_tensor, 'float32')
        modelPrediction = model(datasetInputData_tensor, self.numberOfInferences)
        return modelPrediction

    def CreateDataSet(self, inputTuple, output):
        outputItemTensor = tf.convert_to_tensor(output)
        inputTensorTuple = tf.convert_to_tensor(inputTuple)

        datasetInputData_tensor = tf.stack(inputTensorTuple, axis = -1)
        dataset_tensor = tf.data.Dataset.from_tensor_slices((datasetInputData_tensor, outputItemTensor))

        dataset = dataset_tensor.batch(3, drop_remainder = True)
        return dataset

if __name__ == '__main__':
    unittest.main()
