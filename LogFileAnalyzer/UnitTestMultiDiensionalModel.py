import unittest
import numpy as np
import MultiDimensionalModel
import tensorflow as tf

class UnitTestMultiDiensionalModel(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.datatype        = tf.float64
        self.sequence_length = 16
        self.rnn_units       = 64
        self.num_samples     = 10
        self.numberOfInterences = 201
        
    def test_Model(self):
        print("TestModel test")
        self.featureList = []
        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
        self.featureList.append({"embedded": False})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples))
        return_states   = False
        dense_layer     = False
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_states = return_states, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInterences)
        print("TestModel test DONE")
        self.assertEqual(prediction.shape, (self.numberOfInterences, self.rnn_units))  # Test if 2 + (-7) equals -5

    def test_ModelOutputSequences(self):
        print("TestModelOutputSequences test")
        self.featureList = []
        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
        self.featureList.append({"embedded": False})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples, self.sequence_length, self.rnn_units))
        return_states   = True
        dense_layer     = False
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_states = return_states, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInterences)
        print("TestModelOutputSequences test DONE")
        self.assertEqual(prediction.shape, (self.numberOfInterences, self.sequence_length, self.rnn_units))  # Test if 2 + (-7) equals -5


    def test_ModelDense(self):
        print("TestModelDense test")
        self.featureList = []
        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
        self.featureList.append({"embedded": False})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples, self.featureList[0]["vocab_size"] ))
        return_states   = False
        dense_layer     = True
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_states = return_states, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInterences)
        print("TestModelDense test DONE")
        self.assertEqual(prediction.shape, (self.numberOfInterences, 1000))  

    def test_ModelDenseAndSequences(self):
        print("TestModelDenseAndSequences test")
        self.featureList = []
        self.featureList.append({"embedded": True, "vocab_size" : 1000, "embedding_dim": 100})
        self.featureList.append({"embedded": False})
        embedded_input_data    = np.random.randint(1000, 1000 + self.featureList[0]["vocab_size"], size=(self.num_samples, self.sequence_length))
        direct_lstm_input_data = np.random.randint(100, 105,                                       size=(self.num_samples, self.sequence_length ))
        regression_labels      = np.random.randint(0, 3,                                           size=(self.num_samples, self.sequence_length, self.featureList[0]["vocab_size"] ))
        return_states   = True
        dense_layer     = True
        model = MultiDimensionalModel.LSTMPrintfLogMultiModel(self.featureList, self.rnn_units, self.sequence_length, dense_layer = dense_layer, return_states = return_states, datatype = self.datatype)
        model.compile(optimizer='adam', loss='mse') 
        inputTuple = (embedded_input_data, direct_lstm_input_data)
        dataset = self.CreateDataSet(inputTuple=inputTuple, output=regression_labels)
        model.fit(dataset, epochs=5)
        prediction = self.Inference(model, self.numberOfInterences)
        
        self.assertEqual(prediction.shape, (self.numberOfInterences, self.sequence_length, 1000))  # Test if 2 + (-7) equals -5
        print("TestModelDenseAndSequences test DONE")

    def Inference(self, model, numberOfInterences):
        embedded_input = tf.ones((numberOfInterences, self.sequence_length))
        direct_lstm_input = tf.ones((numberOfInterences, self.sequence_length))
        datasetInputData_tensor = tf.stack((embedded_input, direct_lstm_input), axis = 1)
        modelPrediction = model(datasetInputData_tensor, self.numberOfInterences)
        return modelPrediction

    def CreateDataSet(self, inputTuple, output):
        outputItemTensor = tf.convert_to_tensor(output)
        inputTensorTuple = tf.convert_to_tensor(inputTuple)

        datasetInputData_tensor = tf.stack(inputTensorTuple, axis = 1)
        dataset_tensor = tf.data.Dataset.from_tensor_slices((datasetInputData_tensor, outputItemTensor))

        dataset = dataset_tensor.batch(3, drop_remainder = True)
        return dataset

if __name__ == '__main__':
    unittest.main()
