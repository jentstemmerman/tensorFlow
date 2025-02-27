import tensorflow as tf
import numpy as np
import DataTreatment
import TokenParser
import os

class DatasetStreamWriter:
    def __init__(self, *, inputFile, dataSetFile, itemList, id_from_tokenList, DataTypeForTensors):
        self.dataSetFile = dataSetFile
        self.itemList = itemList
        self.id_from_tokenList = id_from_tokenList
        self.DataTypeForTensors = DataTypeForTensors
        self.dataTreatment = DataTreatment.DataTreatment(len(self.itemList))
        self.input_file_handler = open(inputFile, "r", errors="ignore")

    def getLinesAndTokens(self, line): 
        tokenList = []

        items = TokenParser.GetItemsInLine(line, self.itemList, self.dataTreatment)

        if (len(items) > 0):
            tokenList.append(items)

        return tokenList

    def process_file_line_by_line(self):
        for line in self.input_file_handler:
            tokenList = self.getLinesAndTokens(line)
            yield tokenList  # Return one tokenized line at a time

    def WriteStreamingDataset(self):
        output_dir = os.path.dirname(self.dataSetFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with tf.io.TFRecordWriter(self.dataSetFile) as writer:
            for tokenList in self.process_file_line_by_line():
                transformed_columns = []
                for index, id_from_token in enumerate(self.id_from_tokenList):
                    column = np.array([tokenList[index]])  # Process one tokenized line at a time
                    
                    if id_from_token is not None:
                        TransformedItemTensor = id_from_token(column)
                    else:
                        TransformedItemTensor = tf.convert_to_tensor(column, dtype=self.DataTypeForTensors)
                    
                    transformed_columns.append(TransformedItemTensor.numpy())  # Convert to NumPy

                feature = {
                    "tokens": tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate(transformed_columns).flatten()))
                }
                
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())


class DatasetStreamReader:
    def __init__(self, dataSetFile, itemList, DataTypeForTensors, seq_length, stride):
        self.dataSetFile = dataSetFile
        self.itemList = itemList
        self.DataTypeForTensors = DataTypeForTensors
        self.seq_length = seq_length
        self.num_features = len(self.itemList)
        self.stride = stride
        self.dataTreatment = DataTreatment.DataTreatment(len(self.itemList))
        
    def parse_tfrecord(self, example_proto):
        """Parses a single TFRecord example."""
        features = {
            "tokens": tf.io.FixedLenFeature([self.num_features], self.DataTypeForTensors)  # Adjust shape as needed
        }
        return tf.io.parse_single_example(example_proto, features)["tokens"]

    def GetStreamingDataset(self):
        """Creates a streaming dataset from a TFRecord file with a sliding window."""
        dataset = tf.data.TFRecordDataset(self.dataSetFile)  # Streaming from disk
        dataset = dataset.map(lambda x: self.parse_tfrecord(x))
       
        # Apply sliding window
        dataset = dataset.window(self.seq_length, shift=self.stride, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(self.seq_length))

        return dataset


def StreamDataSet(inputFile, outputDataSetFile, itemList, id_from_tokenList, DataTypeForTensors, seq_length, stride):
    tf.config.run_functions_eagerly(True)  
    tf.data.experimental.enable_debug_mode() 
    stride = 1
                                 
    datasetStreamWriter = DatasetStreamWriter(inputFile, outputDataSetFile, itemList, id_from_tokenList, DataTypeForTensors)
    datasetStreamWriter.WriteStreamingDataset()
    
    datasetStreamReader = DatasetStreamReader(outputDataSetFile, itemList, DataTypeForTensors, seq_length, stride)
    datasetStreamer = datasetStreamReader.GetStreamingDataset()
    

    
    for batch in datasetStreamer.take(3):
        print(batch)

    return datasetStreamer