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
                tokenList = [item for sublist in tokenList for item in sublist]  # Flatten the list of lists
                transformed_columns = []
                columns = []
                
                for index, id_from_token in enumerate(self.id_from_tokenList):
                    column = np.array([tokenList[index]])  # Process one tokenized line at a time
                    
                    if id_from_token is not None:
                        TransformedItemTensor = id_from_token(column)
                    else:
                        TransformedItemTensor = tf.convert_to_tensor(column, dtype=self.DataTypeForTensors)
                    
                    columns.append(column)
                    transformed_columns.append(TransformedItemTensor.numpy())  # Convert to NumPy

                feature = {
                    "Ids": tf.train.Feature(float_list=tf.train.FloatList(value=np.concatenate(transformed_columns).flatten())),
                    "TokenString": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(str(item)) for item in np.concatenate(columns).flatten()]))
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
        self.offset = 0
        
    def parse_tfrecord(self, example_proto):
        features = {
            "Ids": tf.io.FixedLenFeature([self.num_features], self.DataTypeForTensors),
            "TokenString": tf.io.FixedLenFeature([self.num_features], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        return [
            parsed_features["Ids"],
            parsed_features["TokenString"]
        ]

    def GetStreamingDataset(self):
        dataset = tf.data.TFRecordDataset(self.dataSetFile)
        dataset = dataset.map(lambda x: self.parse_tfrecord(x))

        dataset = dataset.window(self.seq_length + 1, shift=self.stride, drop_remainder=True)

        dataset = dataset.flat_map(
        lambda ids, tokens: tf.data.Dataset.zip(
                                                (ids.take(self.seq_length).batch(self.seq_length),
                                                tokens.take(self.seq_length).batch(self.seq_length),
                                                ids.skip(self.seq_length).batch(1),
                                                tokens.skip(self.seq_length).batch(1))
                                                )     
        )

        return dataset




def StreamDataSet(inputFile, outputDataSetFile, itemList, id_from_tokenList, DataTypeForTensors, seq_length, stride):
    tf.config.run_functions_eagerly(True)  
    tf.data.experimental.enable_debug_mode() 
                                 
    datasetStreamWriter = DatasetStreamWriter(inputFile, outputDataSetFile, itemList, id_from_tokenList, DataTypeForTensors)
    datasetStreamWriter.WriteStreamingDataset()
    
    datasetStreamReader = DatasetStreamReader(outputDataSetFile, itemList, DataTypeForTensors, seq_length, stride)
    datasetStreamer = datasetStreamReader.GetStreamingDataset()
    

    
    for batch in datasetStreamer.take(3):
        print(batch)

    return datasetStreamer