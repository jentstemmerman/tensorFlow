import tensorflow as tf
import numpy as np
import DataTreatment
import TokenParser
import os
import itertools


# each shard contains the subset of the dataset that will be used by one thread. 
class DatasetStreamWriter:
    def __init__(self, *, inputFile, dataSetFile, itemList, id_from_tokenList, DataTypeForTensors, numberOfShards, chardGranularity, windowSize):
        self.dataSetFile = dataSetFile
        self.itemList = itemList
        self.numberOfShards = numberOfShards
        self.chardGranularity = chardGranularity
        self.windowSize = windowSize
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
        numberOfLinesInFile = 0
        output_dir = os.path.dirname(self.dataSetFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a list of TFRecordWriters, one for each shard
        writers = [tf.io.TFRecordWriter(f"{self.dataSetFile}_shard_{i}") for i in range(self.numberOfShards)]
        
        for lineIndex, tokenList in enumerate(self.process_file_line_by_line()):
            if (lineIndex % 10000 == 0):
                print(f"Processing line: {lineIndex}")  
                
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
            
            shardsWrittenDebug = []
            prevousShardWritten = False
            nextShardWritten    = False
            
            # Determine which writer to use based on the modulo of the index
            shard_offset = (lineIndex // self.chardGranularity)
            shard_index = shard_offset % self.numberOfShards
            
            writers[shard_index].write(example.SerializeToString()) 
            
            shardsWrittenDebug.append(shard_index)
            
            
            # Watch out. There could be overlap between this shard and the previous shard, if there is one
            if (shard_offset > 0): # is there a previous shard?

                shard_index_of_previous_shard      = (shard_offset  - 1) % self.numberOfShards
                start_line_index_of_current_shard  = (shard_offset)      * self.chardGranularity   
                
                if lineIndex < (start_line_index_of_current_shard + self.windowSize):
                    writers[shard_index_of_previous_shard].write(example.SerializeToString())
                    prevousShardWritten = True
                    shardsWrittenDebug.append(shard_index_of_previous_shard)
            
            # there could also be overlap between this shard and the next shard
            shard_offset_of_next_shard      = shard_offset  + 1
            shard_index_of_next_shard       = shard_offset_of_next_shard  % self.numberOfShards
            start_line_index_of_next_shard  = shard_offset_of_next_shard  * self.chardGranularity   
            
            if lineIndex >= (start_line_index_of_next_shard + 1):
                writers[shard_index_of_next_shard].write(example.SerializeToString())
                nextShardWritten = True
                shardsWrittenDebug.append(shard_index_of_next_shard)
                
            with open("debug_log.txt", "a") as debug_file:
                debug_file.write(f"lineIndex: {lineIndex}, {columns} shard_offset: {shard_offset}, shardsWritten: {shardsWrittenDebug}, prevousShardWritten: {prevousShardWritten}, nextShardWritten: {nextShardWritten}\n")
            
            numberOfLinesInFile += 1
            
        # Close all writers
        for writer in writers:
            writer.close()



class DatasetStreamReader:
    def __init__(self, dataSetFile, itemList, DataTypeForTensors, seq_length, stride, numberOfShards):
        with tf.device('/CPU:0'):
            self.dataSetFile = dataSetFile
            self.itemList = itemList
            self.DataTypeForTensors = DataTypeForTensors
            self.seq_length = seq_length
            self.num_features = len(self.itemList)
            self.stride = stride
            self.numberOfShards = numberOfShards

    def parse_tfrecord(self, example_proto):
        with tf.device('/CPU:0'):
            features = {
                "Ids": tf.io.FixedLenFeature([self.num_features], self.DataTypeForTensors),
                "TokenString": tf.io.FixedLenFeature([self.num_features], tf.string)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return [
                parsed_features["Ids"],
                parsed_features["TokenString"]
            ]

    def GetStreamingDataset(self, shard_index=None):
        with tf.device('/CPU:0'):
            if shard_index is not None:
                # Create a dataset from the specific shard file
                shard_file = f"{self.dataSetFile}_shard_{shard_index}"
                dataset = tf.data.TFRecordDataset(shard_file)
            else:
                # Create a list of shard file paths
                shard_files = [f"{self.dataSetFile}_shard_{i}" for i in range(self.numberOfShards)]
                # Create a dataset from the list of shard files
                dataset = tf.data.TFRecordDataset(shard_files)
            
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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

    def GetStreamingIterator(self, shard_index):
        with tf.device('/CPU:0'):
            """Use an iterator for finer control over dataset streaming."""
            dataset = self.GetStreamingDataset(shard_index)
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
            dataset_as_numpy_iterator = dataset.as_numpy_iterator()
            iterator = iter(dataset_as_numpy_iterator)  # Allows manual control
            return iterator

    def JumpToIndex(self, start, stop, iterator):
        with tf.device('/CPU:0'):
            itertools.islice(iterator, start, stop)
