import tensorflow as tf
import numpy as np
import DataTreatment
import TokenParser
import os
import itertools
import time

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
            # tokenList = self.getLinesAndTokens(line)
            tokenList = TokenParser.GetItemsInLine(line, self.itemList, self.dataTreatment)
            yield tokenList  # Return one tokenized line at a time

    def WriteStreamingDataset(self):
        numberOfLinesInFile = 0
        output_dir = os.path.dirname(self.dataSetFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


        # Create one TFRecordWriter per shard.
        shard_files = [f"{self.dataSetFile}_shard_{i}" for i in range(self.numberOfShards)]
        
        if all(os.path.exists(shard_file) for shard_file in shard_files):
            print("All shard files already exist. Skipping WriteStreamingDataset.")
            return
        else:
            print("Shard files do not exist. Writing new shard files.")
            
        writers = [tf.io.TFRecordWriter(shard_file) for shard_file in shard_files]

        # Set the batch size – adjust this as needed.
        batch_size = 1000
        batch_token_lists = []
        batch_line_indices = []

        start_time = time.time()

        def process_batch(batch_tokens, batch_indices):
            # Process each column in the batch at once.
            transformed_columns_batch = []
            for col_index, transform_fn in enumerate(self.id_from_tokenList):
                # Build a list for the current column from every line in the batch.
                col_batch = [tokens[col_index] for tokens in batch_tokens]
                if transform_fn is not None:
                    # Assume the transform function supports batch processing.
                    transformed_tensor = transform_fn(col_batch)
                else:
                    transformed_tensor = tf.convert_to_tensor(col_batch,
                                                            dtype=self.DataTypeForTensors)
                # Convert the batch result to a NumPy array.
                transformed_columns_batch.append(transformed_tensor.numpy())

            # Now, for each line in the batch, create and write the example.
            for i, lineIndex in enumerate(batch_indices):
                # For the current example, extract the corresponding row from each column.
                row_features = [np.atleast_1d(col[i]) for col in transformed_columns_batch]
                concatenated = np.concatenate(row_features).flatten()
                feature = {
                    "Ids": tf.train.Feature(
                        float_list=tf.train.FloatList(value=concatenated)
                    )
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized_example = example.SerializeToString()

                # Determine the target shard.
                shard_offset = lineIndex // self.chardGranularity
                shard_index = shard_offset % self.numberOfShards
                writers[shard_index].write(serialized_example)

                # Write to the previous shard if the current line falls into its window.
                if shard_offset > 0:
                    shard_index_prev = (shard_offset - 1) % self.numberOfShards
                    start_line_current_shard = shard_offset * self.chardGranularity
                    if lineIndex < (start_line_current_shard + self.windowSize):
                        writers[shard_index_prev].write(serialized_example)

                # Write to the next shard if the current line overlaps.
                shard_offset_next = shard_offset + 1
                shard_index_next = shard_offset_next % self.numberOfShards
                start_line_next_shard = shard_offset_next * self.chardGranularity
                if lineIndex >= (start_line_next_shard + 1):
                    writers[shard_index_next].write(serialized_example)

                nonlocal numberOfLinesInFile
                numberOfLinesInFile += 1

        # Accumulate lines and process in batches.
        for lineIndex, tokenList in enumerate(self.process_file_line_by_line()):
            batch_token_lists.append(tokenList)
            batch_line_indices.append(lineIndex)

            if (lineIndex % 10000 == 0) and batch_token_lists:
                print(f"Processing line: {lineIndex}")
                print(f"Time per 10000 lines: {time.time() - start_time:.4f} seconds")
                start_time = time.time()

            if len(batch_token_lists) >= batch_size:
                process_batch(batch_token_lists, batch_line_indices)
                batch_token_lists = []
                batch_line_indices = []

        # Process any remaining lines.
        if batch_token_lists:
            process_batch(batch_token_lists, batch_line_indices)

        # Close all writers.
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
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)
            return [
                parsed_features["Ids"],
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
                lambda ids: tf.data.Dataset.zip(
                    (ids.take(self.seq_length).batch(self.seq_length),     
                    ids.skip(self.seq_length).batch(1))
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
