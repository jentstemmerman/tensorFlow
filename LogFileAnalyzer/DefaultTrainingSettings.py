TrainingsParameter = {"Model": {"Network":"LSTMMultiModel",
                                "returnState":True,
                                "rnn_units":1024,
                                "seq_length":32 ,
                                "dense_layer": True,
                                "return_states": True
                                },
                      "EPOCHS":5,
                      "Use_Validation":False,
                      "BatchSize":64,
                      "removeDuplicates":False,
                      "num_parallel_calls":64,
                      "stride_dataset": 1,
                      "shuffleTraining":False,
                      "BUFFER_SIZE":65536,
                      "SampleDataSet":False,   # does not really work, using stride is better
                      "DataTypeForTensors":'float32'
                    }
