TrainingsParameter = {"Model": {#"Network":"LSTMMultiModel",
                                "Network":"Transformer",
                                "returnState":True,
                                "rnn_units":1024,
                                "seq_length":64 ,
                                "return_states": True
                                },
                      "EPOCHS":20,
                      "Use_Validation":False,
                      "BatchSize":8,
                      "removeDuplicates":False,
                      "num_parallel_calls":64,
                      "stride_dataset": 1,
                      "shuffleTraining":False,
                      "BUFFER_SIZE":65536,
                      "SampleDataSet":False,   # does not really work, using stride is better
                      "DataTypeForTensors":'float32'
                    }
