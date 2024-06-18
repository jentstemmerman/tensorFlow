TrainingsParameter = {"Model": {#"Network":"LSTMMultiModel",
                                "Network":"Transformer",
                                "returnState":True,
                                "rnn_units":1024,
                                "seq_length":256 ,
                                "return_states": True
                                },
                      "EPOCHS":1,
                      "Use_Validation":False,
                      "BatchSize":32,
                      "removeDuplicates":False,
                      "num_parallel_calls":64,
                      "stride_dataset": 512,
                      "shuffleTraining":True,
                      "BUFFER_SIZE":65536,
                      "SampleDataSet":False,   # does not really work, using stride is better
                      "DataTypeForTensors":'float32'
                    }
