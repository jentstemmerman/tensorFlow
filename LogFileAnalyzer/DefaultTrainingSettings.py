TrainingsParameter = {"Model": {#"Network":"LSTMMultiModel",
                                #"Network":"Transformer",
                                "Network":"Transformer_DecoderOnly",
                                "returnState":True,
                                "rnn_units":1024,
                                "seq_length":32 ,
                                "return_states": True
                                },
                      "EPOCHS":2,
                      "Use_Validation":False,
                      "BatchSize":64,
                      "removeDuplicates":False,
                      "num_parallel_calls":64,
                      "stride_dataset": 5,
                      "shuffleTraining":True,
                      "BUFFER_SIZE":65536,
                      "SampleDataSet":False,   # does not really work, using stride is better
                      "DataTypeForTensors":'float32'
                    }
