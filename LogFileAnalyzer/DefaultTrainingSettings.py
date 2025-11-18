TrainingsParameter = {"Model": {#"Network":"LSTMMultiModel",
                                #"Network":"Transformer",
                                "Network":"Transformer_DecoderOnly",
                                "returnState":True,
                                "rnn_units":1024,
                                "seq_length":64,
                                "return_states": True
                                },
                      "EPOCHS":3,  #etest
                      "Use_Validation":False,
                      "BatchSize":64,
                      "removeDuplicates":True,
                      "num_parallel_calls":64,
                      "stride_dataset": 13,
                      "stride_dataset_inference": 31,
                      "shuffleTraining":True,
                      "BUFFER_SIZE":65536,
                      "SampleDataSet":False,   # does not really work, using stride is better
                      "DataTypeForTensors":'float16'
                    }
