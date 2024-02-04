import tensorflow as tf


class DebugCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.index = 0

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        total_loss = logs["loss"] 
        with open('output_log.txt', 'a') as file:
            file.write("\n******** Callback: **************")
            file.write(str(logs["loss"]) + "\n")
            

        if (total_loss > 100000):
            print(str(self.index) + "...on_train_batch_end: "+ str(logs["loss"]) + "end of batch {}; got log keys: {}\n".format(batch, keys))

        
class EpochEndCallback(tf.keras.callbacks.Callback):
    def __init__(self, datasetForTraining, numberOfSamplesToTrainPerEpoch, numberOfSamplesToValidatePerEpoch):
        print("EpochEndCallback start newcreating new subsetForBatchTraining")
        self.numberOfBatchesToTrainPerEpoch    = numberOfSamplesToTrainPerEpoch
        self.numberOfBatchesToValidatePerEpoch = numberOfSamplesToValidatePerEpoch
        self.subsetForBatchTraining            = datasetForTraining.take(self.numberOfBatchesToTrainPerEpoch)
        Dataset.PrintDatasetTuple(self.subsetForBatchTraining)
        self.subsetForBatchVerification        = None
        print("EpochEndCallback start newcreating new subsetForBatchTraining done")

    def on_epoch_end(self, epoch, logs=None):
        print("on_epoch_end -> Creating new subsetForBatchTraining")
        self.subsetForBatchTraining     = datasetForTraining.shuffle(TrainingsParameter["BUFFER_SIZE"]).take(self.numberOfBatchesToTrainPerEpoch)
        Dataset.PrintDatasetTuple(self.subsetForBatchTraining)
        print("on_epoch_end Creating new subsetForBatchTraining done")