import multiprocessing
import queue
import time
import tensorflow as tf
import Dataset
import numpy as np

def Worker(cpu, inputQueue, outputQueue, id_from_tokenListNumpy):
    with tf.device('/CPU:0'):
        print("Starting worker ", cpu)
        
        id_from_tokenList = []
        token_from_idList = []
        
        for id_from_tokenNumpy in id_from_tokenListNumpy:
            if id_from_tokenNumpy is None:
                id_from_token = None
                token_from_id = None
            else:   
                id_from_token = tf.keras.layers.StringLookup(vocabulary=id_from_tokenNumpy.tolist(), mask_token=Dataset.Universal_Mask_Token)
                token_from_id = tf.keras.layers.StringLookup(vocabulary=id_from_tokenNumpy.tolist(), mask_token=Dataset.Universal_Mask_Token, invert=True)
            id_from_tokenList.append(id_from_token)
            token_from_idList.append(token_from_id)
        
        while True:
            try:
                func, args = inputQueue.get(timeout=1000)  

                if (func == 'exit'):
                    break

                result = func((id_from_tokenList, token_from_idList) , *args)

                outputQueue.put(result)

            except queue.Empty:
                time.sleep(1)
                continue

def KillWorkers(Workers):
    for threadId in range(len(Workers)):
        inputQueue  = Workers[threadId]["inputQueue"]
        inputQueue.put(("exit", ()))

def CreateWorkers(numberOfWorkers, id_from_tokenList):
    
    with tf.device('/CPU:0'):
        multiprocessing.set_start_method('spawn', force=True)   
        workers = None

        inputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]
        outputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]

        workers = []

        id_from_tokenListConverted = []
        
        for id_from_token in id_from_tokenList:
            if id_from_token is None:
                id_from_tokenConverted = None
            else:
                id_from_tokenConverted = np.array(id_from_token.get_vocabulary())
            
            id_from_tokenListConverted.append(id_from_tokenConverted)

        for i in range(numberOfWorkers):
                
            process = multiprocessing.Process(target=Worker, args=(i, inputQueues[i], outputQueues[i], id_from_tokenListConverted))

            process.start()
            worker_dict = {"process":process, "inputQueue":inputQueues[i],"outputQueue":outputQueues[i]}
            workers.append(worker_dict)

        return workers
    