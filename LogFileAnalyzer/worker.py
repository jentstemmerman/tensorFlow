import multiprocessing
import queue
import time
import tensorflow as tf

import numpy as np

def Worker(cpu, inputQueue, outputQueue, id_from_tokenListNumpy, token_from_IdListNumpy):
    with tf.device('/CPU:0'):
        print("Starting worker ", cpu)
        
        id_from_tokenList = []
        
        for id_from_token in id_from_tokenListNumpy:
            lookup = tf.keras.layers.StringLookup(vocabulary=id_from_token.tolist())
            id_from_tokenList.append(lookup)
            
    
        token_from_idList = []
        
        for token_from_id in token_from_IdListNumpy:
            lookup = tf.keras.layers.StringLookup(vocabulary=token_from_id.tolist())
            token_from_idList.append(lookup)
        
        
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

def CreateWorkers(numberOfWorkers, id_from_tokenList, token_from_idList):
    
    with tf.device('/CPU:0'):
        multiprocessing.set_start_method('spawn', force=True)   
        workers = None

        inputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]
        outputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]

        workers = []

        id_from_tokenListConverted = []
        
        for id_from_token in id_from_tokenList:
            id_from_tokenConverted = np.array(id_from_token.get_vocabulary())
            id_from_tokenListConverted.append(id_from_tokenConverted)
            
        token_from_idListConverted = []
        
        for token_from_id in token_from_idList:
            token_from_idConverted = np.array(token_from_id.get_vocabulary())
            token_from_idListConverted.append(token_from_idConverted)

        for i in range(numberOfWorkers):
                
            process = multiprocessing.Process(target=Worker, args=(i, inputQueues[i], outputQueues[i], id_from_tokenListConverted, token_from_idListConverted))

            process.start()
            worker_dict = {"process":process, "inputQueue":inputQueues[i],"outputQueue":outputQueues[i]}
            workers.append(worker_dict)

        return workers
    