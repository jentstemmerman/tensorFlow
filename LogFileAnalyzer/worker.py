import multiprocessing
import queue
import time
import tensorflow as tf

import numpy as np

def Worker(cpu, inputQueue, outputQueue, id_from_tokenListNumpy):
    print("Starting worker ", cpu)
    
    id_from_tokenList = []
    
    for id_from_token in id_from_tokenListNumpy:
        lookup = tf.keras.layers.StringLookup(vocabulary=id_from_token.tolist())
        id_from_tokenList.append(lookup)
    
    with tf.device('/CPU:0'):
        while True:
            try:
                func, args = inputQueue.get(timeout=1000)  

                if (func == 'exit'):
                    break

                result = func(id_from_tokenList, *args)

                outputQueue.put(result)

            except queue.Empty:
                time.sleep(1)
                continue

def KillWorkers(Workers):
    for threadId in range(len(Workers)):
        inputQueue  = Workers[threadId]["inputQueue"]
        inputQueue.put(("exit", ()))

def CreateWorkers(numberOfWorkers, id_from_tokenList):
    multiprocessing.set_start_method('spawn', force=True)   
    with tf.device('/CPU:0'):
        
        workers = None

        inputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]
        outputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]

        workers = []

        for i in range(numberOfWorkers):
            id_from_tokenListConverted = []
            
            for id_from_token in id_from_tokenList:
                id_from_tokenConverted = np.array(id_from_token.get_vocabulary())
                id_from_tokenListConverted.append(id_from_tokenConverted)
                
                
            process = multiprocessing.Process(target=Worker, args=(i, inputQueues[i], outputQueues[i], id_from_tokenListConverted))

            process.start()
            worker_dict = {"process":process, "inputQueue":inputQueues[i],"outputQueue":outputQueues[i]}
            workers.append(worker_dict)

        return workers
    