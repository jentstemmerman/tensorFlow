import multiprocessing
import queue
import time
import tensorflow as tf
import os

def Worker(cpu, inputQueue, outputQueue, parameters):
    print("Starting worker ", cpu)
    
    with tf.device('/CPU:0'):
        while True:
            try:
                func, args = inputQueue.get(timeout=1000)  

                if (func == 'exit'):
                    print("End worker ", cpu, "")
                    break

                result = func(parameters, *args)

                outputQueue.put(result)

            except queue.Empty:
                time.sleep(1)
                continue

def KillWorkers(Workers):
    for threadId in range(len(Workers)):
        inputQueue  = Workers[threadId]["inputQueue"]
        inputQueue.put(("exit", ()))

def CreateWorkers(numberOfWorkers, parameters):

    with tf.device('/CPU:0'):
        workers = None

        inputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]
        outputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]

        workers = []

        for i in range(numberOfWorkers):
            process = multiprocessing.Process(target=Worker, args=(i, inputQueues[i], outputQueues[i], parameters))

            process.start()
            worker_dict = {"process":process, "inputQueue":inputQueues[i],"outputQueue":outputQueues[i]}
            workers.append(worker_dict)

        return workers
    