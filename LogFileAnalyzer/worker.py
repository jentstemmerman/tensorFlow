import multiprocessing
import queue
import time
import tensorflow as tf
import os


def Worker(cpu, inputQueue, outputQueue):
    print("Starting worker ", cpu)

    with tf.device('/CPU:0'):
        while True:
            try:
                func, args = inputQueue.get(timeout=1000)  

                if (func == 'exit'):
                    continue

                result = func(*args)

                outputQueue.put(result)

            except queue.Empty:
                time.sleep(1)
                continue

def CreateWorkers(numberOfWorkers):

    with tf.device('/CPU:0'):
        workers = None

        inputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]
        outputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]

        workers = []

        for i in range(numberOfWorkers):
            process = multiprocessing.Process(target=Worker, args=(i, inputQueues[i], outputQueues[i]))

            process.start()
            worker_dict = {"process":process, "inputQueue":inputQueues[i],"outputQueue":outputQueues[i]}
            workers.append(worker_dict)

        return workers






