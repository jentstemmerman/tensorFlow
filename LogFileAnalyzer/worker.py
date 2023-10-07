import multiprocessing
import queue
import time

def Worker(cpu, inputQueue, outputQueue):
    import tensorflow as tf
    print("Starting worker ", cpu)


    with tf.device(f'/CPU:{cpu}'):
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

def cpu_work(data):
    return data * 2

def CreateWorkers(numberOfWorkers):

    workers = None

    import tensorflow as tf
    # Main process uses GPU
    with tf.device('/GPU:0'):
        print("Main process is using GPU")

    inputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]
    outputQueues = [multiprocessing.Queue() for _ in range(numberOfWorkers)]

    workers = []

    for i in range(numberOfWorkers):
        process = multiprocessing.Process(target=Worker, args=(i, inputQueues[i], outputQueues[i]))

        process.start()
        worker_dict = {"process":process, "inputQueue":inputQueues[i],"outputQueue":outputQueues[i]}
        workers.append(worker_dict)

    return workers

def DemonstrateWorkers(workers, numberOfWorkers):
    # Send messages to child processes
    for i in range(numberOfWorkers):
        workers[i]["inputQueue"].put((cpu_work, (i,)))

    # Send 'exit' message to child processes
    for _ in range(numberOfWorkers):
        workers[i]["inputQueue"].put(('exit', (i,)))

    # Wait for all child processes to finish
    for _ in range(numberOfWorkers):
        workers[i]["process"].join()





