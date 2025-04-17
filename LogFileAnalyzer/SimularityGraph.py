import tensorflow as tf
import BinaryFileHandler
import matplotlib.pyplot as plt

if __name__ == "__main__":
    import sys

    parameters = sys.argv[1:]

    if (len(parameters)==1):
        SimularityFile                = parameters[0]
    else:
        print("Did not receive 1 parameters. Parameters received: " + str(sys.argv-1))
    
    print("SimularityFile: ", SimularityFile)
    outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(SimularityFile, data_type="float32", numbers_per_line=1, sequence_length=None)
    recordCount = outputBinaryFileHandler.get_recordCount()
    
    
    graph_data = outputBinaryFileHandler.read_records(0, recordCount)
    graph_data = tf.squeeze(graph_data)
    
    # Plot the graph_data with very small points
    plt.scatter(range(len(graph_data)), graph_data, s=1)  # 's' sets the size of the points
    plt.title("Simularities")
    plt.xlabel("Line number")
    plt.ylabel("Simularity")
    plt.grid(True)
    
    # Disable scientific notation for the x-axis (line numbers)
    plt.ticklabel_format(style='plain', axis='x')
    
    plt.show()
    
    
    

    outputBinaryFileHandler.close()