import tensorflow as tf
import BinaryFileHandler
import os
from mpld3 import save_html
import sys
import matplotlib
import matplotlib.pyplot as plt
    
    
def MakeSimularityPlot(SimularityFile, DataTypeForTensors, block=False, Save=True):
     
    if block:
        matplotlib.use('TkAgg')  # Use an interactive backend if block is True
    else:
        matplotlib.use('agg')    # Use non-interactive backend for saving only


    if type(SimularityFile) is str:
        base_name = os.path.splitext(os.path.basename(SimularityFile))[0]
        directory = os.path.dirname(SimularityFile)
        print("SimularityFile: ", base_name)
        outputBinaryFileHandler = BinaryFileHandler.BinaryFileHandler(SimularityFile, data_type=DataTypeForTensors, numbers_per_line=1, sequence_length=None)
    else:
        assert isinstance(SimularityFile, BinaryFileHandler.BinaryFileHandler), "SimularityFile must be a BinaryFileHandler instance"
        outputBinaryFileHandler = SimularityFile
        base_name = os.path.splitext(os.path.basename(SimularityFile.filename))[0]
        directory = os.path.dirname(SimularityFile.filename)
        
    recordCount = outputBinaryFileHandler.get_recordCount()
    
    
    graph_data = outputBinaryFileHandler.read_records(0, recordCount)
    graph_data = tf.squeeze(graph_data)
    average = tf.reduce_mean(graph_data)
    print("Average: ", average.numpy())
    # Plot the graph_data with very small points and lines to the zero axis
   # plt.figure(figsize=(16, 9))  # Set the figure size to full screen (16:9 aspect ratio)
    
    plt.scatter(range(len(graph_data)), graph_data, s=1, label="Data Points")  # 's' sets the size of the points
   # plt.vlines(range(len(graph_data)), 0, graph_data, colors='gray', alpha=0.05, linewidth=0.5, label="Lines to Zero Axis")  # Vertical lines to zero axis
    plt.axhline(y=average.numpy(), color='red', linestyle='--', linewidth=1, label=f"Average: {average.numpy():.4f}")  # Add average line
    plt.ylim(-1, 1)  # Set the vertical scale between -1 and +1
    plt.title("Simularities " + base_name)
    plt.xlabel("Line number")
    plt.ylabel("Simularity")
    plt.grid(True)
    plt.legend()
    
    # Disable scientific notation for the x-axis (line numbers)
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability 
    plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels   
    
    if (Save):
        # Save the plot to a file
        output_file = os.path.join(directory, f"SimularitiesPlot_{base_name}.png")
        plt.savefig(output_file, dpi=300)  # Save with high resolution
        print(f"Plot saved as {output_file}")
    
    plt.show(block=block)  # Show the plot, blocking if specified
    
    plt.close()  # Close the plot to free up memory
    
    outputBinaryFileHandler.close()


        
    


if __name__ == "__main__":
   
    parameters = sys.argv[1:]

    if (len(parameters)==1):
        SimularityFile                = parameters[0]
    else:
        print("Did not receive 1 parameters. Parameters received: " + str(sys.argv-1))
    
    MakeSimularityPlot(SimularityFile, DataTypeForTensors='float32', block=True)
