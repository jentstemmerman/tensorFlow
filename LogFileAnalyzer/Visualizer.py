import struct

import matplotlib.pyplot as plt

def plot_time_series(filename):
    # Read the data from the file
    with open(filename, 'rb') as file:
        data = []
        while chunk := file.read(4):  # Assuming each float is 4 bytes (single precision)
            data.append(struct.unpack('f', chunk)[0])

    # Plot the data
   # plt.figure(figsize=(10, 5))
    plt.plot(data, label='Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series Data Plot')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
plot_time_series('Simularities_Markdirty.dat')


