import struct

import matplotlib.pyplot as plt
import SimularityGraph

def plot_time_series(filename):
    SimularityGraph.MakeSimularityPlot(filename, block=False, Save=True, DataTypeForTensors='float32')
        



# Example usage
plot_time_series('../data/UserData/User_0/logFilesAnalyzed/1340_Neo_KioxiaBiCS6_8ch-4TB_SIE_Regression_Combination_RandomBasic_drive_log/Similarities/Simularities_MarkDirty_7.dat')


