import threading
import tensorflow as tf
import numpy as np
import ModelClass

class GenerationClass:
    def __init__(self, step_model, inputTensor, nexttokenIds, states=None):
        self.step_model = step_model
        self.inputTensor  = inputTensor
        self.nexttokenIds = nexttokenIds
        self.states       = states
        self.nextPredictedTokens = None
        self.probabilityPredictions = None
        self.probabilityOfNextTokens = None

    def GenerationThread(self):
        self.nextPredictedTokens, self.probabilityPredictions, self.probabilityOfNextTokens = self.step_model.generate_steps(self.inputTensor, self.nexttokenIds, states=self.states)

    def StartGeneration(self):
        self.generation_thread = threading.Thread(target=self.GenerationThread, args=())
        self.generation_thread.start()

    def WaitForCompletion(self):
        # Wait for the thread to finish 
        self.generation_thread.join()
        return self.nextPredictedTokens, self.probabilityPredictions, self.probabilityOfNextTokens
