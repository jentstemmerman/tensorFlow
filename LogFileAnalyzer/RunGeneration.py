import multiprocessing
import tensorflow as tf
import numpy as np


class GenerationClass:

    def GenerationThread(step_model, inputTensor, nexttokenIds, states=None):
        nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens = step_model.generate_steps(inputTensor, nexttokenIds, states=states)
        return nextPredictedTokens, probabilityPredictions, probabilityOfNextTokens

