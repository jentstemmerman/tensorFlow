import os
import Signature
import LogFileIterator
import Constants


class SignatureCollection:
    def __init__(self, signatureDirectory):
        self.signatureList = []
        signatureIterator = LogFileIterator.FileIterator(signatureDirectory, Constants.GetSignatureFilesExtention())
    
        signature_file = signatureIterator.get_first_file()
        
        while (signature_file is not None):
            signature = Signature.Signature.LoadSignatureFromFile(signature_file)
            self.signatureList.append(signature)
            
            signature_file = signatureIterator.get_next_file()
            
    def CalculateSignatures(self, one_step_model):
        assert one_step_model is not None, "one_step_model must not be None."
        assert isinstance(one_step_model, Signature.OneStepModel), "one_step_model must be an instance of OneStepModel."
        
        for signature in self.signatureList:
            signature.CalculateSignature(one_step_model)        
        

    def add_signature(self, signature):
        self.signatureList.append(signature)

    def get_signatures(self):
        return self.signatureList
    
    def get_signatures_and_calculateVectors(self, one_step_model):
        for signature in self.signatureList:
            signature.CalculateSignature(one_step_model)
        
        return self.signatureList

    def clear_signatures(self):
        self.signatureList.clear()

    def __len__(self):
        return len(self.signatureList)

    def __getitem__(self, index):
        return self.signatureList[index]