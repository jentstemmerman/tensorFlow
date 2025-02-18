import secrets
import base64

class ModelHash:
    def __init__(self, hash=None):
        
        if hash is not None:
            self.hash = hash
        else:   
            self.hash = secrets.token_bytes(64) 
            self.hash = base64.urlsafe_b64encode(self.hash).decode('utf-8')  # make sure this hash does not cause trouble when stored in a file
            
        print("ModelHash __init__: ", hash, self.hash)
        

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return self.hash
    
    def getHash(self):
        return self.hash
    
    def get_config(self):
        return {"hash": self.hash}
    
    @classmethod
    def from_config(cls, config):
        return cls(hash=config["hash"])