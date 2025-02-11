import secrets

class ModelHash:
    def __init__(self, hash=None):
        
        if hash is not None:
            self.hash = hash
        else:   
            self.hash = secrets.token_hex(64)  
        
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