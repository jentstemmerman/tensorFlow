from Mask import Mask

class DataTreatment:
    
    def __init__(self, numberOfItems):

        self.state = []
        self.numberOfItems = numberOfItems

        for _ in range(self.numberOfItems):
            self.state.append({})        


    def TreatValue(self, index, item, value, stripped_line):
        
        result = value

        if ("Numerical" in item):
            if (value == ""):
                result = Mask.Universal_Mask_Token
            else:    
                if ("Categorical" in item):
                    result = str(result)
                else:
                    if (item["Numerical"] == "Hex"):
                        result = int(result, 16)
                    elif (item["Numerical"] == "Dec"):
                        result = int(result)

                    if (("Delta" in item) and item["Delta"] == "True"):
                        if ("Previous" in self.state[index]):
                            previousValue                 = self.state[index]["Previous"]
                            self.state[index]["Previous"] = result
                            result                       -= previousValue
                        else:
                            self.state[index]["Previous"] = result
                            result =  Mask.Universal_Mask_Token
        return result
            
