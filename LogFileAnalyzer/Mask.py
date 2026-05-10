
class Mask:
    
    
    Universal_Mask_Token: str = "MaskedToken"
    
    def __init__(self):
        pass
    
    @staticmethod
    def GetMaskValue(featureIndex, id_from_tokenList, itemList):
        id_from_token = id_from_tokenList[featureIndex]
        
        if ( id_from_token is None):
            if featureIndex >= len(itemList):
                raise IndexError("Synthetic internal features must be categorical and provide a lookup layer")
            maskValue = itemList[featureIndex]['MaskValue']
        else:
            # Use the class-level constant defined here
            maskValue = id_from_token(Mask.Universal_Mask_Token).numpy()
        
        return maskValue
    
    @staticmethod
    def GetMaskValueList(id_from_tokenList, itemList):   # JENTS BUG
        maskValueList = []

        for index, _ in enumerate(id_from_tokenList):
            maskValue = Mask.GetMaskValue(index, id_from_tokenList, itemList)
            maskValueList.append(maskValue)

        return maskValueList
