import DataTreatment
import TokenParser

class Logfile:
    
    def __init__(self, fileName, itemList, DataTypeForTensors):
        self.fileName = fileName
        self.itemList = itemList
        self.DataTypeForTensors = DataTypeForTensors
        self.dataTreatment = DataTreatment.DataTreatment(len(itemList))
        
    def GetTokensFromLines(self, lines): 
        tokenListList = []
        
        for line in lines:
            tokenList = TokenParser.GetItemsInLine(line, self.itemList, self.dataTreatment)

            if (len(tokenList) > 0):
                tokenListList.append(tokenList)

        return tokenListList

    def GetTokens(self, startLine=None, endLine=None):
        with open(self.fileName, 'r', encoding='utf-8', errors='replace') as file:
            lines = []
            for i, line in enumerate(file):
                if startLine is None and endLine is None:
                    lines.append(line)
                else:
                    if i >= startLine and i <= endLine:
                        lines.append(line)
                    if i >= endLine:
                        break
        
        return self.GetTokensFromLines(lines)
    
    def GetTokensFromLineIndexList(self, linesIndexList):
        if not linesIndexList:
            return []
        
        min_index = min(linesIndexList)
        max_index = max(linesIndexList)
        
        with open(self.fileName, 'r', encoding='utf-8', errors='replace') as file:
            lines = []
            for i, line in enumerate(file):
                if i > max_index:
                    break
                if i >= min_index and i <= max_index and i in linesIndexList:
                    lines.append(line)
                    
        return self.GetTokensFromLines(lines)
                   
                    

                    
                    
                    