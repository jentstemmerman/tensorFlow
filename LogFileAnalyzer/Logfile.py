import DataTreatment
import TokenParser

class Logfile:
    
    def __init__(self, fileName, itemList=None):
        self.fileName = fileName
        self.itemList = itemList
        
        if itemList is None or len(itemList) == 0:
            self.dataTreatment = None
        else:
            self.dataTreatment = DataTreatment.DataTreatment(len(itemList))


    def GetNumberOfLinesInFile(self):
        numberOfLines = 0
        with open(self.fileName, "r", errors='ignore') as reading_file:
            for line in reading_file:
                if line.strip():  # Check if the line is not empty
                    numberOfLines += 1
        return numberOfLines
    
    
    def GetTokensFromLines(self, lines): 
        tokenListList = []
        
        for line in lines:
            tokenList = TokenParser.GetItemsInLine(line, self.itemList, self.dataTreatment)

            if (len(tokenList) > 0):
                tokenListList.append(tokenList)

        return tokenListList


    def GetLines(self, startLine=None, endLine=None):
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
        
        return lines


    def GetLinesFromLineIndexList(self, linesIndexList):
        if not linesIndexList:
            return []
        
        min_index = min(linesIndexList)
        max_index = max(linesIndexList)
        
        with open(self.fileName, 'r', encoding='utf-8', errors='replace') as file:
            lines = []
            for i, line in enumerate(file):
                
                if i > max_index:
                    break
                
                if i >= min_index and i <= max_index:
                    if i in linesIndexList:
                        lines.append(line)
                    else:
                        lines.append("")
                        
                    
        return lines


    def GetTokens(self, startLine=None, endLine=None):

        lines = self.GetLines(startLine, endLine)
        return self.GetTokensFromLines(lines)
    
    
    def GetTokensFromLineIndexList(self, linesIndexList):
        
        lines = self.GetLinesFromLineIndexList(linesIndexList)
        return self.GetTokensFromLines(lines)
                   
                    

                    
                    
                    