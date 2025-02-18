class AnnotationId:
    def __init__(self, filterFile: str):
        self.annotationId = annotationId
        self.filterName = filterName
    
    def getAnnotationId(self):
        return self.annotationId
    
    def setAnnotationId(self, annotationId: int):
        self.annotationId = annotationId

    def getFilterName(self):
        return self.filterName
    
    def setFilterName(self, filterName: str):
        self.filterName = filterName
        

class Annotation:
    def __init__(self, annotationId : AnnotationId, lineNumber: int, lineCount: int, lineCount: int, comment: str):
        self.lineNumber = lineNumber
        self.lineCount = lineCount 
        self.annotationId = annotationId
        self.comment = comment
        
    
        
        
