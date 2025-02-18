from Npp import *

def highlight_lines():
    # Define your hidden text pattern
    hidden_text_pattern = "HIDDEN"
    
    # Clear all previous styling
    editor.styleResetDefault()
    editor.clearDocumentStyle()

    # Set the style for highlighted lines
    editor.styleSetFore(1, (255, 0, 0))  # Red text color
    editor.styleSetBold(1, True)

    # Iterate over all lines in the document
    line_count = editor.getLineCount()
    for line_number in range(line_count):
        line_text = editor.getLine(line_number)
        if hidden_text_pattern in line_text:
            start_pos = editor.positionFromLine(line_number)
            end_pos = editor.getLineEndPosition(line_number)
            editor.startStyling(start_pos, 0xFF)
            editor.setStyling(end_pos - start_pos, 1)

# Call the function to highlight lines
highlight_lines()
