# Define color ranges

# Define color ranges
MIN_COLOR = (255, 0, 0)  # Red for low numbers
MAX_COLOR = (0, 255, 0)  # Green for high numbers

def interpolate_color(min_color, max_color, factor):
    """Interpolate between two colors by a factor."""
    return tuple(int(min_color[i] + (max_color[i] - min_color[i]) * factor) for i in range(3))

def get_color_for_number(number, min_number, max_number):
    """Calculate the color for a given number."""
    factor = (number - min_number) / (max_number - min_number)
    return interpolate_color(MIN_COLOR, MAX_COLOR, factor)

print(str(get_color_for_number(20, 0, 255)))