import numpy as np
from scipy import signal

def generate_profile(height, width, type="triangle"):
    def triangle_hump_generator(position):
        # position vs height

        return (height * 0.5 *
                (signal.sawtooth(2 * np.pi / width * position, 0.5) + 1.0))

    return triangle_hump_generator