import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

def fractal_dimension_fft(Z, threshold=0.42):
    assert len(Z.shape) == 2

    # Apply threshold to create binary image
    binary_image = (Z < threshold)

    # Flatten the binary image before calculating the power spectral density
    flat_image = binary_image.flatten()

    # Calculate the power spectral density using FFT
    f, Pxx = periodogram(flat_image, fs=1.0, scaling='density')

    # Trim arrays to have the same length
    min_len = min(len(f), len(Pxx))
    f = f[:min_len]
    Pxx = Pxx[:min_len]

    # Fit a line to the log-log plot of the power spectral density
    coeffs = np.polyfit(np.log(f[1:]), np.log(Pxx[1:]), 1)

    # The fractal dimension is related to the slope of the line
    fractal_dimension = -coeffs[0]

    return fractal_dimension

# Load image using imageio.imread (replace with appropriate method based on your SciPy version)
I = imageio.imread("savegambar/pnemunia/enlarged_4x_PNE2.png")
I_gray = np.mean(I, axis=-1) / 256.0

# Compute fractal dimension using FFT
fractal_dim_fft = fractal_dimension_fft(I_gray)

print("Fractal dimension (computed using FFT):", fractal_dim_fft)
