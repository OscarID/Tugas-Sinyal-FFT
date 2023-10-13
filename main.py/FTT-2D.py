import numpy as np
import matplotlib.pyplot as plt

# Define signal parameters
A = 1
n = 64
x, y = np.meshgrid(np.linspace(-A, A, n), np.linspace(-A, A, n))

# Define 2D signals
signal1_data = np.where(np.logical_and(-A/2 < x, x < A/2) & np.logical_and(-A/2 < y, y < A/2), 1, 0)
signal2_data = np.where(np.logical_and(-A < x, x < A) & np.logical_and(-A < y, y < A), 1, 0)
signal3_data = np.where(np.logical_and(-3*A < x, x < 3*A) & np.logical_and(-3*A < y, y < 3*A), 1, 0)

# Calculate 2D FFT using NumPy
output_1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal1_data)))
output_2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal2_data)))
output_3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(signal3_data)))

# Print Username
print ("Clearesta Frederika Oscar")
print ("5009211064")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(331)
plt.imshow(signal1_data, cmap='gray')
plt.title('Signal 1(A/2)')

plt.subplot(332)
plt.imshow(np.abs(output_1), cmap='viridis')
plt.title('FFT of Signal 1')

plt.subplot(333)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(signal1_data))), cmap='viridis')
plt.title('NumPy FFT of Signal 1')

plt.subplot(334)
plt.imshow(signal2_data, cmap='gray')
plt.title('Signal 2(A)')

plt.subplot(335)
plt.imshow(np.abs(output_2), cmap='viridis')
plt.title('FFT of Signal 2')

plt.subplot(336)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(signal2_data))), cmap='viridis')
plt.title('NumPy FFT of Signal 2')

plt.subplot(337)
plt.imshow(signal3_data, cmap='gray')
plt.title('Signal 3(3A)')

plt.subplot(338)
plt.imshow(np.abs(output_3), cmap='viridis')
plt.title('FFT of Signal 3')

plt.subplot(339)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(signal3_data))), cmap='viridis')
plt.title('NumPy FFT of Signal 3')

plt.tight_layout()
plt.show()
