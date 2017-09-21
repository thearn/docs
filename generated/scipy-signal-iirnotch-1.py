# Design and plot filter to remove the 60Hz component from a
# signal sampled at 200Hz, using a quality factor Q = 30

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

fs = 200.0  # Sample frequency (Hz)
f0 = 60.0  # Frequency to be removed from signal (Hz)
Q = 30.0  # Quality factor
w0 = f0/(fs/2)  # Normalized Frequency
# Design notch filter
b, a = signal.iirnotch(w0, Q)

# Frequency response
w, h = signal.freqz(b, a)
# Generate frequency axis
freq = w*fs/(2*np.pi)
# Plot
fig, ax = plt.subplots(2, 1, figsize=(8, 6))
ax[0].plot(freq, 20*np.log10(abs(h)), color='blue')
ax[0].set_title("Frequency Response")
ax[0].set_ylabel("Amplitude (dB)", color='blue')
ax[0].set_xlim([0, 100])
ax[0].set_ylim([-25, 10])
ax[0].grid()
ax[1].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='green')
ax[1].set_ylabel("Angle (degrees)", color='green')
ax[1].set_xlabel("Frequency (Hz)")
ax[1].set_xlim([0, 100])
ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[1].set_ylim([-90, 90])
ax[1].grid()
plt.show()
