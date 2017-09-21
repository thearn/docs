# For a signal sampled at 100 Hz, we want to construct a filter with a
# passband at 20-40 Hz, and stop bands at 0-10 Hz and 45-50 Hz. Note that
# this means that the behavior in the frequency ranges between those bands
# is unspecified and may overshoot.

from scipy import signal
fs = 100
bpass = signal.remez(72, [0, 10, 20, 40, 45, 50], [0, 1, 0], fs=fs)
freq, response = signal.freqz(bpass)

import matplotlib.pyplot as plt
plt.semilogy(0.5*fs*freq/np.pi, np.abs(response), 'b-')
plt.grid(alpha=0.25)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.show()
