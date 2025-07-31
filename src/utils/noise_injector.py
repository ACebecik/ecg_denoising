"""
This file injects noise to clean ECG segments. Supports MIT electrode motion noise and gaussian noise.
"""
import numpy as np

class NoiseInjector:
    def __init__(self, noise_src=None, seed=None):
        """
        :param noise_src: ndarray, in case of the loaded em noise from mit db.
        :param seed: optional for the reproducibility.
        """
        self.noise_src = noise_src
        if seed is not None:
            np.random.seed(seed)

    def _generate_noise(self, shape):
        """Gaussian noise if no noise is passed to injector."""
        return np.random.normal(0,1, size=shape)

    def _scale_noise_to_snr(self, signal, noise, snr_db):
        signal_power = np.mean(signal ** 2)
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        current_noise_power = np.mean(noise ** 2)
        scaling_factor = np.sqrt(target_noise_power / (current_noise_power + 1e-12))
        return noise * scaling_factor

    def inject_noise(self, clean_signal, snr_db):
        """
        Adds the noise to the clean signal and returns it.
        """
        if isinstance(self.noise_src, np.ndarray):
            noise = np.resize(self.noise_src, clean_signal.shape)
        else:
            noise = self._generate_noise(clean_signal.shape)

        scaled_noise = self._scale_noise_to_snr(clean_signal, noise, snr_db)
        return clean_signal + scaled_noise
