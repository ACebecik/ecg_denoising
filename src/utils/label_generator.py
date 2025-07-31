"""
This file implements the labeling of the ECG segments into usable or unusable.
"""
import numpy as np
from wfdb.processing import xqrs_detect

class LabelGenerator:
    def __init__(self, fs=360, tolerance=5):
        self.fs = fs
        self.tolerance = tolerance

    def detect_peaks(self, segment):
        """
        Detects R-peaks using XQRS algorithm, returns a numpy array of peak indices.
        """
        try:
            return xqrs_detect(sig=segment, fs=self.fs, verbose=False)
        except Exception:
            return np.array([])

    def peaks_match(self, ref_peaks, noisy_peaks):
        """
        Checks if each reference peak has a matching peak in the test signal within a tolerance.
        Return False if one such peak exceeds tolerance.
        """
        for ref in ref_peaks:
            if not any(abs(ref-tested_peak) <= self.tolerance for tested_peak in noisy_peaks):
                return False
        return True

    def generate_label(self, clean_segment, noisy_segment):
        """
        Returns:
            1 if R-peaks match between clean and noisy segments,
            0 if they do not match,
            None if peak detection fails.
        """
        clean_peaks = self.detect_peaks(clean_segment)
        noisy_peaks = self.detect_peaks(noisy_segment)

        if len(clean_peaks) == 0 or len(noisy_peaks) == 0:
            return None

        return int(self.peaks_match(clean_peaks, noisy_peaks))



