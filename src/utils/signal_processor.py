"""
This file implements the signal processing operations such as filtering and scaling.
"""
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, lfilter

class SignalProcessor:
    def __init__(self, fs=360, window_size=360, bandpass = (0.5,40.0), filter_order=2, downsample_factor=None):
        self.fs = fs
        self.window_size = window_size
        self.lowcut, self.highcut = bandpass
        self.order = filter_order
        self.downsample_factor = downsample_factor

    def butter_bandpass_filter(self, signal):
        b, a = butter(self.order, [self.lowcut, self.highcut], fs=self.fs, btype="band")
        return lfilter(b, a, signal)

    def downsample(self, signal):
        if self.downsample_factor is None:
            return signal
        return signal[::self.downsample_factor]

    def segment_signal(self, signal):
        """
        divides signal into fixed length non-overlapping windows. keeps the last short segment.
        """
        segments = []
        for i in range(0, len(signal) - len(signal) % self.window_size, self.window_size):
            segments.append(signal[i:i+self.window_size])
        return segments

    def scale_segment(self, segment):
        scaler = MinMaxScaler()
        return scaler.fit_transform(segment.reshape(-1,1)).squeeze()

    def preprocess(self, signal):
        """
        Full pipeline of preprocessing. Returns list of segments.
        """
        filtered = self.butter_bandpass_filter(signal)
        downsampled = self.downsample(filtered)
        segmented = self.segment_signal(downsampled)
        scaled = [self.scale_segment(seg) for seg in segmented]
        return scaled


