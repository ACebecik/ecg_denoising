"""
This file implements the pipeline class for the preprocessing and saving of the ECG datasets.
"""
from tqdm import tqdm
from collections import defaultdict
from src.data.base_reader import  BaseDatasetReader
from src.utils.data_saver import DataSaver
from src.utils.label_generator import LabelGenerator
from src.utils.noise_injector import NoiseInjector
from src.utils.signal_processor import SignalProcessor

class EcgPipeline:
    def __init__(self, reader:BaseDatasetReader,
                 noise_src=None, seed=None, snr_levels=[5,1],
                 window_size=1024, bandpass=(0.5, 40.0), filter_order=2, downsample_factor=None,
                 tolerance = 5,
                 base_save_path= "dictionaries/" ):

        self.reader = reader
        self.fs = reader.get_sampling_rate()
        self.injector = NoiseInjector(noise_src=noise_src, seed=seed)
        self.snr_levels = snr_levels
        self.processor = SignalProcessor(fs = self.fs, window_size=window_size, bandpass=bandpass,
                                               filter_order=filter_order, downsample_factor=downsample_factor)
        self.labeler = LabelGenerator(fs=self.fs, tolerance=tolerance)
        self.saver = DataSaver(base_save_path=base_save_path)

    def run(self):
        signals = self.reader.load_signals()
        pids = self.reader.get_patient_ids()

        for level in self.snr_levels:
            noisy_data = defaultdict(list)
            labels_to_use = defaultdict(list)
            reference_data = defaultdict(list)

            for pid in tqdm(pids, desc="Processing patient records..."):
                clean_signal = signals[pid]
                injected_signal = self.injector.inject_noise(clean_signal=clean_signal, snr_db=level)

                clean_segments = self.processor.preprocess(clean_signal)
                noisy_segments = self.processor.preprocess(injected_signal)

                for clean_seg, noisy_seg in zip(clean_segments, noisy_segments):
                    label = self.labeler.generate_label(clean_seg, noisy_seg)
                    if label is not None:
                        noisy_data[pid].append(noisy_seg)
                        labels_to_use[pid].append(label)
                        reference_data[pid].append(clean_seg)

                self.saver.save_to_dicts(reference_data=reference_data,
                                         noisy_data=noisy_data,
                                         labels_data=labels_to_use,
                                         snr_level=level
                                         )

