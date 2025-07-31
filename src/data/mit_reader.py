"""
This file implements the record reader for the MIT-BIH dataset, inheriting the base reader.
"""
from .base_reader import BaseDatasetReader
import os
import wfdb
import matplotlib.pyplot as plt

class MitReader(BaseDatasetReader):

    def __init__(self, base_path):
        self.db_path = os.path.join(base_path, "physionet.org/files/mitdb/1.0.0")
        self.records_file = os.path.join(self.db_path, "RECORDS")
        self.fs = 360

    def load_signals(self):
        self.signals = {}
        with open(self.records_file, "r") as f:
            for name in f.read().splitlines():
                record = wfdb.rdrecord(os.path.join(self.db_path, name))
                channel_names = record.sig_name
                try:
                    mlii_channel_index = channel_names.index("MLII")
                    self.signals[name] = record.p_signal[:, mlii_channel_index]
                except Exception:
                    print(f"Record no:{name} does not have MLII channel. Skipping record.")
        return self.signals


    def get_patient_ids(self):
        with open(self.records_file, "r") as f:
            return self.signals.keys()

    def get_sampling_rate(self):
        return self.fs

if __name__ == "__main__":
    reader = MitReader(base_path="/Users/alperencebecik/Desktop/ecg_datasets")
    signals = reader.load_signals()
    ids = reader.get_patient_ids()

