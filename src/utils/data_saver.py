"""
This file implements the saver class for the labeled segments in the form of dictionaries.
"""
import os
import pickle

class DataSaver:
    def __init__(self, base_save_path="dictionaries/"):
        self.base_save_path = base_save_path
        os.makedirs(self.base_save_path, exist_ok=True)

    def save_pickle(self, obj, filename):
        saving_path = os.path.join(self.base_save_path, filename)
        with open(saving_path, "wb") as f:
            pickle.dump(obj, f)

    def save_to_dicts(self, reference_data, noisy_data, labels_data, snr_level=5):
        self.save_pickle(reference_data, f"reference_data_snr_{snr_level}.pkl")
        self.save_pickle(noisy_data, f"noisy_data_snr_{snr_level}.pkl")
        self.save_pickle(labels_data, f"labels_data_snr_{snr_level}.pkl")
