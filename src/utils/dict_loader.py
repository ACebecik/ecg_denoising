"""
This file loads dataset from dicts, as pickled defaultdict files. Need to run a converter/splitter in the pipeline.
"""
import pickle
from pathlib import Path


class DictLoader:
    def __init__(self, base_load_path = "dictionaries/", ):
        root_dir = Path(__file__).resolve().parents[2]
        self.base_load_path = root_dir / base_load_path

    def load(self, training_task= None, snr = None):
        assert isinstance(snr, int) == 1 ,"Provided SNR is not integer!"
        assert training_task in ["classification","compensation"], "training_task must be classification or compensation!"

        if training_task == "classification":
            dataset_class = "noisy"
            data_load_path = self.base_load_path / f"{dataset_class}_data_snr_{snr}.pkl"
            label_load_path = self.base_load_path / f"labels_data_snr_{snr}.pkl"
            with open(label_load_path, "rb") as f:
                data_y = pickle.load(f)
            with open(data_load_path, "rb") as f:
                data_X = pickle.load(f)

        elif training_task == "compensation":
            dataset_class = "noisy"
            data_load_path = self.base_load_path / f"{dataset_class}_data_snr_{snr}.pkl"
            with open(data_load_path, "rb") as f:
                data_X = pickle.load(f)
            dataset_class = "reference"
            data_load_path = self.base_load_path / f"{dataset_class}_data_snr_{snr}.pkl"
            with open(data_load_path, "rb") as f:
                data_y = pickle.load(f)

        return data_X, data_y

if __name__ == "__main__":
    loader = DictLoader()
    X, y = loader.load(training_task="classification",snr=1)
    print(X.keys())