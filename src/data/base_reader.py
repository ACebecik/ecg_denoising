"""
This file implements an interface using an ABC for compatibility across different ECG datasets.
"""

from abc import ABC, abstractmethod

class BaseDatasetReader(ABC):
    @abstractmethod
    def load_signals(self):
        pass

    @abstractmethod
    def get_patient_ids(self):
        pass

    @abstractmethod
    def get_sampling_rate(self):
        pass
