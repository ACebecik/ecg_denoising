
"""
Main entry point for ECG classification training pipeline.
"""

from src.utils.dict_loader import DictLoader
from src.utils.compute_key_split import split_by_keys
from src.utils.data_stacker import stack_all_sets
from src.utils.dataloaders import create_dataloaders, create_dataset
from src.train.classifier_trainer import Trainer
from src.models.classification_model_cnn import CnnClassifier


def main():
    # 1. Load dataset from dictionaries
    loader = DictLoader()
    data_X, data_y = loader.load(training_task="classification", snr=1)

    # 2. Split into train, val, test keys
    train_keys, val_keys, test_keys = split_by_keys(data_X, data_y)

    # 3. Stack data into tensors
    train, val, test = stack_all_sets(
        data_X, data_y, train_keys, val_keys, test_keys
    )


    train_ds, val_ds, test_ds = create_dataset(train, val, test)

    # 4. Create TensorDatasets & DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds,
        val_ds,
        test_ds,
        batch_size=64
    )

    # 5. Initialize model
    model = CnnClassifier(window_size=360)

    # 6. Train
    trainer = Trainer(model, lr=0.001, early_stop_patience=5)
    trainer.train(train_loader, val_loader, no_epochs=20)

    # 7. Test best model
    trainer.test(test_loader)


if __name__ == "__main__":
    main()
