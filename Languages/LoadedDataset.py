from __future__ import annotations
import os
import random
import json
from ImportStuff import *
from Dataset import StringDataset
from Languages.Language import Language


class LoadedDataset:
    def __init__(self, dataset_name, alphabet_size, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, encoding_type):
        self.name: str = dataset_name

        self.alphabet_size: int = alphabet_size

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None

        self.encoding_type = encoding_type

    def set_batch_size(self, batch_size):
        if len(self.train_dataset) > 0:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False)
        if len(self.val_dataset) > 0:
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True)
        if len(self.test_dataset) > 0:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)

    @staticmethod
    def load(language_name, training_weight=10, validation_weight=1, testing_weight=1, random_orderings=False) -> LoadedDataset:
        with open(os.path.join(ROOT_DIR, LANGUAGES_FILE)) as json_data:
            current_languages = json.load(json_data)

        language = current_languages[language_name]
        alphabet_size = language["alphabet_size"]
        file_path = os.path.join("Languages", language["path"] + ".npy")
        include_substrings = language["include_substrings"]

        if "encoding_type" in language:
            encoding_type = language["encoding_type"]
        else:
            encoding_type = Language.LEARNABLE_ENCODINGS

        train_dataset, val_dataset, test_dataset = StringDataset.create_datasets(file_path, training_weight, validation_weight, testing_weight,
                                                                                 include_substrings=include_substrings, random_orderings=random_orderings)
        return LoadedDataset(language_name, alphabet_size, train_dataset, val_dataset, test_dataset, encoding_type)
