import random
from typing import Any
import numpy as np
import json
from ImportStuff import *


class Language:
    LEARNABLE_ENCODINGS = 0
    REPEAT_ENCODINGS = 1

    def __init__(self, alphabet: set[str] | list[str], recogniser, generator, encoding_type=LEARNABLE_ENCODINGS, length_assertion: bool = True):
        self.alphabet = alphabet
        self.recogniser = recogniser
        self.generator = generator
        self.encoding_type = encoding_type
        self.length_assertion = length_assertion

    def __contains__(self, item):
        return self.recogniser(item)

    @staticmethod
    def _get_true_string_length(base_length):
        return base_length

    def generate(self, length, force_length=False):
        sequence = self.generator(length, force_length)
        assert len(sequence) == self._get_true_string_length(length), f"{len(sequence)}, {self._get_true_string_length(length)}"
        return sequence

    def create_dataset(self, language_name, file_path, dataset_size, string_lengths, overflow_factor=10, include_substrings=True):
        positive_examples = set()
        negative_examples = set()
        examples = {}

        print("Percentage filled: 0%, 0%", end='')
        prev_pos_percent = 0
        prev_neg_percent = 0
        while (len(positive_examples) + len(negative_examples) < overflow_factor * dataset_size) and not (len(positive_examples) >= dataset_size // 2 and len(negative_examples) >= dataset_size // 2):
            string = self.generate(string_lengths, not include_substrings)
            labels = []

            if include_substrings:
                for i in range(len(string) + 1):
                    sub_example = string[:i]
                    self.categorise_example(sub_example, positive_examples, negative_examples, labels)
            else:
                labels = [0 for _ in range(len(string))]
                self.categorise_example(string, positive_examples, negative_examples, labels)

            examples[str(string)] = (string, labels)

            pos_percent = round(len(positive_examples) / dataset_size * 100, 1)
            neg_percent = round(len(negative_examples) / dataset_size * 100, 1)

            if prev_neg_percent != neg_percent or prev_pos_percent != pos_percent:
                prev_neg_percent = neg_percent
                prev_pos_percent = pos_percent
                print(f"\rPercentage filled: {pos_percent}%, {neg_percent}%", end="")
        print()

        positive_examples = list(positive_examples)
        negative_examples = list(negative_examples)
        print(f"# unique positive examples = {len(positive_examples)}")
        print(f"# unique negative examples = {len(negative_examples)}")
        print(f"# base examples = {len(examples)}")

        if len(list(self.alphabet)) <= 256:
            sequence_dt = 'u1'
        elif len(list(self.alphabet)) <= 65536:
            sequence_dt = 'u2'
        else:
            sequence_dt = 'u4'

        max_sequence_length = self._get_true_string_length(string_lengths)
        dataset_dtype = np.dtype([('sequence', sequence_dt, max_sequence_length), ('labels', '?', max_sequence_length + 1)])
        np_dataset = np.zeros(len(examples), dataset_dtype)

        alphabet = list(self.alphabet)
        alphabet_map = {str(value): index for index, value in enumerate(alphabet)}
        for i, string_hash in enumerate(examples):
            string, labelling = examples[string_hash]
            for j in range(len(string)):
                np_dataset[i]['sequence'][j] = alphabet_map[str(string[j])]
            np_dataset[i]['labels'][:len(string) + 1] = labelling

        np.save(file_path, np_dataset)

        self.update_languages_json(language_name, file_path, include_substrings)

        print("Dataset created.")

    def update_languages_json(self, dataset_name, file_path, include_substrings):
        with open(os.path.join(ROOT_DIR, LANGUAGES_FILE)) as json_data:
            current_languages = json.load(json_data)

        current_languages[dataset_name] = {"alphabet_size": len(self.alphabet), "path": file_path, "include_substrings": include_substrings,
                                           "encoding_type": self.encoding_type}

        with open(os.path.join(ROOT_DIR, LANGUAGES_FILE), 'w') as f:
            json.dump(current_languages, f)

    def categorise_example(self, example, positive_set, negative_set, labels):
        if example in self:
            positive_set.add(str(example))
            labels.append(True)
        else:
            negative_set.add(str(example))
            labels.append(False)
