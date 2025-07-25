import random

import torch
from ImportStuff import *
from RegularExpression import RegularExpression


class StringDataset(Dataset):
    def __init__(self, data, max_sequence_length, random_ordering=False):
        seen_examples = set()
        # [(pos, neg), ...]
        examples = [([], []) for _ in range(max_sequence_length + 1)]

        for sub_pos_examples, sub_neg_examples in data:
            for example, string in sub_pos_examples:
                if string not in seen_examples:
                    examples[len(example)][0].append(example)
                    seen_examples.add(string)
            for example, string in sub_neg_examples:
                if string not in seen_examples:
                    examples[len(example)][1].append(example)
                    seen_examples.add(string)

        size = 0
        for pos_examples, neg_examples in examples:
            random.shuffle(pos_examples)
            random.shuffle(neg_examples)
            size += min(len(pos_examples), len(neg_examples)) * 2

        xs_array = np.zeros((size, max_sequence_length), np.int32)
        lengths_array = np.zeros(size, np.int32)
        labels_array = np.zeros(size, np.int32)

        i = 0
        for pos_examples, neg_examples in examples:
            for pos, neg in zip(pos_examples, neg_examples):
                if random_ordering:
                    random.shuffle(pos)
                    random.shuffle(neg)

                xs_array[i * 2][:len(pos)] = pos
                lengths_array[i * 2] = len(pos)
                labels_array[i * 2] = True

                xs_array[i * 2 + 1][:len(neg)] = neg
                lengths_array[i * 2 + 1] = len(neg)
                labels_array[i * 2 + 1] = False

                i += 1

        '''
        if size != 0:
            true_counts = np.zeros(np.max(lengths_array) + 1, np.int64)
            counts = np.zeros(np.max(lengths_array) + 1, np.int64)
            np.add.at(true_counts, lengths_array, labels_array)
            np.add.at(counts, lengths_array, np.ones(size))
            average_guessable = np.nan_to_num(true_counts / counts)
            average_guessable[average_guessable < 0.5] = 1 - average_guessable[average_guessable < 0.5]
            average_correct = np.sum(average_guessable * counts)
            average_correct /= np.sum(counts)
            print(f"Minimum accuracy: {round(average_correct * 100, 1)}%.")

            self.get_frequency_analysis(xs_array, lengths_array, labels_array)
        '''

        self.xs = torch.tensor(xs_array, dtype=torch.int, device=DEVICE)
        self.lengths = torch.tensor(lengths_array, dtype=torch.int, device=DEVICE)
        self.ys = torch.tensor(labels_array, dtype=torch.float, device=DEVICE)

        print(f"Partial dataset created of size: {size}.")

    @staticmethod
    def get_frequency_analysis(xs_array, lengths_array, labels_array):
        alphabet_size = np.max(xs_array) + 1
        counts = np.zeros((len(xs_array), alphabet_size), np.float32)

        for i in range(len(xs_array)):
            np.add.at(counts[i], xs_array[i, :lengths_array[i]], 1 / lengths_array[i])

        best_accuracy = 0
        for direction in range(2):
            for threshold in range(10):
                threshold /= 10
                for char in range(alphabet_size):
                    meet_threshold = counts[:, char] > threshold if direction == 0 else counts[:, char] < threshold
                    correct = ~(meet_threshold ^ labels_array).astype(bool)
                    accuracy = np.average(correct.astype(np.float32))
                    best_accuracy = max(accuracy, best_accuracy)

        print(f"Alphabet frequency accuracy: {round(best_accuracy * 100, 1)}%.")

        quit()

    @staticmethod
    def create_datasets(file_path, training_weighting, validation_weighting, testing_weighting, include_substrings=True, random_orderings=False):
        data = np.load(os.path.join(ROOT_DIR, file_path))
        print(f"Dataset at {file_path} loaded.")

        max_sequence_length = len(data["sequence"][0])

        python_dataset = []
        for i in range(len(data)):
            sequence = data[i]["sequence"]
            labels = data[i]["labels"]

            pos_substrings = []
            neg_substrings = []

            string_sequence = list(map(str, sequence))
            current_string = ""

            if include_substrings:
                for j in range(max_sequence_length + 1):
                    if labels[j]:
                        pos_substrings.append((sequence[:j], current_string))
                    else:
                        neg_substrings.append((sequence[:j], current_string))

                    if j < max_sequence_length:
                        current_string += string_sequence[j] + ","

                random.shuffle(pos_substrings)
                random.shuffle(neg_substrings)

                size = min(len(pos_substrings), len(neg_substrings)) + 1
                python_dataset.append((pos_substrings[:size], neg_substrings[:size]))
            else:
                base_string = ",".join(string_sequence)

                if labels[-1]:
                    python_dataset.append(([(sequence, base_string)], []))
                else:
                    python_dataset.append(([], [(sequence, base_string)]))

        random.shuffle(python_dataset)

        total = training_weighting + validation_weighting + testing_weighting

        training_weighting /= total
        validation_weighting /= total
        testing_weighting /= total

        training_size = int(training_weighting * len(python_dataset))
        validation_size = int(validation_weighting * len(python_dataset))

        training_data = python_dataset[:training_size]
        validation_data = python_dataset[training_size:training_size+validation_size]
        testing_data = python_dataset[training_size+validation_size:]
        print(f"Dataset cut into: {len(training_data)}, {len(validation_data)}, {len(testing_data)}")

        training_dataset = StringDataset(training_data, max_sequence_length, random_ordering=random_orderings)
        validation_dataset = StringDataset(validation_data, max_sequence_length, random_ordering=random_orderings)
        testing_dataset = StringDataset(testing_data, max_sequence_length, random_ordering=random_orderings)
        print(f"Dataset passed to DEVICE.")

        return training_dataset, validation_dataset, testing_dataset

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.lengths[idx], self.ys[idx]
