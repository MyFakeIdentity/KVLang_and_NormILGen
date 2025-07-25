import random
from Languages.Language import Language
import math
import re
import numpy as np


class KVLang(Language):
    def __init__(self, num_keys: int = 8, value_length: int = 4, binary_key_representation: bool = True):
        self.num_keys = num_keys
        self.binary_key_representation = binary_key_representation
        self.value_length = value_length

        alphabet = {"0", "1"}

        if self.binary_key_representation:
            self.key_length = math.ceil(math.log2(self.num_keys))

        else:
            self.key_length = 1
            for i in range(self.num_keys):
                alphabet.add(f"k{i}")

        self.section_length = self.key_length + self.value_length

        def recogniser_function(string):
            seen_pairs = {}

            section_count = math.ceil(len(string) / self.section_length)

            for section_num in range(section_count):
                section_start = self.section_length * section_num

                key = string[section_start:section_start+self.key_length]
                value = string[section_start+self.key_length:section_start+self.section_length]

                key_str = str(key)

                if key_str in seen_pairs:
                    seen_value = seen_pairs[key_str]
                    for v1, v2 in zip(seen_value, value):
                        if v1 != v2:
                            return False
                else:
                    seen_pairs[key_str] = value

            return True

        def generator_function(wanted_length, _):
            section_count = math.ceil(wanted_length / self.section_length)

            rand_val = random.random()
            if rand_val <= 0.5:
                corrupted_section_prob = 0
                value_corruption_prob = 0
            else:
                corrupted_section_prob = random.random()
                value_corruption_prob = random.random()

            sequence = []
            true_pairs = {}

            for _ in range(section_count):
                chosen_key = random.randint(0, self.num_keys - 1)

                if self.binary_key_representation:
                    key = []
                    for n in range(self.key_length):
                        key.append(str(chosen_key % 2))
                        chosen_key //= 2
                    key = list(reversed(key))
                else:
                    key = [f"k{chosen_key}"]

                key_str = str(key)

                if key_str in true_pairs:
                    value = true_pairs[key_str]

                    if random.random() <= corrupted_section_prob:
                        other_map = {"0": "1", "1": "0"}

                        for n in range(self.value_length):
                            if random.random() <= value_corruption_prob:
                                value[n] = other_map[value[n]]

                else:
                    value = [str(random.randint(0, 1)) for _ in range(self.value_length)]
                    true_pairs[key_str] = value

                sequence += key + value

            return sequence[:wanted_length]

        super().__init__(alphabet, recogniser_function, generator_function, length_assertion=False)


def print_sequence(sequence, k_size, v_size):
    s_size = k_size + v_size
    section_count = math.ceil(len(sequence) / s_size)

    string = []

    for section_num in range(section_count):
        section_start = section_num * s_size

        key = sequence[section_start:section_start+k_size]
        value = sequence[section_start+k_size:section_start+s_size]

        string.append(f"{''.join(map(str, key))}-{''.join(map(str, value))}")

    print(" ".join(string))


if __name__ == "__main__":
    key_count = 8
    value_length = 1

    test = KVLang(num_keys=key_count, value_length=value_length, binary_key_representation=False)

    test.create_dataset(f'Key-Value Up {key_count}_{value_length} len 50', f'Datasets/kv_up_{key_count}_{value_length}_regular', 2_000_000, 50, overflow_factor=1000)
    test.create_dataset(f'Key-Value Up {key_count}_{value_length} len 100', f'Datasets/kv_up_{key_count}_{value_length}_generalisation_100', 25_000, 100, include_substrings=False, overflow_factor=1000)
    test.create_dataset(f'Key-Value Up {key_count}_{value_length} len 250', f'Datasets/kv_up_{key_count}_{value_length}_generalisation_250', 25_000, 250, include_substrings=False, overflow_factor=1000)

    dataset = np.load(f'Datasets/kv_up_{key_count}_{value_length}_regular.npy')
    for i in range(5):
        print(dataset['labels'][i, ::(value_length + 1)])
        print_sequence(dataset['sequence'][i], 1, value_length)
