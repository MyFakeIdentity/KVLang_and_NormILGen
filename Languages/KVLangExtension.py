import random
from Languages.Language import Language
import math
import re
import numpy as np


class KVLang(Language):
    def __init__(self, num_keys: list[int] = (8,), value_length: int = 4):
        self.num_keys = num_keys
        self.value_length = value_length

        alphabet = {"0", "1"}

        self.keys_length = len(self.num_keys)
        for i, key_selection in enumerate(self.num_keys):
            for key_value in range(key_selection):
                alphabet.add(f"k{i}_{key_value}")

        self.section_length = self.keys_length + self.value_length

        def group_sections(string):
            sections = []

            for section_num in range(len(string) // self.section_length):
                start = section_num * self.section_length
                keys_end = start+self.keys_length
                sections.append((string[start:keys_end], "".join(string[keys_end:start+self.section_length])))

            return sections

        def ungroup_sections(sequence):
            string = []

            for keys, value in sequence:
                string += keys
                string += list(value)

            return string

        def recogniser_function(string):
            reduced_sequence = group_sections(string)

            for key in range(len(self.num_keys) - 1):
                new_reduced_sequence = []
                seen_keys = set()

                for keys, value in reduced_sequence:
                    if keys[0] not in seen_keys:
                        new_reduced_sequence.append((keys[1:], value))

                reduced_sequence = new_reduced_sequence

            string = ungroup_sections(reduced_sequence)
            seen_pairs = {}

            section_count = int(len(string) / self.section_length)

            for section_num in range(section_count):
                section_start = self.section_length * section_num

                key = string[section_start:section_start+self.keys_length]
                value = string[section_start+self.keys_length:section_start + self.section_length]

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
                chosen_keys = [random.randint(0, self.num_keys[j] - 1) for j in range(len(self.num_keys))]

                keys = [f"k{j}_{chosen_key}" for j, chosen_key in enumerate(chosen_keys)]

                if keys[-1] in true_pairs:
                    value = true_pairs[keys[-1]]

                    if random.random() <= corrupted_section_prob:
                        other_map = {"0": "1", "1": "0"}

                        for n in range(self.value_length):
                            if random.random() <= value_corruption_prob:
                                value[n] = other_map[value[n]]

                else:
                    value = [str(random.randint(0, 1)) for _ in range(self.value_length)]
                    true_pairs[keys[-1]] = value

                sequence += keys + value

            return sequence[:wanted_length]

        super().__init__(alphabet, recogniser_function, generator_function, length_assertion=False)


def print_sequence(sequence, num_keys, v_size):
    section_length = num_keys + v_size

    def group_sections(string):
        sections = []

        for section_num in range(math.ceil(len(string) / section_length)):
            start = section_num * section_length
            keys_end = start + num_keys
            sections.append((string[start:keys_end], "".join(map(str, string[keys_end:start+section_length]))))

        return sections

    sections = group_sections(sequence)

    string = []

    for keys, value in sections:
        string.append(f"({','.join(map(str, keys))})-{value}")

    print(" ".join(string))


if __name__ == "__main__":
    key_counts = [8, 4, 2]
    value_length = 1

    test = KVLang(num_keys=key_counts, value_length=value_length)

    lang_string = f"{'_'.join(map(str, key_counts))}_{value_length}"

    test.create_dataset(f'Key-Value {lang_string} len 100', f'Datasets/kv_{lang_string}_regular', 2_000_000, 100, overflow_factor=1000)
    test.create_dataset(f'Key-Value {lang_string} len 250', f'Datasets/kv_{lang_string}_generalisation_250', 25_000, 250, include_substrings=False, overflow_factor=1000)

    # dataset = np.load(f'Datasets/kv_{lang_string}_regular.npy')
    # for entry in range(25):
    #     size = len(key_counts) + value_length
    #     print(dataset['labels'][entry, ::size])
    #     print_sequence(dataset['sequence'][entry], len(key_counts), value_length)
    # print(dataset['labels'][3, ::5])
    # print_sequence(dataset['sequence'][3], 1, 4)
    # print(dataset['labels'][4, ::5])
    # print_sequence(dataset['sequence'][4], 1, 4)
