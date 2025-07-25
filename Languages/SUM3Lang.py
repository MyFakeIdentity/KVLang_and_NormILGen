import random
from Languages.Language import Language
import math
import re
import numpy as np


class SUM3(Language):
    def __init__(self, max_abs_value: int):
        alphabet = [f"{value}" for value in range(-max_abs_value, max_abs_value + 1)]

        def recogniser_function_slow(string):
            for x in range(len(string)):
                for y in range(x + 1, len(string)):
                    for z in range(y + 1, len(string)):
                        v1 = string[x]
                        v2 = string[y]
                        v3 = string[z]

                        if v1 + v2 + v3 == 0:
                            return True

            return False

        def recogniser_function(string):
            values: dict[int, set[int]] = {}

            for z in range(len(string)):
                val = string[z]

                if val not in values:
                    values[val] = set()

                values[val].add(z)

            for x in range(len(string)):
                for y in range(x + 1, len(string)):
                    v1 = string[x]
                    v2 = string[y]

                    remainder = -(v1 + v2)

                    if remainder not in values:
                        continue

                    possible_zs = values[remainder]

                    if possible_zs.issubset({x, y}):
                        continue

                    # assert recogniser_function_slow(string), string
                    return True

            # assert not recogniser_function_slow(string), string
            return False

        def generator_function(wanted_length, force_length):
            target_length = wanted_length

            string = []
            for _ in range(target_length):
                string.append(random.randint(-max_abs_value, max_abs_value))

            if random.randint(0, 1) == 1 and target_length >= 3:
                first, second, third = 0, 0, 0
                first_pos, second_pos, third_pos = 0, 0, 0

                while True:
                    first = random.randint(-max_abs_value, max_abs_value)
                    second = random.randint(-max_abs_value, max_abs_value)

                    sum_val = first + second

                    if not -max_abs_value <= sum_val <= max_abs_value:
                        continue

                    third = -sum_val
                    break

                while first_pos == second_pos or second_pos == third_pos or third_pos == first_pos:
                    first_pos = random.randint(0, target_length - 1)
                    second_pos = random.randint(0, target_length - 1)
                    third_pos = random.randint(0, target_length - 1)

                string[first_pos] = first
                string[second_pos] = second
                string[third_pos] = third

            return string

        super().__init__(alphabet, recogniser_function, generator_function,
                         [Language.REPEAT_ENCODINGS, [value / max_abs_value for value in range(-max_abs_value, max_abs_value + 1)]])


if __name__ == "__main__":
    test = SUM3(30_000)

    test.create_dataset('SUM3 25 len 25', 'Datasets/SUM3_25_regular', 250_000, 25, include_substrings=False)
    test.create_dataset('SUM3 25 len 50', 'Datasets/SUM3_25_generalisation_50', 25_000, 50, include_substrings=False)
    test.create_dataset('SUM3 25 len 100', 'Datasets/SUM3_25_generalisation_100', 25_000, 100, include_substrings=False)
    # test.create_dataset('SUM3 25 len 250', 'Datasets/SUM3_25_generalisation_250', 25_000, 250, include_substrings=False)
    # test.create_dataset('SUM3 25 len 500', 'Datasets/SUM3_25_generalisation_500', 25_000, 500, include_substrings=False)
    dataset = np.load('Datasets/SUM3_25_regular.npy')
    row_1 = dataset['sequence'][0][:9].astype(np.int32) - 100
    print(dataset['sequence'][:10].astype(np.int32) - 100)
    print(row_1 in test)
    print(dataset['labels'][:10])
