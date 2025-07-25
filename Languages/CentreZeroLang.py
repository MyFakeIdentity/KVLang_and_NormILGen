import random
from Languages.Language import Language
import math
import re
import numpy as np


class CentreZeroLang(Language):
    def __init__(self):
        alphabet = {"0", "1"}

        def recogniser_function(string):
            if len(string) % 3 != 0:
                return False

            third = len(string) // 3

            start = string[:third]
            centre = string[third:third*2]
            end = string[third*2:]

            if start != end:
                return False

            for char in centre:
                if char != "0":
                    return False

            return True

        def generator_function(wanted_length, force_length):
            if force_length:
                target_length = wanted_length
            else:
                target_length = random.randint(0, wanted_length)

            rand_val = random.random()
            if rand_val <= 0.3:
                corruption_centre_prob = 0
                corruption_end_prob = 0
            elif rand_val <= 0.6:
                corruption_centre_prob = 0
                corruption_end_prob = random.random()
            else:
                corruption_centre_prob = random.random()
                corruption_end_prob = random.random()

            third_length = target_length // 3

            start = ""
            for _ in range(third_length):
                start += str(random.randint(0, 1))

            centre = ""
            for _ in range(third_length):
                if random.random() < corruption_centre_prob:
                    centre += "1"
                else:
                    centre += "0"

            other_map = {"0": "1", "1": "0"}

            end = ""
            for i in range(third_length):
                if random.random() < corruption_end_prob:
                    end += other_map[start[i]]
                else:
                    end += start[i]

            whole = start + centre + end

            if len(whole) != target_length:
                whole += str(random.randint(0, 1))
                if len(whole) != target_length:
                    whole += str(random.randint(0, 1))

            while len(whole) < wanted_length:
                whole += str(random.randint(0, 1))

            return whole

        super().__init__(alphabet, recogniser_function, generator_function)


if __name__ == "__main__":
    test = CentreZeroLang()

    # Length must be a multiple of 3 for generalisation datasets (due to no sub-length examples)!!!
    test.create_dataset('Zero Centre len 75', 'Datasets/centre_0_regular', 250_000, 75, overflow_factor=1000)
    test.create_dataset('Zero Centre len 150', 'Datasets/centre_0_generalisation_150', 25_000, 150, include_substrings=False, overflow_factor=1000)
    test.create_dataset('Zero Centre len 250', 'Datasets/centre_0_generalisation_252', 25_000, 252, include_substrings=False, overflow_factor=1000)
    test.create_dataset('Zero Centre len 500', 'Datasets/centre_0_generalisation_501', 25_000, 501, include_substrings=False, overflow_factor=1000)
    dataset = np.load('Datasets/centre_0_regular.npy')
    print(dataset['sequence'][:10])
    print(dataset['labels'][:10])
