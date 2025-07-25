import random
from Languages.Language import Language
import math
import re
import numpy as np


class CopyLang(Language):
    def __init__(self, prefix_size: int):
        alphabet = {"0", "1"}

        def recogniser_function(string):
            return len(string) >= prefix_size * 2 and string[:prefix_size] == string[-prefix_size:]

        def generator_function(wanted_length, force_length):
            if force_length:
                target_length = wanted_length
            else:
                target_length = random.randint(0, wanted_length)

            if target_length < prefix_size * 2:
                string = ""
                for _ in range(wanted_length):
                    string += str(random.randint(0, 1))
                return string

            start = ""
            for _ in range(target_length - prefix_size):
                start += str(random.randint(0, 1))
            prefix = start[:prefix_size]

            other_map = {"0": "1", "1": "0"}

            if random.random() < 0.5:
                suffix = prefix
            else:
                error_prob = random.random()
                suffix = ""
                for i in range(prefix_size):
                    if random.random() < error_prob:
                        suffix += other_map[prefix[i]]
                    else:
                        suffix += prefix[i]

            end = "".join([str(random.randint(0, 1)) for _ in range(wanted_length - target_length)])
            assert len(start + suffix + end) == wanted_length

            return start + suffix + end

        super().__init__(alphabet, recogniser_function, generator_function)


if __name__ == "__main__":
    test = CopyLang(5)

    test.create_dataset('Test Dataset', 'Datasets/test_dataset', 20_000, 25)

    test.create_dataset('Memory Copy 5 len 25', 'Datasets/copy_5_regular', 250_000, 25)
    test.create_dataset('Memory Copy 5 len 50', 'Datasets/copy_5_generalisation_50', 25_000, 50, include_substrings=False)
    test.create_dataset('Memory Copy 5 len 100', 'Datasets/copy_5_generalisation_100', 25_000, 100, include_substrings=False)
    test.create_dataset('Memory Copy 5 len 250', 'Datasets/copy_5_generalisation_250', 25_000, 250, include_substrings=False)
    test.create_dataset('Memory Copy 5 len 500', 'Datasets/copy_5_generalisation_500', 25_000, 500, include_substrings=False)
    dataset = np.load('Datasets/copy_5_regular.npy')
    print(dataset['sequence'][:10])
    print(dataset['labels'][:10])
