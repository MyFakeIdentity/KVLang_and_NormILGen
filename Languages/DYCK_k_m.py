import random
from AutomataClasses.Automata import Automata
from AutomataClasses.FlipFlop import FlipFlop
from AutomataClasses.Network import Network
from AutomataClasses.Functions import Function, Identity
from AutomataClasses.PDFA import PDFA
from Languages.Language import Language
import math
import re
import numpy as np


TARGET_MAP = {"High": "Set", "Low": "Reset"}
FLIP_MAP = {"High": "Reset", "Low": "Set"}


class BinaryCounterMapper:
    @staticmethod
    def apply(input_list):
        current_state = input_list[0]
        input_token = input_list[1]

        if input_token == "(":
            # Increment depth counter.

            all_high = True
            for item in input_list[2:]:
                if item != "High":
                    all_high = False
                    break

            if all_high:
                return FLIP_MAP[current_state]

        else:
            # Decrement depth counter.

            all_low = True
            for item in input_list[2:]:
                if item != "Low":
                    all_low = False
                    break

            if all_low:
                return FLIP_MAP[current_state]

        return "Read"


class DepthReachedMapper:
    def __init__(self, max_depth):
        bit_depth = int(math.ceil(math.log2(max_depth)))

        self.binary_encoding = []

        value = max_depth
        for _ in range(bit_depth):
            self.binary_encoding.append(value)
            value >>= 1

        self.binary_encoding = list(reversed(self.binary_encoding))

    def apply(self, input_list):
        current_state, input_token = input_list[0], input_list[1]

        # Check if depth is decreasing below 0.
        if input_token == ")":
            all_low = True
            for item in input_list[2:]:
                if item != "Low":
                    all_low = False
                    break

            if all_low:
                return "Set"

            return "Read"

        # Check if depth is increasing above max_depth.
        else:
            assert len(self.binary_encoding) == len(input_list[2:])

            for i, item in enumerate(input_list[2:]):
                # New depth != max_depth + 1
                if item != self.binary_encoding[i]:
                    return "Read"

            # New depth == max_depth + 1
            return "Set"


class ResultMapper:
    @staticmethod
    def apply(input_list):
        current_state, input_token, too_deep = input_list[0], input_list[1], input_list[-1]
        binary_count = input_list[2:-1]

        # Sentence went too deep to be in language.
        if too_deep == "High":
            return "Reset"

        # Sentence is in language if previous count was 1 and is decreasing to 0.
        elif input_token == ")" and binary_count[0] == "High":
            for item in binary_count[1:]:
                if item == "High":
                    return "Reset"
            return "Set"

        return "Reset"


class DYCK_k_m(Language):
    def __init__(self, m: int):
        alphabet = {"(", ")"}

        # Record current depth.
        # If depth gets to high always return false.
        # If depth ends at 0 return True.
        # Otherwise, return False.

        bit_depth = int(math.ceil(math.log2(m)))

        # [bits containing depth in base 2, depth reached > m, result]
        semi_automata = [FlipFlop() for _ in range(bit_depth + 2)]
        semi_automata[-1].initial_state = "High"
        input_functions = [BinaryCounterMapper() for _ in range(bit_depth)] + [DepthReachedMapper(m), ResultMapper()]

        network = Network(alphabet, input_functions, semi_automata, True)
        recogniser = Automata(network, Identity(), Function({"Low": False, "High": True}))

        pdfa = PDFA()
        depth_0 = pdfa.add_node([("(", 0.8), (")", 0.2)], True)
        depth_1 = pdfa.add_node([("(", 0.5), (")", 0.5)])
        depth_2 = pdfa.add_node([("(", 0.33), (")", 0.66)])
        depth_ge_3 = pdfa.add_node([("(", 0.2), (")", 0.8)])
        pdfa.add_edge(depth_0, "(", depth_1)
        pdfa.add_edge(depth_0, ")", depth_0)
        pdfa.add_edge(depth_1, "(", depth_2)
        pdfa.add_edge(depth_1, ")", depth_0)
        pdfa.add_edge(depth_2, "(", depth_ge_3)
        pdfa.add_edge(depth_2, ")", depth_1)
        pdfa.add_edge(depth_ge_3, "(", depth_ge_3)
        pdfa.add_edge(depth_ge_3, ")", depth_2)

        def recogniser_function(string):
            recogniser.reset()
            for token in string:
                recogniser.update(token)
            return recogniser.get_state()

        def recogniser_function2(string):
            current_depth = 0

            for token in string:
                if token == "(":
                    current_depth += 1
                    if current_depth > m:
                        break
                else:
                    current_depth -= 1
                    if current_depth < 0:
                        break

            return current_depth == 0

        def generator_function(wanted_length):
            string = ""
            generator = pdfa.generate_string()
            for _ in range(wanted_length):
                string += next(generator)
            return string

        def generator_function2(wanted_length, _):
            error_prob = random.random() / wanted_length * 2
            current_depth = 0
            string = ""

            for _ in range(wanted_length):
                if current_depth == 0:
                    if random.random() <= error_prob:
                        string += ")"
                    else:
                        string += "("
                elif current_depth == m:
                    if random.random() <= error_prob:
                        string += "("
                    else:
                        string += ")"
                else:
                    string += ["(", ")"][random.randint(0, 1)]

                if string[-1] == "(":
                    current_depth += 1
                else:
                    current_depth -= 1

            return string

        super().__init__(alphabet, recogniser_function2, generator_function2)


if __name__ == "__main__":
    test = DYCK_k_m(5)

    # regular_expression = []
    # for depth in range(6):
    #     regular_expression.append("(" * depth + ")" * depth)
    # regular_expression = ("|".join(regular_expression))[1:]
    # print(regular_expression)
    # regular_expression = re.compile(regular_expression)

    test.create_dataset('dyck_k_5 len 25', 'Datasets/dyck_k_5_regular', 250_000, 25)
    test.create_dataset('dyck_k_5 len 50', 'Datasets/dyck_k_5_generalisation_50', 25_000, 50, include_substrings=False)
    test.create_dataset('dyck_k_5 len 100', 'Datasets/dyck_k_5_generalisation_100', 25_000, 100, include_substrings=False)
    test.create_dataset('dyck_k_5 len 250', 'Datasets/dyck_k_5_generalisation_250', 25_000, 250, include_substrings=False)
    test.create_dataset('dyck_k_5 len 500', 'Datasets/dyck_k_5_generalisation_500', 25_000, 500, include_substrings=False)
    # dataset = np.load('Datasets/dyck_k_5_generalisation.npy')
    # print(dataset['sequence'][:10])
    # print(dataset['labels'][:10])

    # for _ in range(20):
    #     string = test.generate(10)
    #     # re_match = regular_expression.fullmatch(string) is not None
    #     # assert re_match == (string in test), f"{string}, {re_match}, {string in test}"
    #     print(string, string in test)
