from __future__ import annotations
from dataclasses import dataclass
import random
import re


STAR_MAX_UNFOLD = 10
STAR_SUB_RANDOMNESS = 2
STAR_SUB_MIN_LENGTH = 3
MAX_RETURN_SIZE = 100


@dataclass()
class RegularExpression:
    def gen_rand(self, ideal_length: int, print_path: bool) -> list[int]:
        return []

    @staticmethod
    def gen_rand_expr(max_tokens: int, alphabet: list[int]) -> RegularExpression:
        if max_tokens <= 1:
            if random.randint(0, 4) != 0:
                return Letter(random.choice(alphabet))
            else:
                return Epsilon()

        chosen = random.randint(0, 100)

        if chosen < 30:
            return RegularExpression.gen_rand_union(max_tokens, alphabet)
        elif chosen < 60:
            return RegularExpression.gen_rand_star(max_tokens, alphabet)
        elif chosen < 90:
            return RegularExpression.gen_rand_concat(max_tokens, alphabet)
        else:
            if random.randint(0, 4) != 0:
                return Letter(random.choice(alphabet))
            else:
                return Epsilon()

    @staticmethod
    def gen_rand_union(max_tokens, alphabet):
        left = RegularExpression.gen_rand_expr(random.randint(1, max_tokens // 2), alphabet)
        right = RegularExpression.gen_rand_expr(random.randint(1, max_tokens // 2), alphabet)
        return Union(left, right)

    @staticmethod
    def gen_rand_star(max_tokens, alphabet):
        sub_expr = RegularExpression.gen_rand_expr(random.randint(1, max_tokens - 1), alphabet)
        return Star(sub_expr)

    @staticmethod
    def gen_rand_concat(max_tokens, alphabet):
        left = RegularExpression.gen_rand_expr(random.randint(1, max_tokens // 2), alphabet)
        right = RegularExpression.gen_rand_expr(random.randint(1, max_tokens // 2), alphabet)
        return Concatenate(left, right)

    @staticmethod
    def limit_set(values):
        if len(values) > MAX_RETURN_SIZE:
            return set(random.sample(list(values), MAX_RETURN_SIZE))
        return values

    def check_match(self, string: list[int]):
        return re.fullmatch(str(self), RegularExpression.format_value(string)) is not None

    def compile(self):
        return re.compile(self.get_re_string())

    @staticmethod
    def format_value(string):
        return "".join(map(str, string)).replace("ε", "")

    def get_possible_sizes(self, max_size: int) -> set[int]:
        return set()

    def get_all_examples(self, max_size: int) -> set[str]:
        pass

    def get_re_string(self) -> str:
        return


@dataclass()
class Union(RegularExpression):
    left: RegularExpression
    right: RegularExpression

    def __str__(self):
        return f"({self.left}|{self.right})"

    def gen_rand(self, ideal_length: int, print_path: bool) -> list[int]:
        if random.randint(0, 1) == 0:
            sub_string = self.left.gen_rand(ideal_length, print_path)
            if print_path:
                print(f"{self}: left = {sub_string}")
        else:
            sub_string = self.right.gen_rand(ideal_length, print_path)
            if print_path:
                print(f"{self}: right = {sub_string}")

        return sub_string

    def get_possible_sizes(self, max_size: int) -> set[int]:
        left_sizes = self.left.get_possible_sizes(max_size)
        right_sizes = self.right.get_possible_sizes(max_size)
        return left_sizes | right_sizes

    def get_all_examples(self, max_size: int) -> set[str]:
        left_examples = self.left.get_all_examples(max_size)
        right_examples = self.right.get_all_examples(max_size)
        return RegularExpression.limit_set(left_examples | right_examples)

    def get_re_string(self) -> str:
        return f"({self.left.get_re_string()}|{self.right.get_re_string()})"


@dataclass()
class Star(RegularExpression):
    sub_expr: RegularExpression

    def __str__(self):
        return f"({self.sub_expr})*"

    def get_re_string(self) -> str:
        return f"({self.sub_expr.get_re_string()})*"

    def gen_rand(self, ideal_length: int, print_path: bool) -> list[int]:
        result = []
        sub_results = []

        expected_count = min(ideal_length // STAR_SUB_MIN_LENGTH, STAR_MAX_UNFOLD)

        if expected_count == 0:
            if print_path:
                print(f"{self}: ")
            return []

        while random.randint(0, expected_count) != 0 and len(result) < ideal_length:
            sub_length = ideal_length / expected_count + random.randint(-STAR_SUB_RANDOMNESS, STAR_SUB_RANDOMNESS)
            sub_string = self.sub_expr.gen_rand(sub_length, print_path)
            result += sub_string
            sub_results.append(sub_string)

        if print_path:
            print(f"{self}: {expected_count} = {sub_results}")

        return result

    def get_possible_sizes(self, max_size: int) -> set[int]:
        sub_sizes = self.sub_expr.get_possible_sizes(max_size)

        result = {0}
        for repeats in range(STAR_MAX_UNFOLD):
            partial_result = set(result)

            for sub_value in partial_result:
                for value in sub_sizes:
                    result.add(sub_value + value)

        return result

    def get_all_examples(self, max_size: int) -> set[str]:
        sub_results = self.sub_expr.get_all_examples(max_size)

        result = {""}
        active = {""}

        while active != set():
            new_results = set()

            active = RegularExpression.limit_set(active)

            for partial_value in active:
                for sub_value in sub_results:
                    combined_value = partial_value + sub_value

                    if len(combined_value) < max_size and combined_value not in result:
                        new_results.add(combined_value)
                        result.add(combined_value)

            active = new_results

        return RegularExpression.limit_set(result)


@dataclass()
class Concatenate(RegularExpression):
    left: RegularExpression
    right: RegularExpression

    def __str__(self):
        return f"{self.left}{self.right}"

    def get_re_string(self) -> str:
        return f"{self.left.get_re_string()}{self.right.get_re_string()}"

    def gen_rand(self, ideal_length: int, print_path: bool) -> list[int]:
        left = self.left.gen_rand(ideal_length / 2, print_path)
        right = self.right.gen_rand(ideal_length - len(left), print_path)

        if print_path:
            print(f"{self}: {left}, {right}")

        return left + right

    def get_possible_sizes(self, max_size: int) -> set[int]:
        left_sizes = self.left.get_possible_sizes(max_size)
        right_sizes = self.right.get_possible_sizes(max_size)

        result = set()

        for left_value in left_sizes:
            for right_value in right_sizes:
                if left_value + right_value <= max_size:
                    continue
                result.add(left_value + right_value)

        return result

    def get_all_examples(self, max_size: int) -> set[str]:
        left_values = self.left.get_all_examples(max_size)
        right_values = self.right.get_all_examples(max_size)

        result = set()

        for left_value in left_values:
            for right_value in right_values:
                if len(left_value) + len(right_value) <= max_size:
                    result.add(left_value + right_value)

        return RegularExpression.limit_set(result)


@dataclass()
class Letter(RegularExpression):
    char: int

    def __str__(self):
        return f"{self.char}"

    def get_re_string(self) -> str:
        return f"{self.char}"

    def gen_rand(self, ideal_length: int, print_path: bool) -> list[int]:
        if print_path:
            print(f"{self}: {self.char}")
        return [self.char]

    def get_possible_sizes(self, max_size: int) -> set[int]:
        return {1}

    def get_all_examples(self, max_size: int) -> set[str]:
        return {str(self.char)}


@dataclass()
class Epsilon(RegularExpression):
    def __str__(self):
        return f"ε"

    def gen_rand(self, ideal_length: int, print_path: bool) -> list[int]:
        if print_path:
            print(f"{self}")
        return []

    def get_possible_sizes(self, max_size: int) -> set[int]:
        return {0}

    def get_all_examples(self, max_size: int) -> set[str]:
        return {""}

    def get_re_string(self):
        return ""


if __name__ == "__main__":
    for _ in range(10000):
        test = RegularExpression.gen_rand_expr(100, [0, 1, 2])
        all_results = sorted(list(test.get_all_examples(25)), key=len)
        print(str(test), len(all_results))
    quit()

    test = RegularExpression.gen_rand_expr(100, [0, 1, 2])
    print(test)
    print()
    all_results = sorted(list(test.get_all_examples(25)), key=len)

    for value in all_results:
        if value == "":
            print("ε")
        else:
            print(value)

    print(len(all_results))

    quit()
    print()
    for i in range(100):
        final_result = test.gen_rand(50, False)
        print(RegularExpression.format_value(final_result))
        print(test.check_match(final_result))
        print("length", len(final_result))
