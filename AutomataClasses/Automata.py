from AutomataClasses.SemiAutomaton import SemiAutomaton
from AutomataClasses.Functions import Function


class Automata:
    def __init__(self, semi_automata: SemiAutomaton, input_function: Function, output_function: Function):
        self.semi_automata = semi_automata
        self.input_function = input_function
        self.output_function = output_function

    def get_state(self):
        state = self.semi_automata.get_state()
        return self.output_function.apply(state)

    def update(self, token: str) -> bool:
        parsed_token = self.input_function.apply(token)
        self.semi_automata.update(parsed_token)
        return self.get_state()

    def reset(self):
        self.semi_automata.reset()
