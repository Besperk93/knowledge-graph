import random


class questionGenerator:

    def generate(self):
        self.generate_addition()
        self.generate_subtraction()
        self.generate_multiplication()
        self.generate_division()
        self.find_maximum()
        self.find_minimum()
        self.save_questions()

    def save_questions(self):
        with open(self.OUTPUT, 'w') as output:
            for question in self.QUESTIONS:
                output.write(f"{question} \n")


    def find_minimum(self):
        for q in range(self.QUANT):
            list = []
            for i in range(10):
                a = random.choice(self.NUMERALS)
                list.append(a)
            answer = min(list)
            equation = f"The minimum value of {list} is {answer}"
            self.QUESTIONS.append(equation)

    def find_maximum(self):
        for q in range(self.QUANT):
            list = []
            for i in range(10):
                a = random.choice(self.NUMERALS)
                list.append(a)
            answer = max(list)
            equation = f"The maximum value of {list} is {answer}"
            self.QUESTIONS.append(equation)

    def generate_division(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.DIV_OPERATORS)
            answer = a1 / a2
            equation = f"{a1} {operator} {a2} = {answer}"
            self.QUESTIONS.append(equation)

    def generate_multiplication(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.MUL_OPERATORS)
            answer = a1 * a2
            equation = f"{a1} {operator} {a2} = {answer}"
            self.QUESTIONS.append(equation)

    def generate_subtraction(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.SUB_OPERATORS)
            answer = a1 - a2
            equation = f"{a1} {operator} {a2} = {answer}"
            self.QUESTIONS.append(equation)

    def generate_addition(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.ADD_OPERATORS)
            answer = a1 + a2
            equation = f"{a1} {operator} {a2} = {answer}"
            self.QUESTIONS.append(equation)

    def __init__(self, quantity, number_range, output_loc):
        self.QUANT = quantity
        self.NUMERALS = [i for i in range(1, number_range)]
        self.QUESTIONS = []
        self.ADD_OPERATORS = ["+", "add", "increased by"]
        self.SUB_OPERATORS = ["-", "subtract", "minus", "reduced by"]
        self.DIV_OPERATORS = ["/", "divide by", "split into"]
        self.MUL_OPERATORS = ["*", "x", "multiplied by", "times by"]
        self.OUTPUT = output_loc
