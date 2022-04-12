import random


class questionGenerator:

    def __init__(self, quantity, number_range, output_loc):
        self.QUANT = quantity
        self.NUMERALS = [i for i in range(1, number_range)]
        self.QUESTIONS = []
        self.ADD_OPERATORS = ["+", "add", "increased by"]
        self.SUB_OPERATORS = ["-", "subtract", "minus", "reduced by"]
        self.DIV_OPERATORS = ["/", "divide by", "split into"]
        self.MUL_OPERATORS = ["*", "x", "multiplied by", "times by"]
        self.EQUALITY_OPERATORS = ["=", "equals", "becomes", "results in", "leads to"]
        self.OUTPUT = output_loc

    def save_questions(self):
        with open(self.OUTPUT, 'w') as output:
            for question in self.QUESTIONS:
                output.write(f"{question[0]} \n {question[1]} \n")


    def find_minimum(self):
        for q in range(self.QUANT):
            list = []
            for i in range(10):
                a = random.choice(self.NUMERALS)
                list.append(a)
            answer = min(list)
            equation = f"The minimum value of <e1>{list}</e1> is <e2>{answer}</e2>"
            relation = "Object-Attribute(e1, e2)"
            self.QUESTIONS.append((equation, relation))

    def find_maximum(self):
        for q in range(self.QUANT):
            list = []
            for i in range(10):
                a = random.choice(self.NUMERALS)
                list.append(a)
            answer = max(list)
            equation = f"The maximum value of <e1>{list}</e1> is <e2>{answer}</e2>"
            relation = "Object-Attribute(e1, e2)"
            self.QUESTIONS.append((equation, relation))

    def generate_division(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.DIV_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 / a2
            equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
            relation = "Function-Output(e1, e2)"
            self.QUESTIONS.append((equation, relation))

    def generate_multiplication(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.MUL_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 * a2
            equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
            relation = "Function-Output(e1, e2)"
            self.QUESTIONS.append((equation, relation))

    def generate_subtraction(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.SUB_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 - a2
            equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
            relation = "Function-Output(e1, e2)"
            self.QUESTIONS.append((equation, relation))

    def generate_addition(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.ADD_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 + a2
            equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
            relation = "Function-Output(e1, e2)"
            self.QUESTIONS.append((equation, relation))

    def generate(self):
        self.generate_addition()
        self.generate_subtraction()
        self.generate_multiplication()
        self.generate_division()
        self.find_maximum()
        self.find_minimum()
        self.save_questions()




generator = questionGenerator(10000, 100, "extractInfo/datasets/numeracy_annotated.txt")
generator.generate()
