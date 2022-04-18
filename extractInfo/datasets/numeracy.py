import numpy as np
import random
import os
import re



def combine_raw(loc1, loc2):
    """Combine the raw cnn pretraining data with the generated questions"""
    base_path = "./Vault/mtb/output/"
    with open(os.path.join(base_path, loc1), 'r') as loc1:
        data1 = loc1.read()

    with open(os.path.join(base_path, loc2), 'r') as loc2:
        data2 = loc2.read()

    combined = data1 + data2

    with open(os.path.join(base_path, "combined_train.txt"), 'w') as out_file:
        out_file.write(combined)



def combine_annotated(loc1, loc2):
    """combined the annoated questions with the semeval data"""
    base_path = "./Vault/mtb/eval/"
    with open(os.path.join(base_path, loc1), 'r') as loc1:
        data1 = loc1.read()

    with open(os.path.join(base_path, loc2), 'r') as loc2:
        data2 = loc2.read()

    pattern = re.compile(r"^([0-9]+).*")
    lines = data1.split("\n\n") + data2.split("\n\n")
    len1 = len(lines)
    random.shuffle(lines)
    # print(lines[:3])
    re_indexed = [re.sub(r"^[0-9]+", str(i + 1), line) for i, line in enumerate(lines)]
    # print(re_indexed[:3])
    print(f"First length: {len1}\nSecond length:{len(re_indexed)}")

    with open(os.path.join(base_path, "combined_eval.txt"), 'w') as out_file:
        for item in re_indexed:
            out_file.write(item + "\n\n")


def split_annotated(alpha):
    """Split the combined annotated dataset into train/test"""
    # Alpha of 0.8 would result in an 80/20 train/test split
    assert isinstance(alpha, float)
    assert alpha < 1.0

    base_path = "./Vault/mtb/eval"
    with open(os.path.join(base_path, "combined_eval.txt"), 'r') as combined:
        full = combined.read()

    rows = full.split("\n\n")
    print(len(rows))
    # Check for empty rows
    rows = [row for row in rows if row != ""]
    re_indexed = [re.sub(r"^[0-9]+", str(i + 1), row) for i, row in enumerate(rows)]
    print(len(re_indexed))
    # Should already be shuffled but shuffle again in case txt has been altered
    # random.shuffle(rows)
    split = int(alpha * len(re_indexed))
    train = [row for row in re_indexed[:split]]
    test = [row for row in re_indexed[split:]]

    assert len(test) + len(train) == len(rows)
    print(f"Test: {len(test)}, Train: {len(train)}, Total: {len(rows)}")
    with open(os.path.join(base_path, "combined_train.txt"), 'w') as out:
        for item in train:
            out.write(item + "\n\n")
    with open(os.path.join(base_path, "combined_test.txt"), 'w') as out:
        for item in test:
            out.write(item + "\n\n")


class questionGenerator:

    def __init__(self, quantity, number_range, output_loc, annotated=False):
        self.QUANT = quantity
        self.NUMERALS = [i for i in range(1, number_range)]
        self.QUESTIONS = []
        self.ADD_OPERATORS = ["+", "add", "increased by"]
        self.SUB_OPERATORS = ["-", "subtract", "minus", "reduced by"]
        self.DIV_OPERATORS = ["/", "divide by", "split into"]
        self.MUL_OPERATORS = ["*", "x", "multiplied by", "times by"]
        self.EQUALITY_OPERATORS = ["=", "equals", "becomes", "results in", "leads to", "produces"]
        self.ANNOTATED = annotated
        if annotated:
            self.OUTPUT = os.path.join(output_loc, "eval/numeracy_test_annotated.txt")
        else:
            self.OUTPUT = os.path.join(output_loc, "output/numeracy_pretrain.txt")

    def save_questions(self):
        with open(self.OUTPUT, 'w') as output:
            index = 0
            for question in self.QUESTIONS:
                index += 1
                if self.ANNOTATED:
                    output.write(f"{index} \"{question[0]}\"\n{question[1]}\nComment: \n\n")
                else:
                    output.write(question)


    def find_minimum(self):
        for q in range(self.QUANT):
            list = []
            for i in range(10):
                a = random.choice(self.NUMERALS)
                list.append(a)
            answer = min(list)
            if self.ANNOTATED:
                p = np.random.uniform()
                if p > 0.5:
                    equation = f"The minimum value of <e1>{list}</e1> is <e2>{answer}</e2>"
                    relation = "Array-Minimum(e1, e2)"
                else:
                    equation = f"<e2>{answer}</e2> is the minimum value of <e1>{list}</e1>"
                    relation = "Array-Minimum(e1, e2)"
                self.QUESTIONS.append((equation, relation))
            else:
                equation = f"The minimum value of {list} is {answer}. "
                self.QUESTIONS.append(equation)


    def find_maximum(self):
        for q in range(self.QUANT):
            list = []
            for i in range(10):
                a = random.choice(self.NUMERALS)
                list.append(a)
            answer = max(list)
            if self.ANNOTATED:
                p = np.random.uniform()
                if p > 0.5:
                    equation = f"The maximum value of <e1>{list}</e1> is <e2>{answer}</e2>"
                    relation = "Array-Maximum(e1, e2)"
                else:
                    equation = f"<e2>{answer}</e2> is the maxmimum value of <e1>{list}</e1>"
                    relation = "Array-Maximum(e1, e2)"
                self.QUESTIONS.append((equation, relation))
            else:
                equation = f"The maximum value of {list} is {answer}. "
                self.QUESTIONS.append(equation)



    def generate_division(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.DIV_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 / a2
            if self.ANNOTATED:
                p = np.random.uniform()
                if p > 0.5:
                    equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
                    relation = "Function-Output(e1, e2)"
                else:
                    equation = f"<e1>{a1}</e1> {operator} {a2} {equality} <e2>{answer}</e2>"
                    relation = "Dividend-Quotient(e1, e2)"
                self.QUESTIONS.append((equation, relation))
            else:
                equation = f"{a1} {operator} {a2} {equality} {answer}. "
                self.QUESTIONS.append(equation)


    def generate_multiplication(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.MUL_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 * a2
            if self.ANNOTATED:
                p = np.random.uniform()
                if p > 0.5:
                    equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
                    relation = "Function-Output(e1, e2)"
                else:
                    equation = f"<e1>{a1}</e1> {operator} {a2} {equality} <e2>{answer}</e2>"
                    relation = "Multiplier-Product(e1, e2)"
                self.QUESTIONS.append((equation, relation))
            else:
                equation = f"{a1} {operator} {a2} {equality} {answer}. "
                self.QUESTIONS.append(equation)


    def generate_subtraction(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.SUB_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 - a2
            if self.ANNOTATED:
                p = np.random.uniform()
                if p > 0.5:
                    equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
                    relation = "Function-Output(e1, e2)"
                else:
                    equation = f"<e1>{a1}</e1> {operator} {a2} {equality} <e2>{answer}</e2>"
                    relation = "Minuend-Difference(e1, e2)"
                self.QUESTIONS.append((equation, relation))
            else:
                equation = f"{a1} {operator} {a2} {equality} {answer}. "
                self.QUESTIONS.append(equation)


    def generate_addition(self):
        for q in range(self.QUANT):
            a1 = random.choice(self.NUMERALS)
            a2 = random.choice(self.NUMERALS)
            operator = random.choice(self.ADD_OPERATORS)
            equality = random.choice(self.EQUALITY_OPERATORS)
            answer = a1 + a2
            if self.ANNOTATED:
                p = np.random.uniform()
                if p > 0.5:
                    equation = f"<e1>{a1} {operator} {a2}</e1> {equality} <e2>{answer}</e2>"
                    relation = "Function-Output(e1, e2)"
                else:
                    equation = f"<e1>{a1}</e1> {operator} {a2} {equality} <e2>{answer}</e2>"
                    relation = "Addend-Sum(e1, e2)"
                self.QUESTIONS.append((equation, relation))
            else:
                equation = f"{a1} {operator} {a2} {equality} {answer}. "
                self.QUESTIONS.append(equation)


    def generate(self):
        self.generate_addition()
        self.generate_subtraction()
        self.generate_multiplication()
        self.generate_division()
        self.find_maximum()
        self.find_minimum()
        self.save_questions()




generator = questionGenerator(2000, 100, "./Vault/mtb/", True)
generator.generate()
combine_annotated("semeval2010_task8/TRAIN_FILE.TXT", "semeval2010_task8/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT")
combine_annotated("numeracy_test_annotated.txt", "combined_eval.txt")
split_annotated(0.8)
