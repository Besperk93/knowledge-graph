import re


def format_paper(loc):

    with open(loc, 'r') as input:
        file = input.read()


    start_pattern = re.compile(".*Answer all questions in the spaces provided\.(.*)")
    questions = re.search(start_pattern, file).group(1)
    print(questions)



# Test

format_paper("./extractExams/output/paper1.txt")
