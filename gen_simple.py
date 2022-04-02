from question_generator import questionGenerator

simple_questions = questionGenerator(10000, 1000, "test_questions.txt")
simple_questions.generate()
