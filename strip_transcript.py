import re
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")
transcript = "2 Solving for mk - YouTube.txt"

def clean_script(script):
    transcript = ""
    pattern = re.compile("^(.*\[.*\]\s)(.*)")
    with open(script, "r") as input:
        for count, line in enumerate(input, start=1):
            if count % 2 == 0:
                line = re.sub(pattern, r'\2', line)
                line = line.replace('\n', ' ')
                transcript += line
    sentences = transcript.split(". ")
    string = " ".join(sentences)
    return string


def extract_entities(string):
    doc = nlp(string)
    print(f'{len(doc.ents)} entities were found in this document')
    return [(token.text, token.label_) for token in doc.ents]


print(extract_entities(clean_script(transcript)))
