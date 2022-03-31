import os
import re
import spacy_recogniser
import pandas as pd
import json

class infoExtractor:

    def display_entity_data(self):
        print(self.ENTITY_DATA)
        with open(self.OUTPUT, "w") as output:
            json.dump(self.ENTITY_DATA, output)


    def extract_entities(self, sentences, title):
        extractor = spacyRecogniser.entityRecogniser(sentences)
        entities = extractor.run_ner()
        self.ENTITY_DATA.append({title: entities})

    def clean_script(self, script):
        transcript = ""
        pattern = re.compile("^(.*\[.*\]\s)(.*)")
        with open(script, "r") as input:
            for count, line in enumerate(input, start=1):
                if count % 2 == 0:
                    line = re.sub(pattern, r'\2', line)
                    line = line.replace('\n', ' ')
                    transcript += line
        sentences = transcript.split(". ")
        return self.extract_entities(sentences, script.name)


    def open_transcript(self):
        with os.scandir(self.LOCATION) as scripts:
            for script in scripts:
                self.clean_script(script)

    def __init__(self, script_location, output_location):
        self.LOCATION = script_location
        self.ENTITY_DATA = []
        self.OUTPUT = output_location
