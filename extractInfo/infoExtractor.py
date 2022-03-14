import os
import re

class infoExtractor:

    def extract_entities(self, sentences):
        pass

    def clean_script(self, script):
        transcript = ""
        pattern = re.compile("^(.*\[.*\]\s)(.*)")
        with open(script, "r") as file:
            for count, line in enumerate(file, start=1):
                if count % 2 == 0:
                    line = re.sub(pattern, r'\2', line)
                    line = line.strip('\n')
                    transcript += line
        sentences = transcript.split(". ")
        print(sentences)
        return self.extract_entities(sentences)


    def open_transcript(self):
        with os.scandir(self.LOCATION) as scripts:
            for script in scripts:
                self.clean_script(script)

    def __init__(self, location):
        self.LOCATION = location
