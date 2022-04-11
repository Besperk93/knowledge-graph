import re
import os
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from inference import InferencePipeline
from tqdm import tqdm
from datetime import datetime

class TranscriptProcessor:

    def __init__(self):
        self.OUTPUT = "Vault/graph/"
        self.COLUMNS = ["Video", "Relation", "Entity1", "Entity2"]
        self.ROWS = []
        self.EXTRACTOR = InferencePipeline()


    def batch_process(self, loc, batch_size):
        total_scripts = len([f for f in os.listdir(loc)])
        self.BATCH = []
        self.BATCH_NO = 0
        with os.scandir(loc) as scripts:
            for script in tqdm(scripts, total=total_scripts):
                self.BATCH.append(script)
                if len(self.BATCH) == batch_size:
                    self.BATCH_NO += 1
                    self.process(self.BATCH)
                    self.store_csv()
                    self.BATCH[:] = []
                    self.ROWS[:] = []
                else:
                    continue


    def process(self, loc):
        if isinstance(loc, list):
            time = datetime.now().strftime("%y%m%d_%H-%M")
            self.NAME = f"{time}_batch_{self.BATCH_NO}"
            for script in tqdm(loc, total=len(loc)):
                self.TITLE = script.name
                sentences = self.clean_script(script)
                self.extract_relations(sentences)
            self.store_csv()
        elif (isinstance(loc, str)) and (os.path.isfile(loc)):
                self.NAME = re.search(r"^.*\/(.*)\.txt$", loc).group(1)
                self.TITLE = re.search(r"^.*\/(.*)\.txt$", loc).group(1)
                sentences = self.clean_script(loc)
                self.extract_relations(sentences)
                self.store_csv()
        elif (isinstance(loc, str)) and (os.path.isdir(loc)):
            try:
                self.NAME = re.search(r".*\/([A-z0-9]*)$", loc).group(1)
            except:
                time = datetime.now().strftime("%y%m%d_%H-%M")
                self.NAME = f"{time}_batch"
            total_scripts = len([f for f in os.listdir(loc)])
            with os.scandir(loc) as scripts:
                for script in tqdm(scripts, total=total_scripts):
                    try:
                        self.TITLE = script.name
                        sentences = self.clean_script(script)
                        self.extract_relations(sentences)
                    except Exception as e:
                        pass
                        # print(f"Error processing transcript directory: {repr(e)}")
        else:
            print(f"Invalid input for process")
            return


    def create_graph(self, df):
        graph = nx.from_pandas_edgelist(df, source="Entity1", target="Entity2", edge_attr="Relation")
        nx.draw(graph)
        plt.savefig(f"{self.OUTPUT}{self.NAME}_graph.png")
        graph.clear()
        return

    def store_csv(self):
        if len(self.ROWS) == 0:
            print(f"Batch Failed: {self.BATCH_NO}")
            with open(f"Vault/fails/{self.NAME}.txt", 'w') as fail:
                for script in self.BATCH:
                    fail.write(script.name + '\n')
        df = pd.DataFrame(self.ROWS, columns=self.COLUMNS)
        df.to_csv(f"{self.OUTPUT}{self.NAME}_data.csv", index=False)
        return self.create_graph(df)

    def extract_relations(self, sentences):
        if sentences is not None:
            for sentence in sentences:
                #NOTE: probably worth adding (if len(sentence.split()) > 500: continue)
                if len(sentence.split()) > 500:
                    continue
                try:
                    predictions = self.EXTRACTOR.extract_relations(sentence)
                    e1_pattern = re.compile("\[E1\](.*?)\[\/E1\]")
                    e2_pattern = re.compile("\[E2\](.*?)\[\/E2\]")
                    for result in predictions:
                        e1 = re.search(e1_pattern, result[0]).group(1)
                        e2 = re.search(e2_pattern, result[0]).group(1)
                        rel = result[-1]
                        self.ROWS.append([self.TITLE, rel, e1, e2])
                except Exception as e:
                    continue
        return


    def clean_script(self, script):
        # TODO: a few transcripts are being passes to the model as long sentences.
        transcript = ""
        # Strip [voiceover] flag from the start of each script
        pattern = re.compile("^(.*\[.*\]\s)(.*)")
        with open(script, "r") as input:
            for count, line in enumerate(input, start=1):
                if count % 2 == 0:
                    try:
                        line = re.sub(pattern, r'\2', line)
                    except:
                        pass
                    line = line.replace('\n', ' ')
                    transcript += line
        sentences = transcript.split(". ")
        return sentences
