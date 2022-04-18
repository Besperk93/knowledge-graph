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
        self.OUTPUT = "./Vault/graph/"
        self.COLUMNS = ["Video", "Relation", "Entity1", "Entity2", "Pos1", "Pos2", "Label1", "Label2"]
        self.ROWS = []
        self.EXTRACTOR = InferencePipeline()
        self.TIME = datetime.now().strftime("%y%m%d_%H-%M")

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
        print("Processing Complete")
        return


    def process(self, loc):
        if isinstance(loc, list):
            self.NAME = f"{self.TIME}_batch_{self.BATCH_NO}"
            for script in tqdm(loc, total=len(loc)):
                self.TITLE = script.name
                sentences = self.clean_script(script)
                self.extract_relations(sentences)
        elif (isinstance(loc, str)) and (os.path.isfile(loc)):
                self.NAME = re.search(r"^.*\/(.*)\.txt$", loc).group(1)
                self.TITLE = re.search(r"^.*\/(.*)\.txt$", loc).group(1)
                sentences = self.clean_script(loc)
                self.extract_relations(sentences)
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

    def clean_script(self, script):
        # TODO: a few transcripts are being passes to the model as long sentences.
        transcript = ""
        # Strip [voiceover] flag from the start of each script
        pattern = re.compile("^(.*\[.*\]\s)(.*)")
        with open(script, "r") as input:
            for count, line in enumerate(input, start=1):
                # ignore timestamps
                if count % 2 == 0:
                    try:
                        line = re.sub(pattern, r'\2', line)
                    except:
                        pass
                    line = line.replace('\n', ' ')
                    transcript += line
        sentences = transcript.split(". ")
        return sentences

    def extract_relations(self, sentences):
        if sentences is not None:
            for sentence in sentences:
                #NOTE: probably worth adding (if len(sentence.split()) > 500: continue)
                if len(sentence.split(' ')) > 500:
                    continue
                try:
                    # predictions refers to a (mention, relation) tuple
                    objects = self.EXTRACTOR.extract_relations(sentence)
                    # e1_pattern = re.compile("\[E1\](.*?)\[\/E1\]")
                    # e2_pattern = re.compile("\[E2\](.*?)\[\/E2\]")
                    for obj in objects:
                        row = obj.as_row()
                        row.insert(0, self.TITLE)
                        self.ROWS.append(row)
                except Exception as e:
                    continue

    def store_csv(self):
        if len(self.ROWS) == 0:
            print(f"Batch Failed: {self.BATCH_NO}")
            with open(f"Vault/fails/{self.NAME}.txt", 'w') as fail:
                for script in self.BATCH:
                    fail.write(script.name + '\n')
        df = pd.DataFrame(self.ROWS, columns=self.COLUMNS)
        df.to_csv(f"{self.OUTPUT}{self.NAME}_data.csv", index=False)
        return self.create_graph(df)

    def create_graph(self, df):
        graph = nx.from_pandas_edgelist(df, source="Entity1", target="Entity2", edge_attr="Relation")
        plt.figure(figsize=(50,50))
        nx.draw(graph, node_size=20, with_labels=True, font_size=8)
        plt.savefig(f"{self.OUTPUT}{self.NAME}_graph.png")
        graph.clear()
        plt.clf()
        del graph
        return
