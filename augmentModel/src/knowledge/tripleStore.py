import pandas as pd
import networkx
import os
import re

class TripleStore:

    def __init__(self, csv_dir):
        self.LOC = csv_dir
        self.COLUMNS = ["Entity1", "Relation", "Entity2"]
        self.DF = None
        self.ROWS = []
        self.load_csvs()

    def load_csvs(self):
        with os.scandir(self.LOC) as dir:
            for csv in dir:
                temp = pd.read_csv(self.LOC + csv.name)
                e1 = temp["Entity1"].tolist()
                e2 = temp["Entity2"].tolist()
                rel = temp["Relation"].tolist()
                self.ROWS.extend(list(zip(e1, rel, e2)))
        self.DF = pd.DataFrame(self.ROWS, columns=self.COLUMNS)
        old_len = len(self.DF.index)
        self.DF = self.DF.drop_duplicates()
        print(f"Dropped {old_len - len(self.DF.index)} duplicate rows from store")
        print(f"New Store contains {len(self.DF.index)} relation triples")


    def create_graph(self, df):
        base_path = "./Vault/graph/"
        graph = nx.from_pandas_edgelist(df, source="Entity1", target="Entity2", edge_attr="Relation")
        plt.figure(figsize=(50,50))
        nx.draw(graph, node_size=20, with_labels=True, font_size=8)
        plt.savefig(os.path.join(base_path, f"graph.png"))
        graph.clear()
        plt.close()
        del graph
        return


    def check_entity(self, e1:str):
        # NOTE: Commonly occuring starting sylables may return incorrect entities. I could just use a tokenizer.
        pattern = re.compile(f"^{e1}.*")
        entities = self.DF[self.DF["Entity1"].str.contains(pattern, regex=True)]
        if len(entities.index) > 0:
            return e1
        else:
            return None

    def get_entity_candidates(self, e1:str, range):
        entities = self.DF[self.DF["Entity1"] == e1]
        # TODO: some sort of sort
        entities = entities.head(range)
        if len(entities.index) > 0:
            return entities
        else:
            return None
