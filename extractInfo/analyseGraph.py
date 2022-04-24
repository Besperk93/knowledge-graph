import os
import networkx as nx
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


class GraphAnalysis:

    def __init__(self, dir, output):
        self.DIR = dir
        self.OUTPUT = output
        self.DATE = datetime.now().strftime("%Y%m%d_%H%M")
        self.CUSTOM = ["Addend-Sum(e1, e2)", "Minuend-Difference(e1, e2)", "Multiplier-Product(e1, e2)", "Dividend-Quotient(e1, e2)", "Array-Maximum(e1, e2)", "Array-Minimum(e1, e2)", "Function-Output(e1, e2)"]

    def create_df(self):
        dfs = []
        with os.scandir("Vault/graph") as csvs:
            for csv in csvs:
                df = pd.read_csv(csv)
                dfs.append(df)
        print(f"Gathered {len(dfs)} files from graph directory")
        kb = pd.concat(dfs)
        self.describe_df(kb, "main")
        self.KB = kb
        kb.to_csv(f"{self.OUTPUT}main.csv", index=False)
        return

    def describe_df(self, df, title):
        results = df.describe()
        print(results)
        print(df.head())
        results.to_csv(f"{self.OUTPUT}{title}_results.csv")
        return


    def clean_kb(self):
        print("Removing Duplicates")
        old_len = len(self.KB)
        print(f"KB Length: {old_len}")
        self.KB = self.KB.drop_duplicates()
        print(f"Dropped {old_len - len(self.KB)} duplicate rows from KB")
        self.describe_df(self.KB, "clean")
        self.KB.to_csv(f"{self.OUTPUT}clean.csv", index=False)
        triples = self.KB[["Relation", "Entity1", "Entity2"]]
        triples.to_csv(f"{self.OUTPUT}triples.csv", index=False)
        return

    def create_graph(self, df, cols:tuple, title):
        graph = nx.from_pandas_edgelist(df, source=cols[0], target=cols[1], edge_attr=cols[2], create_using=nx.MultiGraph())
        plt.figure(figsize=(50,50))
        nx.draw(graph, node_size=20, with_labels=True, font_size=8)
        plt.savefig(f"{self.OUTPUT}{title}_{self.DATE}_graph.png")
        graph.clear()
        plt.close()
        del graph
        return

    def create_pie(self, df, column):
        title = column.lower()
        grouped = df.groupby(column)
        self.describe_df(grouped, column+'_pie')
        plot = grouped.size().plot.pie(y=column, figsize=(50, 50), fontsize=12)
        plt.savefig(f"{self.OUTPUT}{title}_{self.DATE}_pie.png")
        plt.close()
        return

    def analysis(self):
        total_length = len(self.KB)
        others = self.KB[self.KB["Relation"] == "Other"]
        print(f"{len(others)} of {total_length} relations are of type Other")
        print("############## Others: ##############")
        self.describe_df(others, "others")
        lst = self.CUSTOM
        custom = self.KB.query("Relation in @lst")
        print(f"{len(custom)} of {total_length} relations are of type Custom")
        print("############## Custom: ##############")
        self.describe_df(custom, "custom")
        print("############## Creating Graphs: ##############")
        main = self.KB[self.KB["Entity1"] != self.KB["Entity2"]]
        self.create_graph(main, ("Entity1", "Entity2", "Relation"), "main")
        print("############## Small: ##############")
        small = pd.concat([main.head(1000), main.tail(1000)])
        self.create_graph(small, ("Entity1", "Entity2", "Relation"), "small")
        print("############## Part of Speech: ##############")
        pos = self.KB[["Pos1", "Pos2", "Relation"]]
        pos = pos.drop_duplicates()
        pos = pos[pos["Pos1"] != pos["Pos2"]]
        self.describe_df(pos, "pos")
        self.create_graph(pos, ("Pos1", "Pos2", "Relation"), "pos")
        print("############## Labels: ##############")
        labels = self.KB[["Label1", "Label2", "Relation"]]
        labels = labels.drop_duplicates()
        labels = labels[labels["Label1"] != labels["Label2"]]
        self.describe_df(labels, "labels")
        self.create_graph(labels, ("Label1", "Label2", "Relation"), "labels")
        print("############## Creating Pie Charts ##############")
        self.create_pie(self.KB, "Relation")
        self.create_pie(self.KB, "Pos1")
        self.create_pie(self.KB, "Pos2")
        self.create_pie(self.KB, "Label1")
        self.create_pie(self.KB, "Label2")
        return

    def analyse(self):
        self.create_df()
        self.clean_kb()
        self.analysis()




# TEST
if __name__=="__main__":
    graph_info = GraphAnalysis("Vault/graph/", "./Vault/")
    graph_info.analyse()
