import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


class Grapher:

    def graph(self):
        self.load_df(self.INPUT)

    def visualise(self, g):
        try:
            nx.draw(g, with_labels=True)
            plt.savefig(self.OUTPUT)
            plt.show()
        except Exception as e:
            print(f"Error drawing graph: {repr(e)}")

    def create_graph(self, df):
        try:
            graph = nx.Graph()
            graph = nx.from_pandas_edgelist(df, source="Entity_1", target="Entity_2", edge_attr="Relation")
            return self.visualise(graph)
        except Exception as e:
            print(f"Error creating graph object: {repr(e)}")

    def load_df(self, csv_loc):
        try:
            df = pd.read_csv(csv_loc)
            return self.create_graph(df)
        except Exception as e:
            print(f"Error reading csv: {repr(e)}")

    def __init__(self, input_loc, output_loc):
        self.OUTPUT = output_loc
        self.INPUT = input_loc
