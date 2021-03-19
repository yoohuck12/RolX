import math
import random
import scipy.stats
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from parser import parameter_parser

def dataset_reader(path):
    edges = pd.read_csv(path).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph

def inducer(graph, node):
    nebs = nx.neighbors(graph, node)
    sub_nodes = nebs + [node]
    sub_g = nx.subgraph(graph, sub_nodes)
    out_counts = np.sum(map(lambda x: len(nx.neighbors(graph,x)), sub_nodes))
    return sub_g, out_counts, nebs

def complex_aggregator(x):
    return [np.min(x),np.std(x),np.var(x),np.mean(x),np.percentile(x,25),np.percentile(x,50),np.percentile(x,100),scipy.stats.skew(x),scipy.stats.kurtosis(x)]

def aggregator(x):
    return [np.sum(x),np.mean(x)]

def state_printer(x):
    print("-"*80)
    print(x)
    print("")

def sub_selector(old_features, new_features, pruning_threshold):
    """
    뭐하는건지 잘 모르겠음.
    """

    print("Cross-temporal feature pruning started.")
    indices = set()
    for i in tqdm(range(0,old_features.shape[1])):
        for j in range(0, new_features.shape[1]):
            c = np.corrcoef(old_features[:,i], new_features[:,j])
            if abs(c[0,1]) > pruning_threshold:
                indices = indices.union(set([j]))
        keep = list(set(range(0,new_features.shape[1])).difference(indices))
        new_features = new_features[:,keep]
        indices = set()
    return new_features


class RecursiveExtractor:

    def __init__(self, args):
        """그래프를 로드하고, feature 를 생성한다."""
        self.args = args
        if self.args.aggregator == "complex":
            # feature 를 뽑아낼 때 사용할 aggregator 에 대한 명세
            self.aggregator = complex_aggregator
        else:
            self.aggregator = aggregator
        # 사용할 aggregator 의 개수
        self.multiplier = len(self.aggregator(0))
        self.graph = dataset_reader(self.args.input)
        self.nodes = nx.nodes(self.graph)
        self.create_features()

    def basic_stat_extractor(self):
        """
        노드 별로 순회하면서 노드의 이웃 정보를 알아내고, 노드가 포함된 subgraph
        에서 노드의 degree, cluster coefficient 등을 알아내고, 이를 이용해서
        basic_features 를 만들어낸다.
        추후 features 구조체에 features[recursion][nodeNum][features] 로 저장
        되어 recursion 의 기본값으로 사용된다.
        """
        self.base_features = []
        self.sub_graph_container = {}
        for node in tqdm(range(0,len(self.nodes))):
            sub_g, overall_counts, nebs = inducer(self.graph, node)
            in_counts = len(nx.edges(sub_g))
            self.sub_graph_container[node] = nebs
            deg = nx.degree(sub_g, node)
            trans = nx.clustering(sub_g, node)
            self.base_features.append([in_counts, overall_counts, float(in_counts)/float(overall_counts), float(overall_counts - in_counts)/float(overall_counts),deg, trans])
        self.features = {}
        self.features[0] = np.array(self.base_features)
        print("")
        del self.base_features

    def single_recursion(self, i):
        """
        이전 라운드에서 사용된 feature 들을 이용해서 새로운 feature 를 만든다.
        노드를 선택하고 해당 노드의 feature 를 가져와서 aggregator 를 이용해
        새로운 feature 를 만든다.
        new_features = (노드 개수, 이전feature개수*multiplier)
        """
        # 이전 라운드에서의 Feature 의 개수
        features_from_previous_round = self.features[i].shape[1]
        new_features = np.zeros((len(self.nodes), features_from_previous_round*self.multiplier))
        for k in tqdm(range(0,len(self.nodes))):
            selected_nodes = self.sub_graph_container[k]
            # i 번째 recursion feature 를 가지고 오고, 타겟 node 의 feature 를
            # main_features 로 가져온 뒤,
            main_features = self.features[i][selected_nodes,:]
            # j 번째 feature 에 대해서 aggregator 들을 이용해서
            # features_from_previous_round 개수만큼의 새로운 array 를 생성하고
            # 해당 array 를 reduce 함수를 통해서 새로운 feature (기존 개수 * multiplier)를 생성한다.
            new_features[k,:]= reduce(lambda x,y: x+y,[self.aggregator(main_features[:,j]) for j in range(0,features_from_previous_round)])
        return new_features

    def do_recursions(self):
        for recursion in range(0,self.args.recursive_iterations):
            state_printer("Recursion round: " + str(recursion+1) + ".")
            new_features = self.single_recursion(recursion)
            new_features = sub_selector(self.features[recursion], new_features, self.args.pruning_cutoff)
            self.features[recursion+1] = new_features

        self.features = np.concatenate(self.features.values(), axis = 1)
        # normalize 해서 0~1 사이로 만들어준다.
        self.features = self.features / (np.max(self.features)-np.min(self.features))

    def binarize(self):
        """
        feature 들에 대해서quantization 을 한다고 하는데 뭔지 잘 모르겠다.
        """
        self.new_features = []
        for x in tqdm(range(0,self.features.shape[1])):
            try:
                # qcut(quantization cut): 동일 개수의 데이터들로 나눈다
                # get_dummies: 가변수화 한다 (one-hot vector 로 생성)
                # feature 들을 동일 개수의 데이터로 나누고, one-hot vector 로
                # 만든뒤, 기존 feature 에 덧붙여주는 작업을 한다.
                self.new_features = self.new_features + [pd.get_dummies(pd.qcut(self.features[:,x],self.args.bins, labels = range(0,self.args.bins), duplicates = "drop"))]
            except:
                pass
        self.new_features = pd.concat(self.new_features, axis = 1)

    def dump_to_disk(self):
        self.new_features.columns = map(lambda x: "x_" + str(x), range(0,self.new_features.shape[1]))
        self.new_features.to_csv(self.args.recursive_features_output, index = None)

    def create_features(self):
        state_printer("노드의 feature 를 뽑고, subgraph 를 만드는 과정을 수행한다.")
        self.basic_stat_extractor()
        state_printer("재귀적으로 노드의 feature 를 정제한다.")
        self.do_recursions()
        state_printer("이진 특성 quantization 을 한다.")
        self.binarize()
        state_printer("Saving the raw features.")
        self.dump_to_disk()
        state_printer("The number of extracted features is: " + str(self.new_features.shape[1]) + ".")
