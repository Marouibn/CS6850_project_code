import networkx as nx
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def draw_Gnp(G, state):
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes   
    nx.draw_networkx_nodes(G, pos, nodelist=[i for i in G.nodes if state[i]==0], node_color="tab:grey", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=[i for i in G.nodes if state[i]==1], node_color="tab:blue", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=[i for i in G.nodes if state[i]>=2], node_color="tab:red", **options)
    nx.draw_networkx_edges(G, pos)
    plt.show()


def greedy_iteration(n, p, B):
    
    G = nx.gnp_random_graph(n,p)


    # G = nx.Graph()
    # G.add_nodes_from(range(11))
    # G.add_edges_from([(1, 0), (0, 2), (1,3),(1,4),(2,5),(2,6),(3,7),(4,8),(5,9), (6,10)])
    # print([ i for i in G.edges()])


    # List to keep track of the state of a node
    # 0 = 
    # 1 = protected
    # 2 = burning
    # 3 = burned (i.e. already checked its neighbours in a previous step)
    state = [0] * len(G.nodes)

    # A boolean variable that indicates when we have to stop
    b=0
    

    # Fire break out here
    s = np.random.choice(G.nodes)
    #print(G.degree(s))
    #print(len(nx.node_connected_component(G,s))/len(G.nodes))
    state[s] = 3
    #draw_Gnp(G,state)


    # Let us find the neighbours of burned nodes
    nei = set()
    for m in G.neighbors(s):
        nei.add(m)
    #print(nei)

    if B >= len(nei):
        P = nei
    else:
        P = rd.sample(nei, B)
    
    for m in nei:
        if m in P:
            state[m] = 1
        else:
            state[m] = 2
            b = 1 # If one new node burns then we continue
    #draw_Gnp(G,state)

    while b:
        
        b=0
        nei.clear()
        for node in G.nodes:
            if state[node] == 2:
                state[node] = 3
                for m in G.neighbors(node):
                    if state[m] == 0:
                        nei.add(m)
        
        if len(nei) <= B:
            P = nei
        else:
            P = rd.sample(nei, B)

        for m in nei:
            if m in P:
                state[m] = 1
            else:
                state[m] = 2
                b=1
    
        #draw_Gnp(G,state)
    return len([i for i in G.nodes if state[i]<=1])



def run_simulation(f, min_n = 5, max_n=1000, n_iterations=20):
    
    Y = []
    for n in tqdm(range(min_n, max_n,5)):
        s=0
        p = f(n)
        for i in range(n_iterations):
            s += greedy_iteration(n,p,4)
        Y.append(s/n_iterations)
    
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of saved nodes")
    plt.plot(range(min_n, max_n,5), Y, '.',label = "Number of nodes saved")
    plt.plot(range(min_n, max_n), range(min_n, max_n), label="Total number of nodes")
    plt.legend()
    plt.show()
    
    return Y

def f(x):
    return 6/x

Y = run_simulation(f,5,400,500)