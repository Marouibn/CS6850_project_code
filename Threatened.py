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
    ratio_cc = len(nx.node_connected_component(G,s))/len(G.nodes)
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
    return len([i for i in G.nodes if state[i]<=1]), ratio_cc



def run_simulation(p_f,B_f, min_n = 5, max_n=1000, n_iterations=20):
    
    Y = []
    total_cc = 0
    for n in tqdm(range(min_n, max_n,5)):
        s=0
        
        p = p_f(n)
        B = B_f(n)
        for i in range(n_iterations):
            saved, cc = greedy_iteration(n, p, B)
            s += saved
            total_cc += cc

        Y.append(s/n_iterations)
    
    total_cc /= len(range(min_n, max_n,5)) * n_iterations

    plt.style.use("ggplot")
    plt.xlabel("Graph size")
    plt.ylabel("Number of saved nodes")
    plt.plot(range(min_n, max_n,5), Y, '.',label = "Number of nodes saved")
    plt.plot(range(min_n, max_n), range(min_n, max_n), label="Total number of nodes")
    a,b = np.polyfit(range(min_n,max_n,5), Y,1)
    plt.plot(range(min_n, max_n,5), a*np.array(range(min_n, max_n,5))+b)
    plt.legend()
    


    print("The slope is {}".format(a))
    print("The average component size {}%".format(total_cc))
    return Y



# p  functions
def p_lin(x):
    def g(n):
        return x/n
    return g
    
def p_log(x):
    def g(n):
        return x/np.log(n)
    return g


def p_log_log(x):
    def g(n):
        return x/np.log(np.log(n))
    return g



# B Functions
def B_lin(x):
    def g(n):
        return n/x
    return g

def B_lin_log(x):
    def g(n):
        return max(1, int(x * n / (np.log(n))))
    return g

def B_cte(x):
    def g(n):
        return x
    return g

def B_log(x):
    def g(n):
        return int(x * np.log(n))
    return g

def B_log_log(x):
    def g(n):
        return 5+int(x * np.log(np.log(n)))
    return g


def B_lin_log_log(x):
    def g(n):
        return int(x * n / np.log(np.log(n)))
    return g



f_p = p_lin(6)
f_B = B_log(4)
n_max = 200
n_iterations = 500


Y = run_simulation(f_p,f_B,5,n_max,n_iterations)
plt.show()