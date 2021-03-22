# A Python program to print topological sorting of a graph 
# using indegrees 
import os
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict 



# Class to represent a graph 
class Graph: 
    def __init__(self, vertices): 
#         self.graph = defaultdict(list) # dictionary containing adjacency List 
#         self.V = vertices # No. of vertices 
        self.graph = np.zeros((6, 6))
    
    # function to add an edge to graph 
    def add_edge(self, u, v, w): 
        self.graph[u][v] = w 


    def get_node_in_degree(self, u):
        return  np.count_nonzero(self.graph[:,u])

        
    # The function to do Topological Sort. 
    def topologicalSort(self, start_node): 
        
        # Create a vector to store indegrees of all 
        # vertices. Initialize all indegrees as 0. 
        path = [start_node]
        in_degree = [0]*(6) 
        in_degree[start_node] = 1
        
        # Traverse adjacency lists to fill indegrees of 
        # vertices. This step takes O(V + E) time 
        current = start_node
        while not all(in_degree):
            # Initilaize minimum distance for next node
            maximum = -10000

            # Search not nearest vertex not in the
            # shortest path tree
            for v in range(6):
                if self.graph[current][v] > maximum and in_degree[v] == False:
                    maximum = self.graph[current][v]
                    max_index = v

            in_degree[max_index] = 1
            path.append(max_index)
            current = max_index

        return path
            
        
if __name__=='__main__':

    PATH = './data/adjacency_matrix/'
    final_result = []
    for test in os.listdir(PATH):
        with open(os.path.join(PATH, test), 'rb') as f:
            edge_list = pickle.load(f) 
            g = Graph(6) 
            
            for i, j, logits, s1, s2 in edge_list:
                if np.argmax(logits, axis=1).flatten():
                    
                    g.add_edge(i, j, logits[0][1])
#                     print(s1, s2)
#                     print(i, j)
#                     print(logits)
#                     print("=======================================")
             
            
            start_node = 0
            for i in range(6):
                if g.get_node_in_degree(i) == 0:
                    start_node = i
                    
            temp = [test.replace(".pkl", "")]
            temp.extend(g.topologicalSort(start_node)) 
        final_result.append(temp)
    
    pd.DataFrame(final_result, columns= ["ID", "index1", "index2", "index3", "index4", "index5", "index6"]).to_csv("results_v1.csv", index=False)
    
    # This code is contributed by Neelam Yadav 
