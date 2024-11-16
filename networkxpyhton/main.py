# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:15:54 2024

"""

# Load Pkgs
import pandas as pd
import networkx as nx

# 
#import warnings


import matplotlib.pyplot as plt

df = pd.read_csv("us_edgelist.csv")

df.head()

# Read Our Edgelist
us_graph = nx.from_pandas_edgelist(df,source="From",target="To")

print(type(us_graph))

# Info
print(us_graph)

# Check All Nodes/Object/People
print("nodes: ")
print(us_graph.nodes())

print("length nodes: ")
print(len(us_graph.nodes()))

print("length edges: ")
print(len(us_graph.edges()))

us_graph.add_edge("Franklin","Lincoln")

print(us_graph.nodes())



# nx.draw
# nx.draw_networkx


nx.draw(us_graph)


nx.draw(us_graph,with_labels=True)

nx.draw(us_graph,with_labels=True,node_color='g')

# method 2
nx.draw_networkx(us_graph)

plt.figure(figsize=(10,10))
nx.draw_networkx(us_graph)
plt.show()

plt.savefig("usa_pres.png")

# General Connection
print("degree matrix", nx.degree(us_graph))

# Degree of Connection 
print("obama degree:", nx.degree(us_graph,"Obama"))

print("Lincoln degree", nx.degree(us_graph,"Lincoln"))

print("centrality dictionary", nx.degree_centrality(us_graph))

print("centrality dictionary values sorted", sorted(nx.degree_centrality(us_graph).values()))

most_influential = nx.degree_centrality(us_graph)

for w in sorted(most_influential, key=most_influential.get, reverse=True):
    print(w, most_influential[w])

### Most Important Connection
#nx.eigenvector_centrality

print("eigen cents:", nx.eigenvector_centrality(us_graph))


most_important_link = nx.eigenvector_centrality(us_graph)

for w in sorted(most_important_link, key=most_important_link.get, reverse=True):
    print(w, most_important_link[w])


# What is the shortest connection between Obama and Bill Clinton
# Closeness central
#nx.shortest_path

print("shortest path Obama to Clinton")
print(nx.shortest_path(us_graph,"Obama","Clinton"))

print("shortest path Trump to Bush")
print(nx.shortest_path(us_graph,"Trump","Bush"))

print("shortest path Ivanka to Laura")
print(nx.shortest_path(us_graph,"Ivanka","Laura"))

### Betweener Centrality
# Bridge/Connect

plt.figure(figsize=(10,10))
nx.draw_networkx(us_graph)
plt.show()

print("betweenness cent", nx.betweenness_centrality(us_graph))

best_connector = nx.betweenness_centrality(us_graph)
for w in sorted(best_connector, key=best_connector.get, reverse=True):
    print(w, best_connector[w])
    
group1 = nx.bfs_tree(us_graph,"Obama")

group2 = nx.bfs_tree(us_graph,"Bush")

group3 = nx.bfs_tree(us_graph,"Hillary")

nx.draw_networkx(group1)
plt.show()
nx.draw_networkx(group2)
plt.show()
nx.draw_networkx(group3)
plt.show()

nx.node_connected_component(us_graph,"Trump")

### Association 
nx.clustering

print("clustering", nx.clustering(us_graph))



    



