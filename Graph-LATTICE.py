#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import random
import networkx as nx 
import axelrod as axl
from axelrod.graph import Graph
from collections import namedtuple
import csv
import pandas as pd
import numpy as np
import tqdm


# In[3]:


strategies = [s() for s in axl.strategies]
axl.seed(0)  


# In[4]:


#Defining players
strategies = [axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp()]

players = random.sample(strategies, 50)
plays = players      
         


# In[5]:


G = nx.Graph()
G.add_nodes_from(range(1,9))
G.add_edges_from([(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,24),(24,25),
         (25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,36),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42),(42,43),(43,44),(44,45),(45,46),(46,47),(47,48),(48,49),
                  (0,25),(1,26),(2,27),(3,28),(4,29),(5,30),(6,31),(7,32),(8,33),(9,34),(10,35),(11,36),(12,37),(13,38),(14,39),(15,40),(16,41),(17,42),(18,43),(19,44),(20,45),(21,46),(22,47),(23,48),(24,49)])
nx.draw(G)
plt.savefig("Graph.png", format="PNG")


# In[6]:


H=nx.grid_2d_graph(25,2)
pos = dict( (n, n) for n in H.nodes() )
labels = dict( ((i, j), i + (10-1-j) * 10 ) for i, j in H.nodes() )
nx.draw_networkx(H, pos=pos, labels= labels)

plt.axis('off')
plt.show()


# In[7]:


#labels for color
n = G.number_of_nodes
labels={}
for i in range(len(players)): 
    labels[i] = '%s' % (players[i])
print (labels)
labels[1]


# In[8]:


#node colors
group_color = []
for pos in range(len(players)):
    if   labels[pos] == 'Cooperator':       
        color ='orange' 
        group_color.append(color)
    elif labels[pos] == 'Defector':          
        color ='green' 
        group_color.append(color)
    elif labels[pos] == 'Adaptive':       
        color ='grey' 
        group_color.append(color)
    elif labels[pos] == 'Handshake':  
        color ='purple' 
        group_color.append(color)
    elif labels[pos] == 'Tricky Cooperator':
        color ='pink' 
        group_color.append(color)
    elif labels[pos] == 'Hopeless': 
        color ='darkblue' 
        group_color.append(color)
    elif labels[pos] == 'Desperate':    
        color ='gold' 
        group_color.append(color)
    elif labels[pos] == 'Geller':
        color ='blue' 
        group_color.append(color)
    elif labels[pos] == 'Tit For Tat':
        color ='skyblue' 
        group_color.append(color)
    elif labels[pos] == 'LookerUp':    
        color ='indianred' 
        group_color.append(color)

group_color


# In[9]:


pos = nx.random_layout(G)

nx.draw_networkx_nodes(G, pos, node_color = group_color ,node_size = 150)

# for the edges
nx.draw_networkx_edges(G, pos, width=2, alpha = 0.5)
nx.draw
nx.draw_networkx_labels(G,pos,labels=labels,font_size=10)
plt.show()


# In[10]:


print(G.edges)
G.edge_attr_dict_factory()


# In[11]:


matches = len(G.edges)
edges = G.edges
scores = []

tournament = axl.Tournament(players, edges=edges)
results = tournament.play(processes=1)
scores.append(zip(*results.scores))


# In[12]:


#Top ten
results.ranked_names[:10]


# In[1]:


plot = axl.Plot(results)


# In[14]:


#viewing the outputs of tournaments with a large number of strategies:
_, ax = plt.subplots()
title = ax.set_title("Payoff")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid()
p = plot.boxplot(ax=ax)
p.show()


# In[15]:


print(results.payoff_matrix)


# In[16]:


_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid(color='white')
p = plot.payoff(ax =ax)
p.show()


# In[17]:


#Distributions of wins
_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid()
p = plot.winplot(ax=ax)
p.show()


# In[18]:


_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid()
p = plot.sdvplot(ax =ax)
p.show()


# In[19]:


_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid(color='white')
p = plot.pdplot(ax =ax,title="Payoff differences ")
p.show()


# In[20]:


#Moran Process 
edges = G.edges()
grp = Graph(edges)
mp = axl.MoranProcess(players = players,interaction_graph = grp, turns = 200)
mp.play()
print("The winner is:", mp.winning_strategy_name)


# In[ ]:


# Plot the results MP
player_names = mp.populations[0].keys()

plot_data = []
labels = []
for name in player_names:
    labels.append(name)
    values = [counter[name] for counter in mp.populations]
    plot_data.append(values)
    domain = range(len(values))

plt.stackplot(domain, plot_data, labels=labels)
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Number of Individuals")
plt.show()

mp.populations_plot()

