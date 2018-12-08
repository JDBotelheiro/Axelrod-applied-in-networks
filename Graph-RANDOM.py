#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#Estrategias disponiveis
strategies = [s() for s in axl.strategies]
axl.seed(0)  


# In[3]:


#Defining players
strategies = [axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp(),axl.Cooperator(), axl.Defector(),axl.Adaptive(),axl.Handshake(),axl.TrickyCooperator(),axl.Hopeless(),axl.Desperate(),axl.Geller(),axl.TitForTat(),axl.LookerUp()]

players = random.sample(strategies, 50)
plays = players


# In[5]:


G = nx.random_regular_graph(3, len(players))


# In[6]:


#labels for color
n = G.number_of_nodes
labels={}
for i in range(len(players)): 
    labels[i] = '%s' % (players[i])
print (players)

#for graph
n = G.number_of_nodes
sec={}
for i in range(len(plays)): 
    sec[i] = '%s%s' % (1+i,plays[i])
print (sec)


# In[58]:


#node colors
group_color = []
for pos in range(len(players)):
    if labels[pos] == 'Cooperator': color = 'orange' 
    elif labels[pos] == 'Defector': color = 'green' 
    elif labels[pos] == 'Geller': color = 'blue' #changed with 'Aggravater'
    elif labels[pos] == 'Handshake': color = 'purple' 
    elif labels[pos] == 'Tricky Cooperator': color = 'pink' 
    elif labels[pos] == 'Hopeless': color = 'darkblue' 
    elif labels[pos] == 'Desperate': color = 'gold' 
    elif labels[pos] == 'Tit For Tat': color = 'skyblue' 
    elif labels[pos] == 'LookerUp': color = 'indianred' 
    elif labels[pos] == 'Adaptive': color = 'grey' 
    group_color.append(color)


# In[60]:


pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_color = group_color ,node_size = 150)

# for the edges
nx.draw_networkx_edges(G, pos, width=2, alpha = 0.5)
nx.draw
#nx.draw_networkx_labels(G,pos,labels= labels,font_size=10)
plt.show()


# In[61]:


print(G.edges)


# In[64]:


matches = len(G.edges)
edges = G.edges
scores = []

tournament = axl.Tournament(players, edges=G.edges())
results = tournament.play(processes=1)
scores.append(zip(*results.scores))

#eco = axl.Ecosystem(results)
#eco.reproduce(35) # Evolve the population over 100 time steps


# In[66]:


#Top ten
results.ranked_names[:10]


# In[68]:


plot = axl.Plot(results)


# In[72]:


#viewing the outputs of tournaments with a large number of strategies:
_, ax = plt.subplots()
title = ax.set_title("Payoff")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid()
p = plot.boxplot(ax=ax)
p.show()
plt.savefig("r2.png", format="PNG")


# In[78]:


_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid(color='white')
p = plot.payoff(ax =ax)
p.show()
plt.savefig("r3.png", format="PNG")


# In[77]:


#Distributions of wins
_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid()
p = plot.winplot(ax=ax)
p.show()
plt.savefig("r4.png", format="PNG")


# In[80]:


_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid()
p = plot.sdvplot(ax =ax)
p.show()
plt.savefig("r5.png", format="PNG")


# In[82]:


_, ax = plt.subplots()
title = ax.set_title("Payoff differences ")
xlabel = ax.set_xlabel('Strategies')
grid = ax.grid(color='white')
p = plot.pdplot(ax =ax,title="Payoff differences ")
p.show()
plt.savefig("r6.png", format="PNG")


# In[83]:


#plot = axl.Plot(results)
#stackplot = plot.stackplot(eco, logscale=False);
#stackplot.savefig("logo-raw.png", dpi=400)


# In[84]:


#Moran Process 
edges = G.edges()
grp = Graph(edges)
mp = axl.MoranProcess(players = players,interaction_graph = grp, turns = 200)
mp.play()
print("The winner is:", mp.winning_strategy_name)


# In[86]:


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
plt.savefig("r7.png", format="PNG")


# In[ ]:





# In[ ]:




