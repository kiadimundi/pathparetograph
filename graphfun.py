# %%
import igraph as ig
# import matplotlib.pyplot as plt
import networkx as nx
import leidenalg
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import colorlover as cl
import tqdm
from fa2 import ForceAtlas2
import pickle

from analysis import *
# from sklearn.cluster import AgglomerativeClustering

# g = ig.load("visible.gml")
#fig, ax = plt.subplots(figsize=(5,5))
# ig.plot(
#     g,
#     target=ax,
#     layout="circle", # print nodes in a circular layout
#     vertex_size=0.1,
#     # vertex_color=["steelblue" if gender == "M" else "salmon" for gender in g.vs["gender"]],
#     vertex_frame_width=4.0,
#     vertex_frame_color="white",
#     # vertex_label=g.vs["name"],
#     # vertex_label_size=7.0,
#     # edge_width=[2 if married else 1 for married in g.es["married"]],
#     # edge_color=["#7142cf" if married else "#AAA" for married in g.es["married"]]
# )
# %%
# n1 = g.vs[0]
def graphfig(H,pos=None,clusterComp=None):
    # H = nx.read_gml("visible.gml")
    G = H
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        if pos==None:
            x0 = G.nodes[edge[0]]['graphics']["x"]
            y0 = G.nodes[edge[0]]['graphics']["y"]
            z0 = G.nodes[edge[0]]['graphics']["z"]
            x1 = G.nodes[edge[1]]['graphics']["x"]
            y1 = G.nodes[edge[1]]['graphics']["y"]
            z1 = G.nodes[edge[1]]['graphics']["z"]
        else:
            x0 = pos[edge[0]][0]
            y0 = pos[edge[0]][1]
            z0 = pos[edge[0]][0]
            x1 = pos[edge[1]][0]
            y1 = pos[edge[1]][1]
            z1 = pos[edge[1]][0]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_z.append(z0)
        edge_z.append(z1)
        edge_z.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,# z=edge_z,
        line=dict(width=0.1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_z = []
    node_clusters=[]
    for node in G.nodes():
        if pos==None:
            x = G.nodes[node]['graphics']["x"]
            y = G.nodes[node]['graphics']["y"]
            z = G.nodes[node]['graphics']["z"]
        else:
            x = pos[node][0]
            y = pos[node][1]
            z = pos[node][0]
        # cluster = G.nodes[node]['Cluster']
        if clusterComp==None:
            cluster=0
            clustermin=0
            clustermax=0
        else:
            cluster = clusterComp[node]
            clustermin=np.min(clusterComp)
            clustermax=np.max(clusterComp)
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_clusters.append(int(cluster))

    node_clusters_str=[]
    for i in node_clusters:
        node_clusters_str.append(str(i))
    # node_trace.marker.color=node_clusters_str
    nc=np.array(node_clusters)
    # nc=np.interp(nc, (np.min(nc), np.max(nc)), (-1, 1))
    my_array = np.hstack((np.array(node_x)[:, np.newaxis],np.array(node_y)[:, np.newaxis],nc[:, np.newaxis]))
    df = pd.DataFrame(my_array, columns = ['X','Y','Cluster'])
    # Get the number of unique clusters
    num_clusters = len(df['Cluster'].unique())

    # Generate a custom colorscale with perceptually distinct colors for the given number of clusters
    colors = cl.scales['12']['qual']['Paired']  # Use 'Paired' colorscale as an example, adjust as needed
    custom_colorscale = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
                        '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#8c6d31', '#393b79', '#637939', '#843c39',
                        '#7b4173', '#5254a3', '#bd9e39']
    
    node_trace = go.Scatter(
        x=df['X'], y=df['Y'],
        # x=node_x, y=node_y,#, z=node_z,
        mode='markers',
        customdata=clusterComp,
        hovertemplate =
        '<i>X</i>: %{x}'+
        '<br><i>Y</i>: %{y}<br>'+
        '<b>%{customdata}</b>',
        # text = ['Custom text {}'.format(i + 1) for i in range(5)],
        marker=dict(
            showscale=False,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            # colorscale='Viridis',
            colorscale=custom_colorscale,
            # reversescale=True,
            cmin=clustermin,
            cmax=clustermax,
            color=df['Cluster'],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))


    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    node_adjacencies=np.interp(node_adjacencies, (np.min(node_adjacencies), np.max(node_adjacencies)), (5, 25)).tolist()
    node_trace.marker.size = node_adjacencies
    # node_trace.marker.color=['blue','yellow']
    # node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    # plot_bgcolor='white',  # Set the background color of the plot
                    paper_bgcolor='white',
                    title=dict(
                    text='Route Change Graph',  # Specify the title text
                    x=0.5,  # Set the x-coordinate to center the title
                    y=0.95  # Adjust the y-coordinate for vertical positioning
                    ),
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    # annotations=[ dict(
                    #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                    #     showarrow=False,
                    #     xref="paper", yref="paper",
                    #     x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    # fig.show()
    return fig

def createGraph(commons,paths,thresholdQuantile=None,thresholdAbsolute=None, disableTqdm=False):
    assert (thresholdQuantile == None) ^ (thresholdAbsolute == None), "You need to specify either thresholdQuantile XOR thresholdAbsolute"
    G=nx.Graph()
    
    if thresholdAbsolute == None:
        threshold = np.quantile(commons[~np.isnan(commons)].flatten(), thresholdQuantile)
        if not disableTqdm:
            print("Using quantile (",thresholdQuantile,") as threshold")
    else:
        threshold=thresholdAbsolute
        if  not disableTqdm:
            print("Using absolute value of (",thresholdAbsolute,") as threshold")
    
    for p1 in tqdm(range(len(paths)), desc="Creating nodes...",disable=disableTqdm):
        G.add_node(p1)
    # threshold = 0
    for p1 in tqdm(range(len(paths)), desc="Creating edges...",disable=disableTqdm):
        for p2 in range(p1 + 1, commons.shape[1]):
            val =  commons[p1,p2]
            if (val >= threshold) & (val > 0):
                G.add_edge(p1,p2,weight=val)
    return G

def getLayout(G,weight="weight",pos=None,iterations=10000,useKamada=False):
    if useKamada:
        positions = nx.kamada_kawai_layout(G,weight=weight)
    else:
        forceatlas2 = ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution=True,  # Dissuade hubs
                            linLogMode=False,  # NOT IMPLEMENTED
                            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence=1.0,

                            # Performance
                            jitterTolerance=1.0,  # Tolerance
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # NOT IMPLEMENTED

                            # Tuning
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,

                            # Log
                            verbose=True)

        positions = forceatlas2.forceatlas2_networkx_layout(G, pos=pos, iterations=iterations, weight_attr=weight)
    return positions

def getCommunities(H, iterations=10, resolution=0.1, cpm=False):
    h = ig.Graph.from_networkx(H)
    if not cpm:
        part = leidenalg.find_partition(h, leidenalg.ModularityVertexPartition,n_iterations=iterations,weights='weight')
    else:
        part = leidenalg.find_partition(h, leidenalg.CPMVertexPartition,n_iterations=iterations,weights='weight',resolution_parameter=resolution)
    return part.membership

def saveGraphAsGml(G):
    nx.write_gml(G,"out.gml")
    # nx.write_graphml(G,"out.graphml")
    
def getNumberOfConnectedComponents(G):
    if nx.is_directed(G):
        return nx.number_weakly_connected_components(G)
    else:
        return nx.number_connected_components(G)
    
def graphDegree(G):
    # Assuming you have a NetworkX graph 'G'
    # Calculate degree centrality for each node
    closeness_centralities = nx.degree_centrality(G)

    # Find the node with the highest closeness centrality
    v_star = max(closeness_centralities, key=closeness_centralities.get)

    # Get the set of nodes from G
    V = set(G.nodes())

    # Calculate C_D(G) using the formula
    sum_term = sum([closeness_centralities[v_star] - closeness_centralities[v_i] for v_i in V])
    numerator = sum_term
    denominator = (len(V) ** 2) - (3 * len(V)) + 2

    C_D_G = numerator / denominator

    # print("C_D(G) =", C_D_G)
    return C_D_G


# %%
