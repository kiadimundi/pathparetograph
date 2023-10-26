# %% 
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from setfun import *
import pickle
import multiprocessing
from graphfun import *
# from clusterring import *
from simplification import *
from collections import Counter
from pymoo.algorithms.moo.nsga2 import NonDominatedSorting
import warnings
from plotly.subplots import make_subplots


def nan_ptp(a):
    return np.ptp(a[np.isfinite(a)])

# b = (a - np.nanmin(a))/nan_ptp(a)

def doAnalysis(dir,name,realworld=False):
    # # Load the .mat file and access the MATLAB array
    # mat_contents = sio.loadmat('medianObjectivesAll.mat')
    # matlab_array = mat_contents['medianObjectivesAll']

    # # Convert to a NumPy array (if needed, as sometimes SciPy may return a structured array)

    # medObjs = np.squeeze(matlab_array)  # Use np.squeeze to remove any unnecessary dimensions



    # mat_contents = sio.loadmat('medianPathsAll.mat')
    # matlab_array = mat_contents['medianPathsAll']

    # # Convert to a NumPy array (if needed, as sometimes SciPy may return a structured array)
    # medPaths = np.squeeze(matlab_array)  # Use np.squeeze to remove any unnecessary dimensions
    # name = "ASLETISMAC_LA_X11_Y11_PM_K2_BF"
    # name= "ASLETISMAC_LA_X30_Y30_PM_K3_BT"
    if realworld:
        medObjs=readObjectives(dir,name+"obj","csv",";")
        medPaths=readRealWorldPath(dir,name+"set_","csv",";")
    else:
        dir=dir+name
        medPaths=readMorpJson(dir,name+".ps",False)
        medObjs=readObjectives(dir,name,"pf")



    # Assuming medianPathsAll and medianObjectivesAll are NumPy arrays

    paths = medPaths
    unObj = medObjs
    
    data=unObj
    variable_names = ['Length', 'Ascent', 'Time', 'Smoothness', 'Accidents']
    df = pd.DataFrame(data, columns=variable_names)

    # commons = np.zeros((len(paths), len(paths)))
    commons = np.empty((len(paths), len(paths)))
    commons[:] = np.nan
    commonsNumber= []
    # for i in tqdm(range(len(ia) - 1), desc="Processing sublists..."):
    numbers = range(len(paths))
    # numbers = range(2)
    with multiprocessing.get_context("fork").Pool(processes=multiprocessing.cpu_count()) as pool:
        # args = [(number, paths2, epsilon) for number in numbers]
        args = zip(numbers, [paths]*len(paths))
        # results = list(tqdm(executor.map(task_function, numbers), total=len(numbers)))
        results = list(tqdm(pool.imap(helper, args), total=len(numbers)))
        
        
    # for p1 in tqdm(range(len(paths)), desc="Processing sublists..."):
    #     for p2 in range(p1 + 1, len(paths)):
    #         pCoords1 = np.array(paths[p1]).astype('float64')
    #         pCoords2 = np.array(paths[p2]).astype('float64')
    #         # inters = np.intersect(pCoords1, pCoords2, axis=0)
    #         # changeProb = 1 - (len(np.setxor1d(pCoords1, pCoords2)) / (len(pCoords1) + len(pCoords2)))
    #         # changeProb = 1 - (max(len(np.setxor1d(pCoords1, pCoords2)) - 2, 0) / (len(unionSet(pCoords1,pCoords2)) - 2))
    #         # commons[p1, p2] = len(inters) - 2
    #         numberOfOptionsToChange = len(commonsublists(pCoords1,pCoords2))
    #         # numberOfOptionsToChange2 = len(commonsublists_perf(pCoords1,pCoords2))
    #         commons[p1,p2] = numberOfOptionsToChange
    #         commonsNumber.append(numberOfOptionsToChange)

    

    # Assuming commonsNumber is a 1D NumPy array or a Python list
    # commons = np.append(commons, np.zeros((1, commons.shape[1])), axis=0)

    

    # Assuming you have a function called 'normalize' to normalize the commons array, replace it with your implementation
    # commons = normalize(commons, 'range')
    # You can use the following lines to normalize the commons array between 0 and 1
    commons=results
    commons.reverse()
    max_length = max(len(arr) for arr in commons)

        # Pad the shorter arrays with NaNs
    padded_arrays = [np.pad(arr, (0, max_length - len(arr)), mode='constant', constant_values=0) for arr in commons]
    commons=np.column_stack(padded_arrays)
    commons=np.vstack([commons,[0]*np.size(commons,axis=1)])
    commons = (commons - np.nanmin(commons)) / (np.nanmax(commons) - np.nanmin(commons))
    commonsNumberTemp=commons.flatten()
    filter_arr = commonsNumberTemp != 0

    commonsNumber = commonsNumberTemp[filter_arr]
    test=commons==0
    commons[test]=np.nan
    if np.max(commonsNumber) == np.min(commonsNumber):
        test=commons==np.max(commonsNumber)
        commons[test]=1
    else:
        commons = (commons - np.nanmin(commons)) / (np.nanmax(commons) - np.nanmin(commons))
    return commons,commonsNumber,paths,unObj,df

def compute_1_n_commonSublists(p1,paths):
    # for p1 in tqdm(range(len(paths2))):
    pCoords1 = np.array(paths[p1]).astype('float64')
    until=len(paths)
    commis=[]
    for p2 in range(p1+1,until):
        # time.sleep(0.1)
        
        pCoords2 = np.array(paths[p2]).astype('float64')
        # pa1=rdp(np.column_stack((paths2[p1][:,0],paths2[p1][:,1])),epsilon)
        # pa2=np.column_stack((paths2[p2][:,0],paths2[p2][:,1]))
        commonsP1P2=len(commonsublists(pCoords1,pCoords2))
        commis.append(commonsP1P2)
        # distance_matrix[p1,p2]=distanceDf
    return commis

def helper(args):
    return compute_1_n_commonSublists(*args)

def getBoxPlot(commonsNumber):
    
    # Create the box plot using plotly graph_objects
    boxPlot = go.Figure()
    boxPlot.add_trace(go.Box(y=commonsNumber, boxpoints='all', jitter=0.3, pointpos=-1.8, name='Box Plot'))

    # Update the layout (optional)
    boxPlot.update_layout(title='Box Plot Example', yaxis_title='Values')

    # Show the plot
    # fig.show()
    return boxPlot

def getThreeObjectivesFig(commons,unObj):

    commons[commons == 0] = np.nan
    pure = commons[~np.isnan(commons)].flatten()

    firstScaleMin = np.min(pure)
    firstScaleMax = np.max(pure)

    # threshold = 0
    threshold = np.quantile(pure, 0.99)  # Use this line if you want to compute the threshold from the data

    # Assuming you have a function called 'filterandwritegraph' to process commons and threshold and save to a CSV file
    # replace it with your implementation
    # filterandwritegraph(commons, threshold, 'pareto_graph/graph.csv')

    # quantis = [firstScaleMin, *np.quantile(pure, np.linspace(0.1, 1, 10)), firstScaleMax]

    # amq = []
    # for qi in range(1, len(quantis)):
    #     amq.append(np.sum((commons > quantis[qi-1]) & (commons <= quantis[qi])))

    # plotted = np.sum(commons > threshold)

    # Assuming rescale function rescales a given array to a new range
    # def rescale(arr, new_min, new_max):
    #     return new_min + (arr - arr.min()) * (new_max - new_min) / (arr.max() - arr.min())

    # If you want to rescale 'commons' between secondScaleMin and secondScaleMax, use the following line
    # commons = rescale(commons, secondScaleMin, secondScaleMax)

    secondScaleMin = 0.2
    secondScaleMax = 0.8




    numObj = 3
    count = 0
    graph3dfig = go.Figure()

    for p1 in tqdm(range(commons.shape[0]),desc="Processing commons array..."):
        for p2 in range(p1 + 1, commons.shape[1]):
            fPath = unObj[p1, :numObj]
            tPath = unObj[p2, :numObj]
            pa = np.vstack((fPath, tPath))
            val = commons[p1, p2]
            if val > threshold:
                resVal = secondScaleMin + ((val - threshold) / (firstScaleMax - threshold)) * (secondScaleMax - secondScaleMin)
                graph3dfig.add_trace(go.Scatter3d(x=pa[:, 0], y=pa[:, 1], z=pa[:, 2], mode='lines', name='Line {}'.format(count)))
                graph3dfig.data[-1].line.width = (resVal + 1) ** 2
                graph3dfig.data[-1].line.color = 'rgba(1, 0, 0, {})'.format(1 - resVal)

    graph3dfig.add_trace(go.Scatter3d(x=unObj[:, 0], y=unObj[:, 1], z=unObj[:, 2], mode='markers', name='UnObj Points'))
    d=go.scattergl

    graph3dfig.update_layout(showlegend=False)
    # fig.show()
    return graph3dfig

def getParCoordsFig(df):

    parcoordsfig = go.Figure(data=
        go.Parcoords(
            line_color='blue',
            dimensions = list([
                dict(
                    # constraintrange = [4,8],
                    label = 'Length', values = df['Length']),
                dict(
                    label = 'Ascent', values = df['Ascent']),
                dict(
                    label = 'Time', values = df['Time']),
                dict(
                    label = 'Smoothness', values = df['Smoothness']),
                dict(
                    label = 'Accidents', values = df['Accidents'])
            ])
        )
    )

    parcoordsfig.update_layout(
        plot_bgcolor = 'white',
        paper_bgcolor = 'white'
    )
    return parcoordsfig

def saveAnalysis(commons,commonsNumber,paths2,unObj,df,file='data.pickle'):
    with open(file, 'wb') as f:
        pickle.dump(commons, f)
        pickle.dump(commonsNumber, f)
        pickle.dump(paths2, f)
        pickle.dump(unObj, f)
        pickle.dump(df, f)
    
def loadAnalysis(file='data.pickle'):
    with open(file, 'rb') as f:
        commons = pickle.load(f)
        commonsNumber= pickle.load(f)
        paths2 = pickle.load(f)
        unObj = pickle.load(f)
        df = pickle.load(f)
    return commons,commonsNumber,paths2,unObj,df

def getPathsOnMapFigure(pathsToPlot):
    fig = go.Figure(go.Scattermapbox())
    pathToPlot=pathsToPlot
    for p in range(len(pathToPlot)):
        p1=pathToPlot[p]
        fig.add_trace(go.Scattermapbox(
            mode = "markers+lines",
            lon = p1[:,1],
            lat = p1[:,0],
            # line={wid},
            marker = {'size': 4}))

    # latlim =[52.3408   52.5558];
    # lonlim =[13.1274   13.5916];
    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0},
        mapbox = {
            'center': {'lon': 13.5916, 'lat': 52.5558},
            'style': "open-street-map",
            'center': {'lon': 13.1274, 'lat': 52.3408},
            'zoom': 4
            },
        showlegend=False
        )
    return fig


def getComponentsFigure(commons, paths2, startIndex=500, endIndex=999, divisor=1000.0):
    co = []
    components = []
    myrange = reversed(range(startIndex, endIndex))
    for i in tqdm(myrange):
        G = createGraph(commons, paths2, thresholdQuantile=i/divisor, disableTqdm=True)
        numEdges = len(G.edges)
        co.append(numEdges)
        numComp = getNumberOfConnectedComponents(G)
        components.append(numComp)

    myrange = reversed(range(startIndex, endIndex))
    newList = [x / divisor for x in list(myrange)]
    trace1 = go.Scatter(x=newList, y=co, mode='lines', name='Edges')

    # Create the second trace using the second data set and assign it to a secondary y-axis
    trace2 = go.Scatter(x=newList, y=components, mode='lines', name='Components', yaxis='y2')

    # Create the layout with a secondary y-axis
    layout = go.Layout(
        title=dict(text='Edges and connected components against Threshold', x=0.5, y=0.95),
        xaxis=dict(title="$$\\tau$$"),
        yaxis=dict(title='$$\\lvert E \\rvert$$'),
        yaxis2=dict(title='$$\\lvert C \\rvert$$', overlaying='y', side='right'),
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(xanchor="left", x=1.07)
    )

    # Create the figure and add both traces
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    # Find threshold to have only one connected component but least number of edges
    comparr=np.array(components)
    indices = np.where(comparr == 1)

    # Get the last index (if the element exists in the array)
    noResult=False
    if len(indices[0]) > 0:
        first_index = indices[0][0]
    else:
        first_index=0
        print("There is no element where the number of connected components is 1")
        noResult=True

    foundThreshold=(list(reversed(range(startIndex,endIndex)))[first_index])/divisor
    # print(f"Threshold is {foundThreshold}")

    return fig,foundThreshold,noResult


def getGraphAndCommunities(commons, paths2, foundThreshold, startIndex=1, endIndex=300, divisor=1000.0):
    G = createGraph(commons, paths2, thresholdQuantile=foundThreshold)
    print(G.number_of_nodes(), G.number_of_edges())
    
    pos = getLayout(G, useKamada=False)

    commis=[]
    stds=[]
    myrange=range(startIndex,endIndex)
    for i in tqdm(myrange):
        communities=getCommunities(G,resolution=i/divisor,cpm=True)
        nComms=np.max(communities)
        commis.append(nComms+1)
        stds.append(np.std(communities))

    myrange=range(startIndex,endIndex)
    newList = [x / divisor for x in list(myrange)]
    trace1 = go.Scatter(x=newList, y=commis, mode='lines', name='Communities')
    trace2 = go.Scatter(x=newList, y=stds, mode='lines', name='Std', yaxis='y2')

    layout = go.Layout(
        title=dict(text='Communities', x=0.5, y=0.95),
        yaxis=dict(title='# Communities'),
        yaxis2=dict(title='Std', overlaying='y', side='right')
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    communityarray=np.array(commis)
    indices = np.where(communityarray > 3)
    moreThanThreeComms=True
    if len(indices[0]) > 0:
        first_index = indices[0][0]
    else:
        print("There is no element where the number of communities is greater than 3")
        first_index=0
        moreThanThreeComms=False
        return fig, -1, communities, G, pos, True

    foundResolution=(list(range(startIndex,endIndex))[first_index])/divisor
    print(f"Resolution is {foundResolution}")

    communities=getCommunities(G,resolution=foundResolution, iterations=10,cpm=True)
    frequency_counter = Counter(communities)
    frequency_dict = dict(frequency_counter)

    hasCommunityWithLessThanThreeSolutions=False
    for key, value in frequency_dict.items():
        print(f"{key}: {value} times")
        if value<3:
            hasCommunityWithLessThanThreeSolutions=True

    if hasCommunityWithLessThanThreeSolutions:
        warnings.warn("Partitioning resulted in at least one community with less than 3 members. Trying Modularity instead of CPM...")
        communities=getCommunities(G,resolution=foundResolution, iterations=2,cpm=False)

    for key, value in frequency_dict.items():
        print(f"{key}: {value} times")

    
    return fig, foundResolution, communities, G, pos, hasCommunityWithLessThanThreeSolutions
        

def create_histogram_and_graph_fig(G, pos, communities):
    np.max(communities)
    histogram_trace = go.Histogram(
        x=communities,
        xbins=dict(start=min(communities), end=max(communities), size=1),
        marker=dict(color='blue'),
        opacity=0.75
    )

    layout = go.Layout(
        title=dict(text='Histogram Example', x=0.5, y=0.95),
        xaxis=dict(title='Values'),
        yaxis=dict(title='Frequency'),
    )

    histogramFig = go.Figure(data=[histogram_trace], layout=layout)

    graphFig = graphfig(G, pos=pos, clusterComp=communities)

    return histogramFig, graphFig

def create_data_frame(G, communities):
    commarr = np.array(communities)
    columns = ['Community', 'Density', 'Avg. Cluster Coefficient', 
               'Group Betweeness Centrality', 'Graph Degree Centrality']
    
    data_frame = pd.DataFrame(columns=columns)
    xvals = np.array(range(np.max(communities)+1))
    figs=[]
    for i in xvals:
        indices = np.where(commarr == i)[0]
        subg = G.subgraph(indices)
        
        data_frame.loc[len(data_frame)] = [
            i,
            nx.density(subg),
            nx.average_clustering(subg, weight="weight"),
            nx.group_degree_centrality(G, S=list(subg.nodes)),
            graphDegree(subg)
        ]
        indices = np.where(commarr == i)[0]
        subGpos = getLayout(subg)
        fig = graphfig(subg,pos=subGpos,clusterComp=communities)
        figs.append(fig)
        
    return data_frame,figs


def create_scatter_plot(data_frame):
    fig = go.Figure()
    markers = ["diamond", "pentagon", "square", "cross"]
    
    xvals=data_frame['Community'].to_numpy(dtype=int)
    
    for i, column in enumerate(data_frame.columns):
        if column != 'Community':
            if column != 'Graph Degree Centrality':
                scatter_plot = go.Scatter(
                    x=data_frame['Community'], 
                    y=data_frame[column], 
                    mode='markers', 
                    name=column, 
                    marker=dict(size=10, symbol=f'{markers[i%len(markers)]}{"-open"}')
                )
            else:
                scatter_plot = go.Scatter(
                    x=data_frame['Community'], 
                    y=data_frame[column], 
                    mode='markers', 
                    name=column, 
                    yaxis='y2',
                    marker=dict(size=10, symbol=f'{markers[i%len(markers)]}{"-open"}')
                )
            fig.add_trace(scatter_plot)

    # Add labels and title
    fig.update_layout(
        title=dict(text='Community Characteristics', x=0.5, y=0.98),
        yaxis_title='Value',
        xaxis=dict(title='Community', tickmode='array', tickvals=xvals),
        yaxis2=dict(title='Graph Degree Centrality', overlaying='y', side='right', range=[0,0.055]),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.update_layout(legend=dict(xanchor="left", x=1.07))
    
    return fig


def create_combined_graph(G, communities, figs):
    commarr = np.array(communities)
    # figs = []
    subplotsize = int(np.ceil(np.sqrt(len(figs))))
    
    xvals = np.array(range(np.max(communities)+1))
    figs=[]
    for i in xvals:
        indices = np.where(commarr == i)[0]
        subg = G.subgraph(indices)
        subGpos = getLayout(subg)
        fig = graphfig(subg, pos=subGpos, clusterComp=communities)
        figs.append(fig)
        
    figcomb = make_subplots(rows=subplotsize, cols=subplotsize,
                            subplot_titles=tuple([f'Community {x}' for x in range(0,np.max(communities)+1) ]))
    
    for i, subplot in enumerate(figs):
        for trace in subplot.data:
            rowind = np.unravel_index([i], (subplotsize,subplotsize))[0][0]+1
            colind = np.unravel_index([i], (subplotsize,subplotsize))[1][0]+1
            figcomb.add_trace(trace, row=rowind, col=colind)
            
    figcomb.update_layout(showlegend=False)
    figcomb.update_layout(margin=dict(l=0, r=0, b=0, t=30))
    figcomb.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    figcomb.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    return figcomb

def doNDSort(data_frame):
    F = data_frame[['Density', 'Avg. Cluster Coefficient','Group Betweeness Centrality', 'Graph Degree Centrality']].values

    sorting = NonDominatedSorting().do(F, only_non_dominated_front=True)
    # print(sorting)
    return sorting
