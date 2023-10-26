import dash
from dash import dcc, html
import dash_loading_spinners as dls
import dash_bootstrap_components as dbc
import os
from analysis import *

# Get list of subfolders in the directory
dir_path = "fronts/artificial/"
subfolders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
subfolders.append("_REALWORLD")
subfolders.sort(reverse=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
experimentName="ASLETISMAC_LA_X12_Y12_P1_K3_BF"


app.layout = html.Div([
    html.H1("Graph Analyzer"),
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='subfolder-dropdown',
                options=[{'label': i, 'value': i} for i in subfolders],
                placeholder="Select a subfolder",
    )
        ),
        dbc.Col([
            dbc.Row(html.Div(id='dropdown-output-container'),
            ),
            dbc.Row(
            dbc.Button("Analyze", color="primary", className="mr-1", id="analyze-button"),
            
        ),
        ] 
        ),
        
    ])
    ,
    dbc.Row(
            [
                dbc.Col(
                    dls.GridFade(
                        dcc.Graph(id='component-graph-output-container'),
                    ),
                    md=9,
                ),
                dbc.Col(
                    dls.GridFade(
                        html.Div(id="threshold-output-container"),
                )
                ),
            ])    ,
    dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Analyze 2", color="primary", className="mr-1", id="analyze2-button"),
                ),
                dbc.Col(
                    dls.GridFade(
                        dcc.Graph(id='std-graph-output-container'),
                    ),
                    md=9,
                ),
                dbc.Col(
                    dls.GridFade(
                        html.Div(id="nodes-edges-output-container"),
                )
                ),
                dbc.Col(
                    dls.GridFade(
                        dcc.Graph(id='histo-graph-output-container'),
                    ),
                    md=9,
                ),
                dbc.Col(
                    dls.GridFade(
                        dcc.Graph(id='graphfig-output-container'),
                    ),
                    md=9,
                ),
                
            ]),
    
    dbc.Row(
            [
                dbc.Col(
                    dbc.Button("Analyze 3", color="primary", className="mr-1", id="analyze3-button"),
                ),
                dbc.Col(
                    dls.GridFade(
                        dcc.Graph(id='comm-prop-graph-output-container'),
                    ),
                    md=9,
                ),
                dbc.Col(
                    dls.GridFade(
                        dcc.Graph(id='subgraphs-graph-output-container'),
                    ),
                    md=9,
                ),
            ])    ,
    
])


@app.callback(
    dash.dependencies.Output('dropdown-output-container', 'children'),
    [dash.dependencies.Input('subfolder-dropdown', 'value')])
def update_output(value):
    global experimentName
    print(f'Value is {value}')
    experimentName=value
    return 'You have selected "{}"'.format(value)
    

@app.callback(
    [dash.dependencies.Output('component-graph-output-container', 'figure'),
     dash.dependencies.Output('threshold-output-container', 'children')
    ],
    [dash.dependencies.Input('analyze-button', 'n_clicks')])
def update_graph(n_clicks):
    print('Update graph...')
    global commons,commonsNumber,paths2,unObj,df,threshold
    print(f'Val: {experimentName}')
    if not experimentName==None:
        print("Analyzing...")
        if experimentName=="_REALWORLD":
            dir="fronts/realworld/results10000_highDouglas"
            commons,commonsNumber,paths2,unObj,df = doAnalysis(dir,"w-m_",True)
        else:
            commons,commonsNumber,paths2,unObj,df = doAnalysis(dir_path,experimentName,False)
        print("Done analyzing. Doing getComponentsFigure...")
        fig,threshold,noResult = getComponentsFigure(commons,paths2)
        print(f'Threshold: {threshold}')
        if noResult:
            return fig,'No result'
        else:
            return fig,f'Threshold: {threshold}'
    print('Precheck failed at update_graph')
    return go.Figure(),0.0
    
@app.callback(
    [dash.dependencies.Output('std-graph-output-container', 'figure'),
     dash.dependencies.Output('nodes-edges-output-container', 'children'),
     dash.dependencies.Output('histo-graph-output-container', 'figure'),
     dash.dependencies.Output('graphfig-output-container', 'figure'),
    ],
    [dash.dependencies.Input('analyze2-button', 'n_clicks')])    
def updateStdFigAndCreateGraph(n_clicks):
    print('Update std fig and create graph...')
    variables=['commons','paths2','threshold']
    preCheckPassed=True
    for variable in variables:
        if not (variable in globals() and globals()[variable] is not None):
            preCheckPassed=False
            print('Precheck failed at updateStdFigAndCreateGraph')
            break
    if preCheckPassed:
        global G,communities
        stdFig, foundResolution, communities, G, pos,hasCommunityWithLessThanThreeSolutions=getGraphAndCommunities(commons, paths2, threshold)
        histoFig,graphFig=create_histogram_and_graph_fig(G, pos, communities)
        if foundResolution==-1:
            return stdFig,'No result',histoFig,graphFig
        else:
            return stdFig,f'Nodes: {len(G.nodes)} Edges: {len(G.edges)} Used CPM: {not hasCommunityWithLessThanThreeSolutions} Resolution: {foundResolution} NumCommunities: {np.max(communities)+1}',histoFig,graphFig
    else:
        return go.Figure(),"No computation has been done",go.Figure(),go.Figure()
        

@app.callback(
    [dash.dependencies.Output('comm-prop-graph-output-container', 'figure'),
     dash.dependencies.Output('subgraphs-graph-output-container', 'figure'),
    ],
    [dash.dependencies.Input('analyze3-button', 'n_clicks')])    
def updateCommPropAndSubGraphs(n_clicks):
    print('Update comm prop and subgraphs...')
    variables=['G','communities']
    preCheckPassed=True
    for variable in variables:
        if not (variable in globals() and globals()[variable] is not None):
            preCheckPassed=False
            print('Precheck failed at updateCommPropAndSubGraphs')
            break
    if preCheckPassed:
        communityData,figs=create_data_frame(G, communities)
        commPropertyFig = create_scatter_plot(communityData)
        subGraphFigs=create_combined_graph(G, communities,figs)
        return commPropertyFig,subGraphFigs
    else:
        return go.Figure(),go.Figure()
    

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")
