"""
A dash app that allows the user to (optionally) upload new data (eventually model?), select a 
(local/huggingface) dataset, select a (local/huggingface) model, embed the data using the model,
and cache raw/processed results for quick access later if new processing is applied to the data.

"""
# IMPORTS
from dash import Dash, html, dcc, Output, Input
import plotly.express as px
import numpy as np
import pandas as pd
from ml_utils.caching import EmbeddedTextLoader

# GLOBAL VARS
APP = Dash(__name__)
RNG = np.random.RandomState(0)
# df = pd.DataFrame(RNG.randint(0, 100, (30, 5)), columns=['x', 'y', 'z', 'c1', 'c2'])
# df = pd.DataFrame(dx.load_from_disk('train3')['train'].to_pandas()['MAX'].tolist())
proj_name = 'rotten_tomatoes'#'mytrain1'
model_name = 'facebook/opt-125m'
emb_name = f'{proj_name}/{model_name}'
DF = pd.DataFrame(EmbeddedTextLoader.load(emb_name)['train'].to_pandas()['MAX'].tolist())
DF = DF.iloc[:, :5].rename(columns=dict(enumerate(['x', 'y', 'z', 'c1', 'c2'])))
print(f'{DF=}')
scatter_opts = dict(data_frame=DF, x='x', y='y', height=400)
hist_opts = dict(height=400)

# FUNCTIONS
def get_new_scatter(col='c1'):
    print(f'{col=}')
    assert col in DF.columns
    fig = px.scatter(color=col, **scatter_opts)
    # fig.update_traces(mode='')
    return fig

def get_new_hist(maxx, maxy, col='c1'):
    print(f'{maxx=}, {maxy=}, {col=}')
    assert col in DF.columns
    fig = px.histogram(
        data_frame=DF[ (DF['x'] <= maxx) & (DF['y'] <= maxy) ], 
        x='x', 
        # color='green' if col == 'c1' else 'red',
        **hist_opts)
    return fig

# LAYOUT DEF
APP.layout = html.Div([
    
    html.H1(children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for your data.
    '''),
    
    html.Div([
        dcc.Graph(
            id='embedding-scatter',
            figure=get_new_scatter(),
        ),
    ],  style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(
            id='embedding-hist',
            figure=get_new_hist(DF.x.max(), DF.y.max()),
        ),
    ],  style={'width': '49%', 'display': 'inline-block'}),
    
    html.Br(),
    html.Div([
        "Pick a color column: ",
        dcc.Dropdown(
            id='df-col-dropdown',
            options=['c1', 'c2'],
            value='c1',
        ),
    ],  style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Br(),
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False,
        ),
        html.Div(id='output-data-upload'),
    ]),
])

# CALLBACKS
@APP.callback(
    Output(component_id='embedding-scatter', component_property='figure'),
    Input(component_id='df-col-dropdown', component_property='value'),
)
def update_embedding_scatter(col):
    fig = get_new_scatter(col)
    return fig

# CALLBACKS
@APP.callback(
    Output(component_id='embedding-hist', component_property='figure'),
    Input(component_id='df-col-dropdown', component_property='value'),
    Input(component_id='embedding-scatter', component_property='hoverData'),
)
def update_embedding_hist(col, hover_data):
    print(f'{col=}, {hover_data=}')
    maxx = hover_data['points'][0]['x'] if hover_data is not None else np.inf
    maxy = hover_data['points'][0]['y'] if hover_data is not None else np.inf
    fig = get_new_hist(maxx, maxy, col)
    return fig

if __name__ == "__main__":
    APP.run_server(debug=True)