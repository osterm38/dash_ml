# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
# IMPORTS
from dash import Dash, html, dcc, Output, Input
import datasets as dx
import plotly.express as px
import numpy as np
import pandas as pd
from dash_ml.caching import EmbeddedTextLoader

# GLOBAL VARS
app = Dash(__name__)
rng = np.random.RandomState(0)
# df = pd.DataFrame(rng.randint(0, 100, (30, 5)), columns=['x', 'y', 'z', 'c1', 'c2'])
# df = pd.DataFrame(dx.load_from_disk('train3')['train'].to_pandas()['MAX'].tolist())
proj_name = 'rotten_tomatoes'#'mytrain1'
model_name = 'facebook/opt-125m'
emb_name = f'{proj_name}/{model_name}'
df = pd.DataFrame(EmbeddedTextLoader.load(emb_name)['train'].to_pandas()['MAX'].tolist())
df = df.iloc[:, :5].rename(columns=dict(enumerate(['x', 'y', 'z', 'c1', 'c2'])))
print(f'{df=}')
scatter_opts = dict(data_frame=df, x='x', y='y', height=400)
hist_opts = dict(height=400)

# FUNCTIONS
def get_new_scatter(col='c1'):
    print(f'{col=}')
    assert col in df.columns
    fig = px.scatter(color=col, **scatter_opts)
    # fig.update_traces(mode='')
    return fig

def get_new_hist(maxx, maxy, col='c1'):
    print(f'{maxx=}, {maxy=}, {col=}')
    assert col in df.columns
    fig = px.histogram(
        data_frame=df[ (df['x'] <= maxx) & (df['y'] <= maxy) ], 
        x='x', 
        # color='green' if col == 'c1' else 'red',
        **hist_opts)
    return fig

# LAYOUT DEF
app.layout = html.Div([
    
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
            figure=get_new_hist(df.x.max(), df.y.max()),
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

])

# CALLBACKS
@app.callback(
    Output(component_id='embedding-scatter', component_property='figure'),
    Input(component_id='df-col-dropdown', component_property='value'),
)
def update_embedding_scatter(col):
    fig = get_new_scatter(col)
    return fig

# CALLBACKS
@app.callback(
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

class Model:
    """sklearn-like model (API-wise) for my classification bert models?"""
    def __init__(self):
        # store params
        # self.model_ = None
        pass
    
    def fit(self, X, y=None):
        # use run_glue and follow do_fit
        model = self.load_model()
        # self.model_ = model
        trainer = ...
        trainer.train()
        trainer.save()
        return self
    
    def predict(self, X):
        # use evaluate, break into procs 1-1 with gpus
        model = self.load_model()
        # self.model_ = model
        y_pred = ...
        return y_pred
    
    def load_model(self):
        model = ...
        return model

def serve_up():
    """
    - choose model (dropdown)
      - host/await in gpu
    - show 3D embedding space (plot)
      - on standby
    - update (text) data (upload, new data)
      - runs through predictions/embeddings
        - cache/checksum each, since expensive?
      - (re)train a x-dim -> y-dim embedding
      - a.(re)train a y-dim -> 3-dim
        - (re)create dataframe with xyz+hovering/coloring options
        - change colorings (dropdown)
        - store this
      - b.perform y-dim topic extraction/salience something or other?
        - tx2 borrowings?
        - bertopic?
    """

def main():
    # instantiate model
    m = Model() # or any other model/pipeline from sklearn

    # read in data
    ds = ...
    # format data
    X, y = ds, None

    # a. train up the model
    m = m.fit(X, y)
    # b. and/or predict with model
    y_pred = m.predict(X)

# KICK OFF THE SERVER
if __name__ == "__main__":
    # main()
    app.run_server(debug=True)
