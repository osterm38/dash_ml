"""
A dash app that allows the user to (optionally) upload new data (eventually model?), select a 
(local/huggingface) dataset, select a (local/huggingface) model, embed the data using the model,
and cache raw/processed results for quick access later if new processing is applied to the data.

"""
# IMPORTS
import base64
from dash import Dash, html, dcc, Output, Input, State
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from ml_utils.caching import MODEL_NAMES, TempLoader, TextLoader, EmbeddedTextLoader, ModelLoader
from ml_utils.embedding import Embedder
from ml_utils.utils import get_logger, zip_dir

# GLOBAL VARS
LOG = get_logger(name=__name__, level='DEBUG')
APP = Dash(__name__)
# TODO: why is print working but not LOG.debug?
print(f'{APP=}') # LOG.debug(f'{APP=}') #TODO: find/replace print with LOG.debug?
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
    assert col in DF.columns
    fig = px.scatter(color=col, **scatter_opts)
    # fig.update_traces(mode='')
    return fig

def get_new_hist(maxx, maxy, col='c1'):
    assert col in DF.columns
    fig = px.histogram(
        data_frame=DF[ (DF['x'] <= maxx) & (DF['y'] <= maxy) ], 
        x='x', 
        # color='green' if col == 'c1' else 'red',
        **hist_opts)
    return fig


# LAYOUT DEF
APP.layout = html.Div([ dcc.Tabs([ dcc.Tab(label='Upload/Download Datasets', children=[
    # html.H1(children='Hello Dash'),
    # html.Div(children='''
    #     Dash: A web application framework for your data.
    # '''),
        html.Div([
            html.H1('Upload Dataset'),
            html.Div('Datasets with same filename (regardless of extension) will be replaced'),
            html.Br(),
            dcc.Upload(
                id='upload-data-input',
                children=html.Div([
                    'Drag and Drop csv/xls/parquet File, or ',
                    html.B('CLICK HERE'),
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
                multiple=False,
            ),
            dcc.Loading(
                children=[
                    html.Br(),
                    html.Div(id='upload-data-output', children=html.Div([])),
                ],
                type='default', #'circle',
                fullscreen=False,
            ),
        ]),
        html.Br(),
        html.Div([
            html.H1('Download Dataset'),
            html.Div('To include generated embeddings, select appropriate model as well as dataset'),
            html.Br(),
            dcc.Dropdown(
                id='download-dataset-selection',
                options=TextLoader.get_existing_names(),
                placeholder='select a dataset to download',
            ),
            html.Br(),
            dcc.Dropdown(
                id='download-model-selection',
                options=ModelLoader.get_existing_names(),
                placeholder='(optionally) select an embedding model',
            ),
            html.Br(),
            # TODO: fix the download of embeddings, for hf datasets, since they don't show up in local cache
            html.Button("Download Dataset", id="download-ds-button"),
            dcc.Loading(
                children=dcc.Download(id="download-dataset"),
                type='default', #'circle',
                fullscreen=False,
            ),
        ]),
    ]), dcc.Tab(label='Model Embeddings', children=[
        html.H1('Embeddings'),
        html.Div('Large datasets/models may take some time to embed'),
        html.Br(),
        dcc.Dropdown(
            id='embed-dataset-selection',
            options=TextLoader.get_existing_names(),
            placeholder='select a dataset to embed',
        ),
        html.Br(),
        dcc.Dropdown(
            id='embed-model-selection',
            options=ModelLoader.get_existing_names(),
            placeholder='select a model to embed',
        ),
        html.Br(),
        html.Button("Embed Dataset", id="embed-ds-button"),
        dcc.Loading(
            children=[
                html.Br(),
                html.Div(id='embed-data-output', children=html.Div([])),
            ],
            type='default', #'circle',
            fullscreen=False,
        ),
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
    ]),
])])


# CALLBACKS
@APP.callback(
    Output("download-dataset", "data"),
    Input("download-ds-button", "n_clicks"),
    State("download-dataset-selection", 'value'),
    State("download-model-selection", 'value'),
    prevent_initial_call=True,
)
def download_data_after_click(n_clicks, data_name, model_name):
    if data_name is None:
        print(f'oops, please select a dataset!')
        return
    # check the most processed to least processed before zipping/sending
    name = f'{data_name}/{model_name}'
    if EmbeddedTextLoader.name_exists(name):
        print(f'found embedded text for {name=}, zipping before sending')
        path = EmbeddedTextLoader.get_output_path(name)
    elif TextLoader.name_exists(data_name):
        print(f'found (unembedded) text for {data_name=}, zipping before sending')
        path = TextLoader.get_output_path(data_name)
    elif TempLoader.name_exists(data_name):
        print(f'found (unembedded, uncached?) text for {data_name=}, zipping before sending')
        path = TextLoader.get_output_path(data_name)
    path = zip_dir(path)
    print(f'downloading zip of {path=}')
    return dcc.send_file(str(path))


@APP.callback(
    Output("embed-data-output", "children"),
    Input("embed-ds-button", "n_clicks"),
    State("embed-dataset-selection", 'value'),
    State("embed-model-selection", 'value'),
    prevent_initial_call=True,
)
def embed_data_after_click(n_clicks, data_name, model_name):
    if data_name is None:
        print(f'oops, please select a dataset!')
        return
    if model_name is None:
        print(f'oops, please select a model!')
        return
    # check the most processed to least processed before zipping/sending
    name = f'{data_name}/{model_name}'
    print(f'{name=} before embedding')
    # TODO: think about if we want an overwrite flag or assume clicking means rerun!
    if not EmbeddedTextLoader.name_exists(name):
        emb_path = EmbeddedTextLoader.get_output_path(name)
        print(f'{emb_path=}')
        # will be most computational part when using LLMs!
        text_ds = TextLoader.load(data_name)
        print(f'{text_ds=}')
        emb_ds = Embedder(model_name).embed(text_ds, concat=True)
        print(f'{emb_ds=}')
        emb_ds.save_to_disk(str(emb_path))
    else:
        print(f'already exists, skipping rerunning of embedding!')
    return html.Div(f'Finished embedding {name=}')

@APP.callback(
    Output('embed-dataset-selection', 'options'),
    Output('download-dataset-selection', 'options'),
    Output('embed-model-selection', 'options'),
    Output('download-model-selection', 'options'),
    Input('upload-data-output', 'children'),
)
def refresh_available_names(children):
    """call this after a new dataset is uploaded/cached, to update the datasets available for processing"""
    ds = TextLoader.get_existing_names()
    ms = ModelLoader.get_existing_names()
    return [ds, ds, ms, ms]


@APP.callback(
    Output('upload-data-output', 'children'),
    Input('upload-data-input', 'contents'),
    State('upload-data-input', 'filename'),
    State('upload-data-input', 'last_modified'),
)
def upload_and_cache_new_dataset(contents, filename, last_modified):
    """call this when new contents uploaded; it will cache the data and notify user the name of new dataset"""
    print(f'inputs: {contents=}, {filename=}, {last_modified=}')
    if contents is None:
        return
    # else: continue on
    # - determine name for dataset (include suffix for TempLoader?)
    name = Path(filename).name
    # - parse and format buffer of contents
    contents = base64.b64decode(contents.split(',')[1])
    # - assume file is new, or will overwrite previous one of same name (in temp and text cache stores)
    df = TempLoader.load(name, overwrite=True, buffer=contents)
    print(f'loaded temp {df=}')
    ds = TextLoader.load(name, overwrite=True, paths=TempLoader.get_output_path(name))
    print(f'cached/loaded text {ds=}')
    # - tell user it's done, pointing to name
    # transform df to dataset and store?
    # return something indicating done uploading and what the name is?
    children = html.Div([
        'Sucessfully uploaded dataset as: ',
        html.B(f'{TextLoader.get_clean_name(name)}'),
    ])
    return children
    

@APP.callback(
    Output(component_id='embedding-scatter', component_property='figure'),
    Input(component_id='df-col-dropdown', component_property='value'),
)
def update_embedding_scatter(col):
    fig = get_new_scatter(col)
    return fig


@APP.callback(
    Output(component_id='embedding-hist', component_property='figure'),
    Input(component_id='df-col-dropdown', component_property='value'),
    Input(component_id='embedding-scatter', component_property='hoverData'),
)
def update_embedding_hist(col, hover_data):
    maxx = hover_data['points'][0]['x'] if hover_data is not None else np.inf
    maxy = hover_data['points'][0]['y'] if hover_data is not None else np.inf
    fig = get_new_hist(maxx, maxy, col)
    return fig


if __name__ == "__main__":
    APP.run_server(debug=True)