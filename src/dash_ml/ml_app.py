# -- imports --
import base64
import datetime
import io
import pandas as pd
import pathlib

from dash import (
    Dash,
    html,
    dcc,
    dash_table,
)
from dash.dependencies import (
    Output,
    Input,
    State,
)

# -- globals --
MODES = ['Train', 'Infer']
TASKS_TO_MODEL_TYPES = {
    'Binary Classification': {
        'RF': 'Random Forest (RF)',
        'BERT_bin': 'BERT',
    },
    'Clustering': {
        'NMF': 'Non-negative Matrix Factorization (NMF)',
        'LDA': 'Latent Dirichlet Allocation (LDA)',
    },
    'Masked Language Modeling': {
        'BERT_mlm': 'BERT',
    },
}
MODELS_TO_INSTANCES = {
    model: [
        # TO FILL IN DYNAMICALLY?
    ] for task, dct in TASKS_TO_MODEL_TYPES.items() for model in dct
}
# TASKS_TO_MODEL_INSTANCES = {
#     task: {
#         model: [
#             # TO FILL IN DYNAMICALLY?
#         ] for model in dct 
#     } for task, dct in TASKS_TO_MODEL_TYPES.items()
# }
MODELS_TO_PARAMS = {
    # binary classifiers
    'RF': {
        'n_estimators': 100,
    },
    'BERT_bin': {
        
    },
    # clustering algs
    'NMF': {
        
    },
    'LDA': {
        
    },
    # MLM
    'BERT_mlm': {
        
    },
}
assert set(MODELS_TO_INSTANCES) == set(MODELS_TO_PARAMS)

# -- classes --

# -- functions --

# -- app def --
app = Dash(__name__)
app.layout = html.Div([
    # (top) title
    html.H1('Welcome to your Dash ML app!'),
    html.Br(),

    # (left) selections, (right) show what was input
    html.Div([
        # (left) selections
        html.Div([
            # - MODE
            html.Label('Choose MODE:'),
            dcc.Dropdown(MODES, id='dropdown-modes'),
            html.Br(),
            
            # - TASK
            html.Label('Choose TASK:'),
            dcc.Dropdown(sorted(TASKS_TO_MODEL_TYPES), id='dropdown-tasks'),
            html.Br(),
            
            # - UPLOAD DATA
            html.Label('Upload DATA:'),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                },
                # Allow multiple files to be uploaded
                multiple=True),
            html.Br(),
            
            # - MODEL
            html.Label('Choose MODEL:'),
            dcc.Dropdown(id='dropdown-models'),
            html.Br(),

            # - PARAMS
            html.Label('Choose PARAMS:'),
            html.Div(id='inputs-params'),
            html.Br(),
            
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        # (right) show what was input
        html.Div([
            # - output of read-in data
            html.Label('Preview DATA:'),
            html.Div(id='table-data'),
            html.Br(),
            
            # - output of training?
            # TODO:?
        ], style={'width': '48%', 'display': 'inline-block'}),
    ]),
    html.Br(),
            
    # (bottom) save/execute button
    html.Button('Click to being training/inferencing', id='button-save', n_clicks=0),
    html.Div(id='button-response-text'),
    html.Br(),
    
    # ?
])

def parse_upload_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    filepath = pathlib.Path(filename)
    decoded = base64.b64decode(content_string)
    try:
        if filepath.suffix == '.csv':
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filepath.suffix == '.xls' or filepath.suffix == '.xlsx':
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            # TODO: raise error, or silently fail here??
            df = pd.DataFrame()
    except Exception as e:
        print(e)
        return html.Div('There was an error processing this file.')
    else:
        return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            dash_table.DataTable(
                df.head(5).to_dict('records'),  # only print top 5 cols
                [{'name': i, 'id': i} for i in df.columns],
            ),

            html.Hr(),  # horizontal line

            # For debugging, display the raw contents provided by the web browser
            html.Div('Raw Content'),
            html.Pre(contents[0:200] + '...', style={
                'whiteSpace': 'pre-wrap',
                'wordBreak': 'break-all',
            }),
        ])

@app.callback(
    Output('table-data', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
)
def update_table(contents, names, dates):
    if contents is not None:
        return [parse_upload_contents(c, n, d) for c, n, d in zip(contents, names, dates)]

@app.callback(
    Output('dropdown-models', 'disabled'),
    Output('dropdown-models', 'options'),
    Input('dropdown-tasks', 'value'),
)
def update_model_dropdown(task):
    if task is None:
        return True, []
    else:
        assert task in TASKS_TO_MODEL_TYPES, f'{task=} should be in {TASKS_TO_MODEL_TYPES=}'
        return False, TASKS_TO_MODEL_TYPES[task]
    
# TODO: separate into two, so we can track each separate div produced here
# like when we need the param values to train, or the id to feed it the input data...
@app.callback(
    Output('inputs-params', 'children'),
    Input('dropdown-models', 'value'),
    Input('dropdown-modes', 'value'),
)
def update_param_inputs(model, mode):
    if model is None or mode is None:
        return True, []
    else:
        assert mode in MODES, f'{mode=} should be in {MODES=}'
        if mode == 'Train': # return param inputs with defaults for new model
            assert model in MODELS_TO_PARAMS, f'{model=} should be in {MODELS_TO_PARAMS=}'
            return [
                html.Div([
                    html.Label(f'{p}: '),
                    dcc.Input(v, 'number'),
                ]) for p, v in MODELS_TO_PARAMS[model].items()
            ]
        elif mode == 'Infer': # return dropdown of existing model ids
            assert model in MODELS_TO_INSTANCES, f'{model=} should be in {MODELS_TO_INSTANCES=}'
            return dcc.Dropdown(MODELS_TO_INSTANCES[model])

@app.callback(
    Output('button-response-text', 'children'),
    Input('button-save', 'n_clicks'),
)
def update_button_text(n_clicks):
    if n_clicks > 0:
        return 'Click: time to get training!'