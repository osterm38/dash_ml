from dash import Dash, html, dcc
from dash.dependencies import Output, Input, State

app = Dash(__name__)
app.layout = html.Div(
    'hi there!',
)
# """
# Workflow needed:
# I want to TRAIN a new model to CLASSIFY on my DATA with LABELS, using ALG
# 1. choose to train or infer
# 2. choose which task you want (e.g. classification, Q&A, next token, etc.)
#   - inform user that this is supervised (needs labels) or unsupervised (no label needed)
# 3. display button to import input data file/url
#   - check whether you have correctly formatted/labeled data
#   - might include any subset of training, val, testing data?
# 4. choose which particular model to use (e.g. RF, BERT, etc)
#   - simply display saved model ids to select (no input params needed?)
#   - display necessary/optional parameters to enter
#   - prepopulate with defaults
#   - allow uploading a config file?
# 5. display a button to save as configuration/start training or infering
#   - save config, input file, model, logs somewhere not to be overwritten
# 6. display a busy spinner 
#   - possibly on another page, so we can reuse this page
#   - show progress for some/all models?
# 7. display results
#   - depends on 2 and 3 mostly?
# 8. display button to export results file
#   - and figures, or the dashboard itself (maybe code to generate it as stand-alone app)???

# So, how do i:
# - have one button selection change subsequent selection options: https://dash.plotly.com/basic-callbacks#dash-app-with-chained-callbacks
# - import data: upload: https://dash.plotly.com/dash-core-components/upload
# - export data: ?
# - kick of long-running processes and display busy icon: loading: https://dash.plotly.com/dash-core-components/loading
# - manage users: ??
# - spawn multi-pages/new port subapps? tab or tabs: https://dash.plotly.com/dash-core-components/tabs
# - display tables: https://dash.plotly.com/dash-html-components/table
# - display text snippets with highlights?: xml from label-studio as example???
# """

# # Run this app with `python app.py` and
# # visit http://127.0.0.1:8050/ in your web browser.

# import datetime
# from dash import Dash, html, dcc
# from dash.dependencies import Input, Output, State
# import plotly.express as px
# import pandas as pd

# app = Dash(__name__)

# app.layout = html.Div([
#     html.H1('Welcome to my Dash ML app!'),
#     # 1. Train or infer
#     html.Div([
#         html.H2("Choose to train a new model, or infer with an existing one:"),
#         dcc.Dropdown(['TRAIN', 'INFER'], 'TRAIN'),
#     ]),
#     # 2. Task
#     html.Div([
#         html.H2("Choose which task to perform:"),
#         dcc.Dropdown(['Classification', 'Clustering'], 'Classification'),
#     ]),
#     # 3. Input data
#     html.Div([
#         html.H2("Import your data:"),
#         html.Div([
#             dcc.Upload(
#                 id='upload-image',
#                 children=html.Div([
#                     'Drag and Drop or ',
#                     html.A('Select Files')
#                 ]),
#                 style={
#                     'width': '100%',
#                     'height': '60px',
#                     'lineHeight': '60px',
#                     'borderWidth': '1px',
#                     'borderStyle': 'dashed',
#                     'borderRadius': '5px',
#                     'textAlign': 'center',
#                     'margin': '10px'
#                 },
#                 # Allow multiple files to be uploaded
#                 multiple=True
#             ),
#             html.Div(id='output-image-upload'),
#         ]),
#     ]),
# ])

# def parse_contents(contents, filename, date):
#     return html.Div([
#         html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),

#         # HTML images accept base64 encoded strings in the same format
#         # that is supplied by the upload
#         html.Img(src=contents),
#         html.Hr(),
#         html.Div('Raw Content'),
#         html.Pre(contents[0:200] + '...', style={
#             'whiteSpace': 'pre-wrap',
#             'wordBreak': 'break-all'
#         })
#     ])

# @app.callback(Output('output-image-upload', 'children'),
#               Input('upload-image', 'contents'),
#               State('upload-image', 'filename'),
#               State('upload-image', 'last_modified'))
# def update_output(list_of_contents, list_of_names, list_of_dates):
#     if list_of_contents is not None:
#         children = [
#             parse_contents(c, n, d) for c, n, d in
#             zip(list_of_contents, list_of_names, list_of_dates)]
#         return children


# if __name__ == '__main__':
#     app.run_server(debug=True)
