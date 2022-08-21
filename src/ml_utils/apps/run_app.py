import click
import dash

# As more apps are implemented, update this
APP_TYPES = [
    'ml',
]

# each app_type points to modules which should implement a global Dash app formed and ready to run
def get_app(app_type):
    assert app_type in APP_TYPES, f'{app_type=} is not (yet?) supported'
    if app_type == 'ml':
        from .ml_app import app
    # elif app_type == '??':
    #     from ??_app import app
    else:
        raise Exception(f'{app_type=} not supported')
    assert type(app) == dash.Dash
    return app

@click.command()
@click.option('-a', '--app-type', default=APP_TYPES[0], type=click.Choice(APP_TYPES), help='which app type to run')
@click.option('-p', '--port', default=8050, help='port to host dash app')
@click.option('-d', '--debug', default=True, help='whether to run server in debug mode')
def run_app(app_type, port, debug):
    app = get_app(app_type)
    app.run_server(debug=debug, port=port)