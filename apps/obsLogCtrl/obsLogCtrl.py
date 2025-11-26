import sys
import logging
import xconf
from magaox.indi.device import XDevice, BaseConfig

import dash
from dash import Dash, html, dcc, Output, Input, callback, State
import dash_daq as daq
import dash_bootstrap_components as dbc

from pathlib import Path


log = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent
PAGES_DIR = BASE_DIR/"pages"
ASSETS_DIR = BASE_DIR/'assets'

app = Dash(__name__,
            use_pages=True,
            pages_folder=str(PAGES_DIR),
            assets_folder=str(ASSETS_DIR),
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = html.Div(
    id='dark-theme-container',
    children=[
        html.Div(
            daq.BooleanSwitch(
                on=True,
                id='darktheme-boolean',
                label=['Light', 'Dark'],
                color="#3DA5ff" # swtich color
            ),
            style={
                "position": "absolute",
                'top' : '20px',
                'right' : "20px",
                "zIndex": "1000"
            }
        ),
        html.Br(),
        #title
        html.H1('MagAO-X Observation log system',
                id='title-text', style={'textAlign':'center'}),
        html.Br(),
        #links
        html.Div([
            html.Div([
                dbc.Button(
                    page['name'], color="primary",
                    outline=True, class_name= "gap-2",
                    href=page['relative_path'],
                    external_link=True
                )
            ],style={'padding': '1px'}) for page in dash.page_registry.values()
        ]),
        dash.page_container,
    ],
    style={'padding' :'10px',
            'backgroundColor': '#303030',
            'color': "#eff0f1"} #default dark
)

#callback to change container style
@callback(
    Output('dark-theme-container', 'style'),
    Input('darktheme-boolean', 'on')
)
def switch_theme(dark):
    if dark:
        return {
            'margin': '0',
            'padding': '0',
            'backgroundColor': '#23262a',
            'color'  : '#eff0f1',
            'padding': '10px', #padding for how far in txt is
        }
    else:
        return {
            'padding': '10px',
            'backgroundColor': 'white',
            'color' : 'black'
        }


def run_server(debug=False):
    """
    Start the Dash observation log server
    """
    # You can tune host and port as needed
    log.info('starting Dash server for observation log')
    app.run(
        debug=debug,
        use_reloader=debug,
        port=8050,
    )

@xconf.config
class ExampleConfig(BaseConfig):
    """Example Python INDI device for MagAO-X

    Write your command-line help here, and it will be displayed when
    someone runs `pythonIndiExample -h` in the terminal (along with
    a summary of available options)
    """
    configurable_doodad_1 : str = xconf.field(default="abc", help="Configurable doodad 1")

class ObsLogCtrl(XDevice):
    config : ExampleConfig

    def loop(self):
        log.info("ObsLogCtrl.loop() launching Observation Log web UI")
        run_server(debug=False)

def app_entry(args=None):
    """Entry point function MagAO-X can call."""
    # If MagAO-X truly just wants an app() function that starts the script,
    # you can make this your official entry function.
    run_server(debug=False)

# if __name__ == "__main__":
#     run_server(debug=True)