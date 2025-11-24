import dash
from dash import html, dcc
import plotly.graph_objects as go


"""
image comes from the assets page. this is just flaviour. CSS standards
"""

dash.register_page(__name__, path='/') #this is needed for home page since '/' will always be inxed 0


layout = html.Div([
    html.H1("Welcome to the Steward Observatory Observation Viewer",
            style={"textAlign": "center"}),
    html.H2([
        "Visualize and explore real-time or historic observation logs",
        html.Br(),
        "with clarity and precision"
    ],style={"textAlign": "center"}),
    html.Div(
        html.Img(
            src="/assets/magaox_pic2.png",
            style={'display' : "block",
                    "margin-left": 'auto',
                    "margin-right" : 'auto'
            }
        ),
        style={
            "height": "80hv",
            "overflow": "hidden",
        }
    )
],style={
    "height": "100vh", "display": "flex", "flexDirection": "column"}
)

