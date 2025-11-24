import dash
from dash import Dash, html, dash_table, dcc, html, Input, Output, callback, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta, date
import pandas as pd
import psycopg
import json
from magaox import db
from observation_log import dash_start as dash_start
"""
for history lookups. this one won't cache nor will it auto refresh like the active log. (unless change date period of course.)
"""

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash.register_page(__name__, external_stylesheets=external_stylesheets)

time_frame = "2025-04-16 12:00:00"
time_format = "%Y-%m-%d %H:%M:%S"
date_time_object = datetime.strptime(time_frame, time_format)

today = datetime.now().date()
tomorrow = today + timedelta(days=1) - timedelta(seconds=0)

layout = html.Div([
    html.H1("Historic observations"),
    # for active observing
    html.Div([
        #pick dates to see
        dcc.DatePickerRange(
            id='my-date-picker-range',
            #persistant values
            persistence= True,
            persisted_props= ['start_date', 'end_date'],
            persistence_type='local', #or session for single tab session
            month_format= "YYYY-MM-DD",
            initial_visible_month=today,
            clearable=True,
        ),
    ]),
    html.Div([
        dbc.Button(
            'Search', id='search-button', n_clicks=0,
            color='primary', class_name="gap-2"
        ),
    ], style={'padding': 10}),


    dcc.Loading(
        id='loading-symbol',
        type='default',
        children=[
            html.Div(id='hist-status-msg'),
        ]
    ),
    html.Div([
        dash_table.DataTable(
            id='history-datatable',
            columns=[], #start empty
            data=[],
            sort_action="native",
            page_size=50,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center',
                        'padding':'5px'},
            style_data={
                "backgroundColor": "#1e1e1e",
                "color": "#eff0f1"
            },
            style_data_conditional=[
                {
                    'if' : {'row_index': 'odd'},
                    'backgroundColor': "#303030"
                }
            ],
            style_header={
                "backgroundColor": "#303030",
                "color": "#eff0f1"
            },
        export_format='xlsx',
        export_headers='display'
        ),
    ]),
])


@callback(
        Output('history-datatable', 'data'),
        Output('history-datatable', 'columns'),
        Output('hist-status-msg', 'children'),
        #Output('history-download', 'style'),
        Input('search-button', 'n_clicks'),
        State('my-date-picker-range', 'start_date'),
        State('my-date-picker-range', 'end_date'),
        prevent_initial_call=True, #want to pick range first
)
def load_date(_,start_date, end_date):
    if ctx.triggered_id == 'search-button':
        if start_date == None or end_date == None:
            return [],[], 'Pick a start and end date', {'display': 'none'}
        start_dt = datetime.fromisoformat(start_date).replace(hour=12, minute=0, second=0)
        end_dt = datetime.fromisoformat(end_date).replace(hour=12,minute=0, second=0)

        #run Query
        df = dash_start(start_dt, end_dt) 
        #session Store only for filtering instead of reloading full each time
        dcc.Store(id='raw-df', data=df.to_dict('records'), storage_type='session'),
        dcc.Store(id='filtered-df', storage_type='session'),
            
        #prep table
        columns =[{'name': i, 'id': i, 'deletable':True} for i in df.columns]
        return df.to_dict('records'), columns, f"Loaded {len(df)} rows from {start_dt} to {end_dt}"#, {'display': 'flex'}
