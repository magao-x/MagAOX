import dash
from dash import Dash, html, dash_table, dcc, html, Input, Output, callback, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta, date, timezone
import pandas as pd
import psycopg
import json
from magaox import db
from observation_log_listener import match_per_val
from observation_log import dash_start
"""
this page will only be for the active observing
"""

#global cache
_obs_cache = None #full combined + annotated df
_last_ts = None  #last timestamp seen 
_cache_initialized = False

dash.register_page(__name__)

time_frame = "2025-04-16 12:00:00"
time_format = "%Y-%m-%d %H:%M:%S"
date_time_object = datetime.strptime(time_frame, time_format)

now = datetime.now()
today = datetime.now().date()
tomorrow = today + timedelta(days=1) - timedelta(seconds=0)

#if time is before noon (1200) then the start date will be yesterday at noon
if now.hour < 12:
    start = today - timedelta(days=1)
else:
    start = today

layout = html.Div([
    html.H3("Press 'Start Log' to start live feed", id='start-end-text'),
    #make sure dates align to observation
    html.Div([
        dcc.DatePickerRange(
            id='my-date-picker-range',
            #persistant values
            persistence= True,
            persisted_props= ['start_date', 'end_date'],
            persistence_type='local', #or session for single tab session
            month_format= "YYYY-MM-DD",
            end_date_placeholder_text= "YYYY-MM-DD",
            initial_visible_month=today,
            start_date=start,
            end_date=tomorrow,
            disabled=True,
            ),
    ]),
    html.Div([
        dbc.Button('Start Log', id='start-button', active=False, n_clicks=0),
        dbc.Button('End Log', id='stop-button', disabled=True, n_clicks=0)
    ]),
    html.Br(),

    html.Div(id='status-msg'),
    html.Div([
        dcc.Store(id='active-raw-data', storage_type='session'),
        dcc.Store(id='filtered-active-df', storage_type='session'),
        dash_table.DataTable(
            id='active-datatable',
            virtualization=True,
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
            #TODO: add condition to change highlight color to the blue to match
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
        export_format = 'xlsx',
        export_headers = 'display',
        ),
        dcc.Interval(
            id='interval-component',
            interval= 3*1000,
            #interval = 500,
            n_intervals=-1,
            disabled=True # starts off
        )
    ]),
])

@callback(
        Output('start-button', 'active'),
        Output('stop-button', 'disabled'),
        Input('start-button', 'n_clicks'),
        Input('stop-button', 'n_clicks'),
        prevent_initial_call=True
)
def activate_deactive(*_):
    ctx_id = ctx.triggered_id
    if ctx_id =='start-button':
        return True, False
    elif ctx_id == 'stop-button':
        return False, True

@callback(
        Output('interval-component', 'disabled'),
        Output('start-end-text', 'children'),
        Input('start-button', 'n_clicks'),
        Input('stop-button', 'n_clicks'),
        prevent_initial_call=True
)
def toggle_interval(*_):
    ctx_id = ctx.triggered_id 
    if ctx_id == 'start-button':
        return False, "Press 'Stop Log' to stop live feed" #enable interval
    if ctx_id == 'stop-button':
        return True, "Press 'Start Log' to start live feed" # disable interval
    return dash.no_update

@callback(
        Output('active-datatable', 'data'),
        Output('active-datatable', 'columns'),
        Output('status-msg', 'children'),
        Output('active-raw-data', 'children'),
        Output('filtered-active-df', 'data'),
        State('my-date-picker-range', 'start_date'),
        State('my-date-picker-range', 'end_date'),
        Input('interval-component', 'n_intervals'),
        prevent_initial_call=False
)
def load_data(start_date, end_date, *_):
    #TODO:update the time to make quicker queries
    #start at 12:00 today. can adust if started after midnight
    start_dt = datetime.fromisoformat(start_date).replace(hour=12, minute=0,second=0)
    end_dt = datetime.fromisoformat(end_date).replace(hour=12, minute=0,second=0)
    #db query
    #df= dash_start(start_dt, end_dt) #function to query data
    df = refresh_observation_cache(start_dt, end_dt)
    df.sort_values(by='ts_utc', ascending=False, inplace=True)
    #cashe the session for filtering purposes
    # dcc.Store(id='active-raw-data', data=df.to_dict('records'), storage_type='session')
    # dcc.Store(id='filtered-active-df', storage_type='session')
    
    #prep table
    columns =[{'name': i, 'id': i} for i in df.columns]

    status=f"Loaded {len(df)} rows" # from {start_dt} to {end_dt} at {now.time()}"

    records = df.to_dict('records')
    return records, columns, status, records, records

def init_observation_cache(start_dt, end_dt):
    """
    This should only run once during intitial request
    """
    global _obs_cache, _last_ts, _cache_initialized
    end = datetime.now(timezone.utc)

    start = end - timedelta(hours=24)

    df = dash_start(start_dt, end_dt, init_df=None)
    if df.empty:
        _obs_cache=pd.DataFrame()
        _last_ts = end
        _cache_initialized=True
        return _obs_cache
    
    _obs_cache = df
    _last_ts = df['ts_utc'].max()
    _cache_initialized = True
    return _obs_cache

def refresh_observation_cache(start_dt, end_dt):
    """
    Get new rows since _last_ts and merge with cache.
    start_dt and end_dt are from the date picker but
    _last_ts drives the incremental window.
    """

    global _obs_cache, _last_ts, _cache_initialized

    if not _cache_initialized or _obs_cache is None:
        return init_observation_cache(start_dt, end_dt)
    
    start = _last_ts
    end = datetime.now(timezone.utc)

    cont_df = dash_start(start_date=start, end_date=end, init_df = _obs_cache)

    if cont_df.empty:
        return _obs_cache
    
    _obs_cache = cont_df
    _last_ts = cont_df['ts_utc'] .max()

    return _obs_cache