import dash
from dash import Dash, html, dash_table, dcc, html, Input, Output, callback, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from datetime import datetime, timedelta, date, timezone
import time
import pandas as pd
import psycopg
import json
from magaox import db
from observation_log import dash_start
"""
this page will only be for the active observing
"""
DEBUG = True

COLUMNS_IDS = ['observer', 'obsname', 'target', 'comments_changes', 'ts_utc',
        'holoop_state', 'fwsci1', 'camsci1_exptime', 'camsci1_emgain',
        'camsci1_read_out_speed', 'camsci1_shutter_state', 'camsci1_roi',
        'fwsci2', 'camsci2_exptime', 'camsci2_emgain', 'camsci2_read_out_speed',
        'camsci2_shutter_state', 'camsci2_roi', 'camwfs_exptime', 'camwfs_gain',
        'flipacq', 'stagebs', 'fwpupil', 'fwfpm', 'fwlyot', 'stagescibs',
        'flipwfsf'
    ]
COLUMNS = [{'name':col, 'id': col} for col in COLUMNS_IDS]

MIN_INTERVAL = 1_000   # 1 second
MAX_INTERVAL = 3_000   # 3 seconds

#global cache
MAX_CACHE_HOURS=12

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
            columns=COLUMNS, 
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
            interval=2*1000,
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
        Output('active-raw-data', 'data'),
        Output('filtered-active-df', 'data'),
        Output('status-msg', 'children'),
        Output('interval-component', 'interval'),
        State('my-date-picker-range', 'start_date'),
        State('my-date-picker-range', 'end_date'),
        Input('interval-component', 'n_intervals'),
        prevent_initial_call=False
)
def load_data(start_date, end_date, *_):
    start_cb = time.perf_counter()
    #start at 12:00 today. can adust if started after midnight
    start_dt = datetime.fromisoformat(start_date).replace(hour=12, minute=0,second=0)
    end_dt = datetime.fromisoformat(end_date).replace(hour=12, minute=0,second=0)
    #db query
    df = refresh_observation_cache(start_dt, end_dt)
    #df.sort_values(by='ts_utc', ascending=False, inplace=True)


    records = df.to_dict('records')

    #cvompute interval time
    compute_ms = (time.perf_counter()-start_cb) *1000.0
    new_interval = choose_interval(compute_ms)
    if DEBUG:
        status=f"Loaded {len(df)-1} rows @ {new_interval}ms" 
    else:
        status=f"Loaded {len(df)-1}"


    return records, records, status, new_interval

@callback(
        Output('active-datatable', 'data'),
        Input('filtered-active-df', 'data'),
        prevent_initial_call = False
)
def update_table(filtered_records):
    if not filtered_records:
        return [], []
    
    df = pd.DataFrame(filtered_records)
    return filtered_records

def choose_interval(compute_ms:float)->int:
    """
    Docstring for choose_interval
    
    :param compute_ms: Description
    :type compute_ms: float
    :return: Description
    :rtype: int
    """
    # if compute_ms <500:
    #     interval = 1_000
    if compute_ms <1_500:
        interval = 1_500
    elif compute_ms < 2_000:
        interval = 2_000
    elif compute_ms <2_500:
        interval = 2_500
    elif compute_ms < 3_000:
        interval = 3_000
    else:
        interval = int(min(MAX_INTERVAL, compute_ms*1.5))
    
    interval = max(MIN_INTERVAL, min(interval, MAX_INTERVAL))
    return interval

def prune_cache():
    global _obs_cache
    if _obs_cache is None or _obs_cache.empty:
        return
    cutoff = _obs_cache['ts_utc'].max() - pd.Timedelta(hours=MAX_CACHE_HOURS)
    _obs_cache = _obs_cache[_obs_cache['ts_utc'] >= cutoff]

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

    prune_cache()
    return _obs_cache