from magaox import db
import psycopg
from typing import Optional, Union
import pandas as pd
import numpy as np

import time
import os
import sys
from contextlib import contextmanager #python with statement
import logging
from datetime import datetime, timedelta

from threading import Thread


DEBUG = False
START_UP =False
LISTENING = True
FORMAT = '%(asctime)s %(clientip)-15s %(user)-8s %(message)s'

logger = logging.getLogger(__name__)
logging.basicConfig(filename='mainLog.log', level=logging.INFO, format=FORMAT)

os.environ["XTELEMDB_PASSWORD"] = 'extremeAO!'   #####temp needs to be fixed####
#ensures connection closes automaticaly even with an exception occuring. don't rely on user to close connection
@contextmanager
def get_cursor():
    conn = db.connect()
    try:
        yield conn.cursor() #passes to caller
    finally:
        conn.close() #ensures closure

def query_dataframe(query, date):
    start = date[0]
    end = date[1]

    if START_UP:print('connecting to database')
    #get cursor
    with get_cursor() as cursor:
        if START_UP:print('connected to database')
        try:
            #execute query
            if date is not None:
                cursor.execute(query, (start, end))
            else:
                cursor.execute(query)
            #get rows from cursor
            rows = cursor.fetchall()
            #get the columns
            columns = [desc[0] for desc in cursor.description]
            #create the df 
            return pd.DataFrame(rows, columns= columns)
            #print(f"df: df.head(5)")
        except Exception as e:
            logging.error(f'df query failed {e}')
            sys.exit #return empty to prevent crash

def find_active_observing(df, collapse_dupes=True):
    """
    Finds the groups during active ('observing' = 'true') periods.
    """
    find_start = time.perf_counter()
    if df.empty:
        return df
    
    df = df.copy()
    if DEBUG:print('finding observing windows')
    # Step 1: Normalize observing values and fill missing ones
    df['observing'] = df['observing'].str.strip().str.lower()  # Normalize case and whitespace
    
    obs = df['observing']
    df['observing'] = np.where(obs.isna(), pd.NA, obs.astype(str).str.strip().str.lower())

    #tie break
    df['_seq'] = np.arange(len(df))
    df = df.sort_values(['ts_utc', '_seq']).reset_index(drop=True)

    #df['observing'] = df['observing'].ffill()  # Fill missing states

    df['observer'] = df['observer'].ffill() 

    # fill observing only 
    df['observing']= df['observing'].ffill().bfill()

    prev_obs = df['observing'].shift(1).fillna('false')
    next_obs = df['observing'].shift(-1).fillna('false')

    df['start_of_window']= ((df['observing'] == 'true') & (prev_obs != 'true')).astype(int)

    #window_id increments only on true start
    df['window_id']= df['start_of_window'].cumsum()
    #mask out non-observing rows so they don't shar the window_id
    #df['window_id'] = df['window_id'].where(df['observing']=='true', 0)

    #step 3:new comments_changes column
    if 'comments_changes' not in df.columns:
        df.insert(loc=3, column='comments_changes', value=pd.NA)
    #TODO Somewhere here it is placing end observation even though it is still true
    start_mask = (df['observing']=='true') & (prev_obs != 'true')
    end_mask = (df['observing']=='false') & (prev_obs == 'true')

    df.loc[start_mask, 'comments_changes']=(
        df.loc[start_mask, 'comments_changes'].replace('', pd.NA).fillna('Start Observation')
    )
    df.loc[end_mask, 'comments_changes']=(
        df.loc[end_mask, 'comments_changes'].replace('', pd.NA).fillna('End Observation')
    )

    #keep only observing ==True rows for downstream
    filtered = df.loc[((df['observing']=='false') & (df['observing'].shift() == 'true')) | (df['observing']=='true')].copy()

    #collapse duplicate timestamps i think this is the big thing messing it up
    if collapse_dupes:
        filtered = (filtered.sort_values(['window_id', 'ts_utc', '_seq'])
                    .drop_duplicates(['window_id', 'ts_utc'], keep='last'))

    # --- 7) fill ONLY safe columns, within each window (no bleed across windows) ---
    exclude = {'ts_utc', 'window_id', 'observing', 'comments_changes', '_seq'}
    cols_to_fill = [c for c in filtered.columns if c not in exclude]
    # groupwise fill prevents crossing window boundaries
    filled = (filtered.sort_values(['window_id', 'ts_utc', '_seq'])
                .groupby('window_id', group_keys=False)[cols_to_fill]
                .apply(lambda g: g.ffill().bfill()))
    filtered.update(filled)

    # --- 8) tidy up ---
    filtered = filtered.sort_values(['window_id', 'ts_utc', '_seq']).reset_index(drop=True)
    filtered.drop(columns=['start_of_window', '_seq'], errors='ignore', inplace=True)
    if DEBUG: print(f"find_active_time: {time.perf_counter()-find_start}")
    return filtered 

def clean_window(df):
    clean_time = time.perf_counter()
    if df.empty:
        return df
    df = df[df['comments_changes'].str.len() > 0]
    #df = df.loc[(df['comments_changes']==pd.notna)|(df['comments_changes']!='')]
    #df = df.drop(columns=['observing', 'window_id', 'start_of_window'], errors='ignore')
    if DEBUG: print(f"cleaning: {time.perf_counter()-clean_time}")
    return df

def get_telem(start_date, end_date):   
    """
    telemetry should be retrieved from table 'telem_partition'
    telem is partitioned based on months (previous, current, next)
    """
    telem_start = time.perf_counter()
    table_name = f"telem_{start_date.strftime('%Y_%m')}"
    if START_UP:print('getting telem')
            #CASE WHEN device = 'tcsi' AND ec = 'telem_telsee' THEN msg ->> 'dimm_fwhm_corr' END AS dimm_fwhm_corr,
    query = f"""
        SELECT
            msg ->> 'email' AS observer,
            msg ->> 'obsName' AS obsname,
            msg ->> 'tgt_name' AS target,
            msg ->> 'observing' AS observing,
            ts AS ts_utc,
            CASE WHEN device = 'holoop' THEN msg ->> 'state' END AS holoop_state,
            CASE WHEN device = 'fwsci1' THEN msg ->> 'presetName' END AS fwsci1,
            CASE WHEN device = 'camsci1' THEN msg ->> 'exptime' END AS camsci1_exptime,
            CASE WHEN device = 'camsci1' THEN msg ->> 'emGain' END AS camsci1_emgain,
            CASE WHEN device = 'camsci1' THEN msg ->> 'adcSpeed' END AS camsci1_read_out_speed,
            CASE WHEN device = 'camsci1' THEN (msg -> 'shutter' ->> 'state') END AS camsci1_shutter_state,
            CASE WHEN device = 'camsci1' THEN (msg -> 'roi' ->> 'h') END || 'x' ||
                CASE WHEN device = 'camsci1' THEN (msg -> 'roi' ->> 'w') END AS camsci1_roi,
            CASE WHEN device = 'fwsci2' THEN msg ->> 'presetName' END AS fwsci2,
            CASE WHEN device = 'camsci2' AND ec = 'telem_stdcam' THEN msg ->> 'exptime' END AS camsci2_exptime,
            CASE WHEN device = 'camsci2' AND ec = 'telem_stdcam' THEN msg ->> 'emGain' END AS camsci2_emgain,
            CASE WHEN device = 'camsci2' AND ec = 'telem_stdcam' THEN msg ->> 'adcSpeed' END AS camsci2_read_out_speed,
            CASE WHEN device = 'camsci2' AND ec = 'telem_stdcam' THEN (msg -> 'shutter' ->> 'state') END AS camsci2_shutter_state,
            CASE WHEN device = 'camsci2' AND ec = 'telem_stdcam' THEN (msg -> 'roi' ->> 'h') END || 'x' ||
                CASE WHEN device = 'camsci2' AND ec = 'telem_stdcam' THEN (msg -> 'roi' ->> 'w') END AS camsci2_roi,
            CASE WHEN device = 'camwfs' AND ec = 'telem_stdcam' THEN msg ->> 'exptime' END AS camwfs_exptime,
            CASE WHEN device = 'camwfs' AND ec = 'telem_stdcam' THEN msg ->> 'emGain' END AS camwfs_gain,
            CASE WHEN device = 'flipacq' THEN msg ->> 'presetName' END AS flipacq,
            CASE WHEN device = 'stagebs' THEN msg ->> 'presetName' END AS stagebs,
            CASE WHEN device = 'fwpupil' THEN msg ->> 'presetName' END AS fwpupil,
            CASE WHEN device = 'fwfpm' THEN msg ->> 'presetName' END AS fwfpm,
            CASE WHEN device = 'fwlyot' THEN msg ->> 'presetName' END AS fwlyot,
            CASE WHEN device = 'stagescibs' THEN msg ->> 'presetName' END AS stagescibs,
            CASE WHEN device = 'flipwfsf' THEN msg ->> 'presetName' END AS flipwfsf
        FROM {table_name}
        WHERE ts > %s AND ts <= %s
        AND device IN ('holoop', 'fwsci1', 'camsci1', 'fwsci2', 'camsci2', 'camwfs',
                'flipacq', 'stagebs', 'fwpupil', 'fwfpm', 'fwlyot',
                'stagescibs', 'flipwfsf')
        ORDER BY ts;
    """
    df = query_dataframe(query, (start_date,end_date))
    #time to run query
    query_time = (time.perf_counter() - telem_start)   
    #total run time
    if DEBUG: print(f'telem_time: {time.perf_counter()-telem_start}')
    logging.debug(f"total time: {(time.perf_counter()-telem_start)}")
    return df
#--'2025-02-10'

def get_user_logs(start_date, end_date):
    """
    user_logs should be retrieved form table 'user_log_partition'
    """
    table_name = f"user_log_{start_date.strftime('%Y_%m')}"

    if START_UP:print('getting user logs')
    query= f""" 
        SELECT 
            ts AS ts_utc,
            msg ->> 'message' AS comments_changes
        FROM 
            {table_name}
        WHERE 
            ec = 'user_log'
        AND
            ts > %s AND ts <= %s
        ORDER BY
            ts  
        ;
        """
    return query_dataframe(query,(start_date,end_date))

def get_logs(start_date=None, end_date=None):
    """
    Gets telemety from telemetry table
    also gets user_logs for the past 2 days.
    ts__utc switches to datetime for pandas to be able to sort. 
    """

    logs_start = time.perf_counter()
    if start_date is None and end_date is None:
        start_date = (datetime.now() - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)

        telem_df = get_telem(date= start_date)
        #active_df = find_active_observing(telem_df)

        #NEED TS for merger
        #future might look at not switching to string in dataframe
        #telem_df['ts_utc'] = pd.to_datetime(telem_df['ts_utc'], errors='coerce')
        #print(f'telem col check:\n{telem_df.columns}')

        #fetch user-log
        user_log_df = get_user_logs(start_date= start_date, end_date=end_date)
        #user_log_df['ts_utc'] = pd.to_datetime(user_log_df['ts_utc'], errors='coerce')

        if DEBUG:
            print(f"user log check:\n{user_log_df[['ts_utc', 'comments_changes']].head()}")
    
    else: #comming from ploty
        telem_df = get_telem(start_date= start_date, end_date=end_date)
        #telem_df = find_active_observing(telem_df)
        #NEED TS for merger
        #future might look at not switching to string in dataframe
    #    telem_df['ts_utc'] = pd.to_datetime(telem_df['ts_utc'],utc=True, errors='coerce', exact=False)
        #print(f'telem col check:\n{telem_df.columns}')

        #fetch user-log
        user_log_df = get_user_logs(start_date= start_date, end_date=end_date)
    #    user_log_df['ts_utc'] = pd.to_datetime(user_log_df['ts_utc'], utc=True, errors='coerce', exact=False)
        if DEBUG: print(f'logtime: {time.perf_counter()-logs_start}')
    return telem_df, user_log_df

def combine_dataframe(telem_df, user_log_df):
    #pandas will handle this with outer merge**
    
    try:
        #
        # merge telemetry and user logs
        combined_df = pd.merge(
            left=telem_df,
            right=user_log_df,
            how='outer',
            on=['comments_changes', 'ts_utc']
        )
            
        # fill in the nan to ''
        combined_df.fillna('')
            
        #sort the result by timestamp (importanct after an outer_merge)
        combined_df.sort_values('ts_utc', inplace=True)
        return combined_df
    except Exception as e:
        print('failed to combine dataframe')
        logging.error(f"Failed to combine dataframes: {e}")
        return pd.DataFrame()

def annotate_changes(df: pd.DataFrame) -> pd.DataFrame:
    a_start = time.perf_counter()

    if df.empty:
        return df

    # Sort for stable window and time order
    df = df.sort_values(['window_id', 'ts_utc']).copy()
    df['__orig_index__'] = df.index

    exclude_cols = {
        'ts_utc', 'comments_changes', '__orig_index__', 'observing',
        'obsname', 'observer', 'target', 'window_id', 'start_of_window'
    }
    compare_cols = [c for c in df.columns if c not in exclude_cols]

    # Make sure the column exists, but do NOT wipe existing content
    if 'comments_changes' not in df.columns:
        df['comments_changes'] = ''

    if compare_cols:
        prev_vals = df.groupby('window_id')[compare_cols].shift(1)

        both_na = df[compare_cols].isna() & prev_vals.isna()
        value_diff = df[compare_cols].ne(prev_vals)
        changed = (~both_na) & value_diff

        # First row in each window should never be treated as "parameter changed"
        first_in_window = df.groupby('window_id').cumcount() == 0
        changed.loc[first_in_window, :] = False

        def build_comment(idx):
            # existing comments, such as start_observation or end_observation
            existing = df.at[idx, 'comments_changes']
            if pd.isna(existing):
                existing = ''

            row_changed = changed.loc[idx]
            cols_changed = row_changed.index[row_changed.values]

            if len(cols_changed) == 0:
                # No new parameter changes for this row, keep whatever was there
                return existing

            new_parts = [f"{col}: {df.at[idx, col]}" for col in cols_changed]
            new_comment = ", ".join(new_parts)

            # If there was already a comment (start_observation, end_observation),
            # append the changes instead of overwriting
            if existing:
                return f"{existing}, {new_comment}"
            else:
                return new_comment

        df['comments_changes'] = [build_comment(i) for i in df.index]

    non_ts_cols = [c for c in df.columns if c != 'ts_utc']
    df.dropna(subset=non_ts_cols, how='all', inplace=True)
    df.fillna('', inplace=True)

    df = df.sort_values('__orig_index__').drop(columns='__orig_index__')

    if DEBUG: print(f'annotate time: {time.perf_counter() - a_start}')
    return df

def annotate_changes1(df):
    a_start = time.perf_counter()
    if df.empty:
        return df

    #work per group without the mutation until end
    for window_id, group in df.groupby('window_id'):
        group = group.sort_values('ts_utc').copy()
        group['__orig_index__'] = group.index #stable key now

        exlude_cols = {'ts_utc', 'comments_changes', '__orig_index__', 'observing',
                        'obsname', 'observer', 'target', 'window_id','start_of_window'}
        
        compare_cols = [c for c in group.columns if c not in exlude_cols]

        if len(group) >1:
            for idx in range(1, len(group)): #skip first row
                prev_row = group.iloc[idx- 1]
                row = group.iloc[idx]

                for col in compare_cols:
                    changes = []
                    v1, v0, = row[col], prev_row[col]
                    if not (pd.isna(v1) and pd.isna(v0)): #both == na  = same
                        if pd.isna(v1) != pd.isna(v0) or  v1 != v0:
                            changes.append(f"{col}: {v1}")

                    if len(changes) >0:
                        group.at[group.index[idx], 'comments_changes']= ", ".join(changes)
        #update main df
        df.loc[group['__orig_index__'], 'comments_changes'] = group['comments_changes']
    
    #clean up 
    #df.drop(columns=['observing', 'start_of_window', 'window_id'], inplace=True)
    non_ts_cols = [c for c in df.columns if c != 'ts_utc']
    df.dropna(subset=non_ts_cols, how='all', inplace=True)
    df.fillna('', inplace=True)
    if DEBUG:  print(f'annotate time: {time.perf_counter() - a_start}')
    return df

def dash_start(start_date, end_date, init_df):
    LISTENING = False
    #non filtered logs
    telem_df, user_log_df = get_logs(start_date=start_date, end_date=end_date)
    #time isn't populating, just date for telem

    active_df = find_active_observing(telem_df)
    filtered_df = annotate_changes(active_df)
    filtered_df = clean_window(filtered_df)

    #combine the non-filtered logs

    #combined_df = track_changes(filtered_df)
    if not user_log_df.empty:
        new_combined_df = combine_dataframe(telem_df=filtered_df, user_log_df=user_log_df)
    else:
        new_combined_df = filtered_df.copy()
    if init_df is not None and not init_df.empty:
        combined_df = pd.concat([init_df, new_combined_df], ignore_index=True)
    else:
        combined_df = new_combined_df
    # in off chance vals got out of wack yo
    combined_df = combined_df.sort_values(by=['ts_utc'], ascending=False)

    combined_df.dropna(subset=[col for col in combined_df.columns if col != 'ts_utc'], how='all', inplace=True)
    combined_df.drop(columns=['window_id', 'observing'], inplace=True)
    return combined_df

    
if __name__ == "__main__":
    # start_dt = '2025-10-16 12:00:00'
    # end_dt = '2025-10-17 12:00:00'
    start_dt = '2025-11-18 22:00:00.00'
    end_dt = '2025-11-19 12:00:00.00'
    dt_format = "%Y-%m-%d %H:%M:%S.%f"
    dt_start_object = datetime.strptime(start_dt, dt_format)
    dt_end_object = datetime.strptime(end_dt, dt_format)
    t_start = time.time()
    dash_start(dt_start_object, dt_end_object, init_df=None)
    t_end = time.time() - t_start
    print(t_end)
    START_UP=True
