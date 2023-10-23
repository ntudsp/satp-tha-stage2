
import os
import glob
import pandas as pd
import numpy as np

from argparse import Namespace

PAQ_ORDER = ['pl', 'ch', 'vi', 'un', 'ca', 'an', 'ev', 'mo']
PAQ = PAQ_ORDER
PAQ_CCW = ['pl', 'vi', 'ev', 'ch', 'an', 'mo', 'un', 'ca']
PAQ_FULL = {
    'pl': 'Pleasant',
    'ev': 'Eventful',
    'an': 'Annoying',
    'un': 'Uneventful',
    'ch': 'Chaotic',
    'ca': 'Calm',
    'vi': 'Vibrant',
    'mo': 'Monotonous'
}


cos45 = np.cos(45.0 * np.pi/180.0)

def standard_basis(n, hot):
    e = np.zeros((n,))
    e[hot] = 1.0
    return e

def normalize_response(x):
    return (x-50.0)/50.0

def likertize_response(x):
    return (4*x/100) + 1

def iso_pl(x):
    raw =  (x.pl - x.an) + cos45 * (x.ca - x.ch) + cos45 * (x.vi - x.mo)
    return raw # / (4.0 + np.sqrt(32))

def iso_ev(x):
    raw = (x.ev - x.un) + cos45 * (x.ch - x.ca) + cos45 * (x.vi - x.mo)
    return raw # / (4.0 + np.sqrt(32))

def likert_norm(x):
    return x/(4.0 + np.sqrt(32))

def norm_norm(x):
    return x/(2 + np.sqrt(8))

def proj_norm(x):
    return x/2

iso_pl_vec = np.array([1, -cos45, cos45, 0, cos45, -1, 0, -cos45])
iso_ev_vec = np.array([0, cos45, cos45, -1, -cos45, 0, 1, -cos45])

iso_proj = np.array([
    [1, 0],
    [cos45, cos45],
    [0, 1],
    [-cos45, cos45],
    [-1, 0],
    [-cos45, -cos45],
    [0, -1],
    [cos45, -cos45]
])

def arr_to_ns(a, ccw=True):
    return Namespace(**{k: v for k, v in zip(PAQ_CCW if ccw else PAQ_ORDER, a.tolist())})

def iso_pl_np(a, ccw=True):
    return iso_pl(arr_to_ns(a, ccw=ccw))

def iso_ev_np(a, ccw=True):
    return iso_ev(arr_to_ns(a, ccw=ccw))

def iso_np(a, ccw=True):
    return np.stack([iso_pl_np(a, ccw=ccw), iso_ev_np(a, ccw=ccw)])


def double_square(x):
    return np.sign(x) * np.square(x)

def pid_from_filename(f):
    return int(f.replace("_Fail_Hearing", "").split("_")[-1].split(".")[0])

def load_data(data_root, calib_mode, drop_failed_hearing=True, normalize=True, compute_iso=True, exclude_pilot=True):
    assert calib_mode in ['hats', 'ocv']
    
    files = sorted(glob.glob(os.path.join(data_root, calib_mode, "*.csv")))
    
    
    
    if drop_failed_hearing:
        files = [f for f in files if "Fail_Hearing" not in f]
        
    dfs = []

    for f in files:

        pid = pid_from_filename(f)
        if exclude_pilot and pid == 1:
            continue

        dfi = pd.read_csv(f, header=None, names=['stimulus_id'] + PAQ_ORDER + ['chk', 't'])
        dfi['pid'] = pid

        dfs.append(dfi)
        
    df = pd.concat(dfs).sort_values(['stimulus_id', 'pid']).reset_index(drop=True)
    
    if normalize:
        for col in PAQ:
            df[col] = df[col].apply(normalize_response) #.apply(double_square)

            
        if compute_iso:
            df['isopl'] = df.apply(iso_pl, axis=1).apply(norm_norm)
            df['isoev'] = df.apply(iso_ev, axis=1).apply(norm_norm)
    else:
        if compute_iso:
            
            for col in PAQ:
                df[col + "_original"] = df[col].apply(lambda x: x)
                df[col] = df[col].apply(normalize_response) #.apply(double_square)
            df['isopl'] = df.apply(iso_pl, axis=1).apply(norm_norm)
            df['isoev'] = df.apply(iso_ev, axis=1).apply(norm_norm)
            
            df = df.drop(columns=PAQ)
            df = df.rename(columns={p + "_original": p for p in PAQ})
    
    return df

def aggregate_by_stimulus(df):
    return df.groupby('stimulus_id').mean()

def define_latex_command(name, value, value_format=None):
    
    if value_format is not None:
        if type(value_format) is str:
            value = value_format.format(value)
        else:
            value = value_format(value)
    
    nc = "\\newcommand{\\" + name + "}{" + value + "}"
    
    return nc

def elem_vec(n, hots):
    x = np.zeros((n,))
    x[hots] = 1
    
    return x

def pc_vec(hot):
    return elem_vec(8, [hot])

def l2normalize(x):
    return x/np.linalg.norm(x)

def hava(v, thresh=0.1, hd="right", vd="bottom"):
    if abs(v[0]) < thresh:
        ha = hd
    elif v[0] < 0:
        ha = "right"
    elif v[0] > 0:
        ha = "left"
        
    if abs(v[1]) < thresh:
        va = vd
    elif v[1] < 0:
        va = "top"
    elif v[1] > 0:
        va = "bottom"
   
    return {"ha": ha, "va": va}