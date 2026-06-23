import pandas as pd
import numpy as np

def apply_log(df):
    df = df.copy()
    intensity_cols = ['det1_I', 'det2_I', 'det3_I']
    for col in intensity_cols:
        df[f'{col}_log'] = np.log(df[col])
    if drop:
        df = df.drop(columns=intensity_cols)
    return df

def add_weighted_centroid(df):
    df = df.copy()
    total_I = df['det1_I'] + df['det2_I'] + df['det3_I']
    df['centroid_x'] = (df['det1_x'] * df['det1_I'] + df['det2_x'] * df['det2_I'] + df['det3_x'] * df['det3_I']) / total_I
    df['centroid_y'] = (df['det1_y'] * df['det1_I'] + df['det2_y'] * df['det2_I'] + df['det3_y'] * df['det3_I']) / total_I
    return df

def add_sensor_distances(df):
    df = df.copy()
    df['dist_12'] = np.sqrt((df['det1_x'] - df['det2_x'])**2 + (df['det1_y'] - df['det2_y'])**2)
    df['dist_23'] = np.sqrt((df['det2_x'] - df['det3_x'])**2 + (df['det2_y'] - df['det3_y'])**2)
    df['dist_13'] = np.sqrt((df['det1_x'] - df['det3_x'])**2 + (df['det1_y'] - df['det3_y'])**2)
    return df

def add_intensity_ratios(df):
    df = df.copy()
    df['ratio_1_to_2'] = df['det1_I'] / (df['det2_I'] )
    df['ratio_2_to_1'] = df['det2_I'] / (df['det1_I'] )
    df['ratio_2_to_3'] = df['det2_I'] / (df['det3_I'] )
    df['ratio_3_to_2'] = df['det3_I'] / (df['det2_I'] )
    df['ratio_1_to_3'] = df['det1_I'] / (df['det3_I'] )
    df['ratio_3_to_1'] = df['det3_I'] / (df['det1_I'] )
    return df

def add_relative_coordinates(df):
    df = df.copy()
    conds_x = [
        df['det1_I'] >= df[['det2_I', 'det3_I']].max(axis=1),
        df['det2_I'] >= df[['det1_I', 'det3_I']].max(axis=1),
        df['det3_I'] >= df[['det1_I', 'det2_I']].max(axis=1)
    ]
    choices_x = [df['det1_x'], df['det2_x'], df['det3_x']]
    choices_y = [df['det1_y'], df['det2_y'], df['det3_y']]
    df['anchor_x'] = np.select(conds_x, choices_x)
    df['anchor_y'] = np.select(conds_x, choices_y)
    df['rel_det1_x'] = df['det1_x'] - df['anchor_x']
    df['rel_det1_y'] = df['det1_y'] - df['anchor_y']
    df['rel_det2_x'] = df['det2_x'] - df['anchor_x']
    df['rel_det2_y'] = df['det2_y'] - df['anchor_y']
    df['rel_det3_x'] = df['det3_x'] - df['anchor_x']
    df['rel_det3_y'] = df['det3_y'] - df['anchor_y']
    #df = df.drop(columns=['anchor_x', 'anchor_y'])
    return df

def sort_detectors_by_intensity(df):
    df_sorted = df.copy()

    intensities = df_sorted[['det1_I', 'det2_I', 'det3_I']].values
    x_coords = df_sorted[['det1_x', 'det2_x', 'det3_x']].values
    y_coords = df_sorted[['det1_y', 'det2_y', 'det3_y']].values

    sorted_idx = np.argsort(intensities, axis=1)[:, ::-1]

    sorted_I = np.take_along_axis(intensities, sorted_idx, axis=1)
    sorted_x = np.take_along_axis(x_coords, sorted_idx, axis=1)
    sorted_y = np.take_along_axis(y_coords, sorted_idx, axis=1)

    df_sorted['det1_I'], df_sorted['det2_I'], df_sorted['det3_I'] = sorted_I[:,0], sorted_I[:,1], sorted_I[:,2]
    df_sorted['det1_x'], df_sorted['det2_x'], df_sorted['det3_x'] = sorted_x[:,0], sorted_x[:,1], sorted_x[:,2]
    df_sorted['det1_y'], df_sorted['det2_y'], df_sorted['det3_y'] = sorted_y[:,0], sorted_y[:,1], sorted_y[:,2]

    return df_sorted