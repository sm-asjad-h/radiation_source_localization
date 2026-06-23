import numpy as np
import pandas as pd
def generate_dataset_area_only(num_samples=30000, area_size=18, del_cons=1.2, back_rad=0.0,SI_min=400.0, SI_max=8000.0, dist_min=0.5,min_area=10.0):
    data = []
    for sample_idx in range(num_samples):
        src_x = np.random.uniform(0, area_size)
        src_y = np.random.uniform(0, area_size)
        src_i0 = np.random.uniform(SI_min, SI_max)

        geom_valid = False
        while not geom_valid:
            row = {'source_x': src_x, 'source_y': src_y, 'I_0': src_i0}

            for i in range(1, 4):
                valid = False
                while not valid:
                    det_x = np.random.uniform(0, area_size)
                    det_y = np.random.uniform(0, area_size)
                    dist_src = np.sqrt((src_x - det_x)**2 + (src_y - det_y)**2)
                    close = dist_src < dist_min

                    for prev_i in range(1, i):
                        d_prev_x = row[f'det{prev_i}_x']
                        d_prev_y = row[f'det{prev_i}_y']
                        dist_det = np.sqrt((d_prev_x - det_x)**2 + (d_prev_y - det_y)**2)
                        if dist_det < dist_min:
                            close = True
                    if not close:
                        valid = True

                row.update({f'det{i}_x': det_x, f'det{i}_y': det_y})

            x1, y1 = row['det1_x'], row['det1_y']
            x2, y2 = row['det2_x'], row['det2_y']
            x3, y3 = row['det3_x'], row['det3_y']

            triangle_area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

            if triangle_area >= min_area:
                geom_valid = True

        for i in range(1, 4):
            dx, dy = row[f'det{i}_x'], row[f'det{i}_y']
            dist = np.sqrt((dx - src_x)**2 + (dy - src_y)**2)
            reading = (del_cons * (src_i0 / (dist**2))) + back_rad
            row[f'det{i}_I'] = reading

        data.append(row)
    return pd.DataFrame(data)