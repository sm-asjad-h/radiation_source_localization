import numpy as np
import pandas as pd
import math
SOURCE_LIBRARY = {
    'Am-241': {'gamma': 0.015, 'rho': 0.0225, 'd3s_eff': 0.20},  # 60 keV 
    'Co-57':  {'gamma': 0.090, 'rho': 0.0175, 'd3s_eff': 0.18},  # 122 keV
    'Tc-99m': {'gamma': 0.072, 'rho': 0.0166, 'd3s_eff': 0.15},  # 140 keV
    'I-131':  {'gamma': 0.220, 'rho': 0.0124, 'd3s_eff': 0.10},  # 364 keV
    'Ir-192': {'gamma': 0.480, 'rho': 0.0105, 'd3s_eff': 0.08},  # 400 keV
    'F-18':   {'gamma': 0.570, 'rho': 0.0098, 'd3s_eff': 0.06},  # 511 keV
    'Cs-137': {'gamma': 0.330, 'rho': 0.0092, 'd3s_eff': 0.05},  # 662 keV
    'Ra-226': {'gamma': 0.825, 'rho': 0.0080, 'd3s_eff': 0.04},  # ~800 keV
    'Na-22':  {'gamma': 1.200, 'rho': 0.0065, 'd3s_eff': 0.03},  # 1.27 MeV
    'Co-60':  {'gamma': 1.320, 'rho': 0.0060, 'd3s_eff': 0.02}   # 1.25+ MeV
}

D3S_BACKGROUND = 0#5.0

def generate_dataset(num_samples=30000, area_size=18,SI_min=400.0, SI_max=8000.0, dist_min=2,min_angle=25.0, max_angle=130.0,max_flat_dist=25,num_bins=25):
    
    data = []
    print(f"Generating {num_samples} samples using D3S detector using 10 randomized source types")

    available_sources = list(SOURCE_LIBRARY.keys())

    corners = np.array([[0,0], [0, area_size], [area_size, 0], [area_size, area_size]])
    bins = np.linspace(dist_min, max_flat_dist, num_bins + 1)
    bin_counts = np.zeros(num_bins)
    target_per_bin = (num_samples * 3) / num_bins

    def get_needed_r():
        valid_bin_indices = np.where(bin_counts < target_per_bin)[0]
        if len(valid_bin_indices) == 0:
            chosen_bin = np.random.randint(0, num_bins)
        else:
            chosen_bin = np.random.choice(valid_bin_indices)
        r = np.random.uniform(bins[chosen_bin], bins[chosen_bin+1])
        return r, chosen_bin

    for sample_idx in range(num_samples):
        geom_valid = False

        while not geom_valid:
            r_targets = []
            bin_indices = []
            for _ in range(3):
                r, b_idx = get_needed_r()
                r_targets.append(r)
                bin_indices.append(b_idx)

            max_r_needed = max(r_targets)
            src_valid = False
            for _ in range(200):
                if max_r_needed > 21.0:
                    src_x = np.random.beta(0.1, 0.1) * area_size
                    src_y = np.random.beta(0.1, 0.1) * area_size
                elif max_r_needed > 15.0:
                    src_x = np.random.beta(0.5, 0.5) * area_size
                    src_y = np.random.beta(0.5, 0.5) * area_size
                else:
                    src_x = np.random.uniform(0, area_size)
                    src_y = np.random.uniform(0, area_size)

                max_r_possible = np.max(np.sqrt(np.sum((corners - [src_x, src_y])**2, axis=1)))

                if max_r_possible >= max_r_needed:
                    src_valid = True
                    src_i0 = np.random.uniform(SI_min, SI_max)
                    break

            if not src_valid:
                continue

            chosen_source = np.random.choice(available_sources)
            gamma = SOURCE_LIBRARY[chosen_source]['gamma']
            rho = SOURCE_LIBRARY[chosen_source]['rho']
            eff = SOURCE_LIBRARY[chosen_source]['d3s_eff'] 
            
            row = {'source_x': src_x, 'source_y': src_y, 'I_0': src_i0}
            detectors = []

            for i in range(3):
                r = r_targets[i]
                det_placed = False

                for theta_attempt in range(250):
                    if r > 18.0:
                        dists_to_corners = np.sum((corners - [src_x, src_y])**2, axis=1)
                        furthest_corner = corners[np.argmax(dists_to_corners)]
                        base_angle = np.arctan2(furthest_corner[1] - src_y, furthest_corner[0] - src_x)
                        theta = base_angle + np.random.uniform(-0.26, 0.26)
                    else:
                        theta = np.random.uniform(0, 2 * np.pi)

                    dx = src_x + r * np.cos(theta)
                    dy = src_y + r * np.sin(theta)

                    if 0 <= dx <= area_size and 0 <= dy <= area_size:
                        too_close = False
                        for prev_dx, prev_dy in detectors:
                            if np.sqrt((prev_dx - dx)**2 + (prev_dy - dy)**2) < dist_min:
                                too_close = True
                                break

                        if not too_close:
                            detectors.append((dx, dy))
                            row[f'det{i+1}_x'] = dx
                            row[f'det{i+1}_y'] = dy
                            det_placed = True
                            break

                if not det_placed:
                    break

            if len(detectors) != 3:
                continue

            x1, y1 = row['det1_x'], row['det1_y']
            x2, y2 = row['det2_x'], row['det2_y']
            x3, y3 = row['det3_x'], row['det3_y']

            a = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
            b = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
            c = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

            cos_A = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)
            cos_B = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0)
            cos_C = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)

            angle_A = np.degrees(np.arccos(cos_A))
            angle_B = np.degrees(np.arccos(cos_B))
            angle_C = np.degrees(np.arccos(cos_C))

            valid_angles = (min_angle <= angle_A <= max_angle) and \
                           (min_angle <= angle_B <= max_angle) and \
                           (min_angle <= angle_C <= max_angle)

            if valid_angles:
                geom_valid = True

        for b_idx in bin_indices:
            bin_counts[b_idx] += 1
        if np.random.random()>0.9:
          row['source_x']=-1
          row['source_y']=-1
          row['I_0']=0+1e-6
          for i in range(1, 4):
              row[f'det{i}_I'] = 0+D3S_BACKGROUND+1e-6
        else:
          for i in range(1, 4):
              dx, dy = row[f'det{i}_x'], row[f'det{i}_y']
              dist = np.sqrt((dx - src_x)**2 + (dy - src_y)**2)
              
              raw_signal = (eff * gamma * src_i0) / (dist**2)
              attenuation = np.exp(-rho * dist)
              lambda_cps = (raw_signal * attenuation) + D3S_BACKGROUND
              reading = np.random.poisson(lam=lambda_cps)
              if reading==0:
                reading=1
              row[f'det{i}_I']=reading
              
        
        data.append(row)

        if (sample_idx + 1) % 10000 == 0:
            print(f"Generated {sample_idx + 1}/{num_samples} samples...")

    return pd.DataFrame(data)

df = generate_dataset(num_samples=300000)

df.to_csv('/home/lus04/trainee4/radiation_source_localization/v1.7_poissondata/dataset/dataset2.csv'  ,index=False)

