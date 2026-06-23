import numpy as np
import pandas as pd
def generate_dataset_raw(num_samples=30000,area_size=18,del_cons=1.2,back_rad=0.0,SI_min=400.0,SI_max=8000.0,dist_min=0.5):
      data=[]
      for _ in range(num_samples):
          src_x=np.random.uniform(0,area_size)
          src_y=np.random.uniform(0,area_size)
          src_i0=np.random.uniform(SI_min,SI_max)
          row={'source_x':src_x,'source_y':src_y,'I_0':src_i0}
          for i in range(1,4):
              valid=False
              while not valid:
                  det_x=np.random.uniform(0,area_size)
                  det_y=np.random.uniform(0,area_size)
                  dist_src=np.sqrt((src_x-det_x)**2+(src_y-det_y)**2)
                  close=dist_src<dist_min
                  for prev_i in range(1,i):
                      d_prev_x=row[f'det{prev_i}_x']
                      d_prev_y=row[f'det{prev_i}_y']
                      dist_det=np.sqrt((d_prev_x-det_x)**2+(d_prev_y-det_y)**2)
                      if dist_det<dist_min:
                          close=True
                  if not close:
                      valid=True
              dist=np.sqrt((det_x-src_x)**2+(det_y-src_y)**2)
              reading=(del_cons*(src_i0/(dist**2)))+back_rad
              row.update({f'det{i}_x':det_x,f'det{i}_y':det_y,f'det{i}_I':reading})
          data.append(row)
      return pd.DataFrame(data)