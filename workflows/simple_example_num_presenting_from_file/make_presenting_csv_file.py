import pandas as pd
import numpy as np

if __name__ == '__main__':
    for seed in [101, 201, 301]:
        prng = np.random.RandomState(seed)
        T = 200
        row_list = list()
        for t in range(T):
            count = prng.poisson(50)
            row_list.append(dict(timestep=t, num_Presenting=count))
        df = pd.DataFrame(row_list)
        df.to_csv("presenting-random_seed=%d.csv" % seed, index=False)
