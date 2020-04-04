import pandas as pd
import numpy as np

if __name__ == '__main__':
    prng = np.random.RandomState(0)
    T = 200
    row_list = list()
    for t in range(T):
        count = prng.poisson(50)
        row_list.append(dict(timestep=t, num_Presenting=count))
    df = pd.DataFrame(row_list)
    df.to_csv("presenting.csv", index=False)
