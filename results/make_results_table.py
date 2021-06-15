import glob
import json
import pandas as pd

files = glob.glob('*_results.json')

df = {}

for f in files:
    with open(f) as _: results = json.load(_)
    df[f[:-5]] = {
        k: (v if ('quantile' in k) else v*1000/300/8760*100) for k, v in results.items()}

dfT = pd.DataFrame(df).T

# dfT.to_csv('results_table.csv', float_format='%5.2f', index_label='station')

dfT.to_csv('results_table.csv', float_format='%4.1f', index_label='station')
