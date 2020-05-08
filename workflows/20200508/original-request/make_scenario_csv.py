import pandas as pd

if __name__ == '__main__':

    xlsx = pd.ExcelFile('scenarios.xlsx')
    print(xlsx.sheet_names)
    n_skip = 2

    col_map = {
        "Day":"timestep",
        "Present for Test":"num_Presenting",
        "Admit to ICU":"num_OffVentInICU",
        }

    for ss, sheet in enumerate(xlsx.sheet_names[n_skip:]):
        csv_df = xlsx.parse(sheet)
        csv_df = csv_df.rename(col_map, axis='columns')  

        csv_file = "incoming-%d.csv" % (ss+1)
        csv_df.to_csv(csv_file, index=False, float_format='%.0f')

