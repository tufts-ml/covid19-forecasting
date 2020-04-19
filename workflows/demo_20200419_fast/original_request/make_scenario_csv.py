import pandas as pd

if __name__ == '__main__':

    xlsx = pd.ExcelFile('scenarios.xlsx')
    print(xlsx.sheet_names)

    col_map = {
        "Day":"timestep",
        "Present for Test":"num_Presenting",
        "Admit to ICU":"num_OffVentInICU",
        }

    MIN_SHEET_ID = 2
    for ss, sheet in enumerate(xlsx.sheet_names):
        if ss < MIN_SHEET_ID:
            continue
        csv_df = xlsx.parse(sheet)
        csv_df = csv_df.rename(col_map, axis='columns')  

        csv_file = "incoming-%d.csv" % (ss - MIN_SHEET_ID + 1)
        csv_df.to_csv(csv_file, index=False, float_format='%.0f')

