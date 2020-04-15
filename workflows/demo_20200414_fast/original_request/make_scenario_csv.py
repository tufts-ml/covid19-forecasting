import pandas as pd

if __name__ == '__main__':

    xlsx = pd.ExcelFile('20200414-scenarios.xlsx')
    print(xlsx.sheet_names)

    col_map = {
        "Day":"timestep",
        "Present for Test":"num_Presenting",
        "Admit to ICU":"num_OffVentInICU",
        }

    for ss, sheet in enumerate(xlsx.sheet_names):
        csv_df = xlsx.parse(sheet)
        csv_df = csv_df.rename(col_map, axis='columns')  

        csv_file = "incoming-%d.csv" % (ss+1)
        csv_df.to_csv(csv_file, index=False, float_format='%.0f')

