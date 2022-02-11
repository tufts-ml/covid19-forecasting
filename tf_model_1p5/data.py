import os
import pandas as pd

def read_data(data_dir='./data', covid_estim_date='20210901', hhs_date='20210903', owid_date='20210903',
              state='Massachusetts', state_abbrev='MA'):
    """Read in HHS, OWID, and CovidEstim datasets. Return a single dataframe with a DatetimeIndex and all columns.

    Args:
        data_dir (str): Path to directory containing directories named "covidestim", "hhs", and "owid", with files
            named YYYYMMDD[_vaccinations].csv where the date is the date of retrieval
        covid_estim_date (str): Date in YYYYMMDD format of the covid estim file to use. Will read
            [data_dir]/covid_estim/[covid_estim_date].csv
        hhs_date (str): Date in YYYYMMDD format of the hhs file to use. Will read
            [data_dir]/hhs/[hhs_date].csv
        owid_date (str): Date in YYYYMMDD format of the Our World In Data file to use. Will read
            [data_dir]/owid/[owid_date]_vaccinations.csv
        state (str): Full, capitalized state name
        state_abbrev (str): 2-letter state abbreviation

    Returns
        pd.DataFrame with a Datetime index and a column for:
            'general_ward' - from hhs
            'asymp' - from covid estim
            'extreme' - from covid estim
            'mild' - from covid estim
            'Rt' - from covid estim
            'vax_pct' - from OWID
    """

    covid_estim_path = os.path.join(data_dir, 'covidestim', covid_estim_date + '.csv')
    hhs_path = os.path.join(data_dir, 'hhs', hhs_date + '.csv')
    owid_path = os.path.join(data_dir, 'owid', owid_date + '_vaccinations.csv')

    # Read covid estim
    covid_estim = pd.read_csv(covid_estim_path)
    # filter to state
    covid_estim = covid_estim[covid_estim['state'] == state]
    # Make sure date is unique
    assert len(covid_estim) == len(covid_estim.date.unique())

    # create datetime index
    covid_estim.loc[:, 'date'] = pd.to_datetime(covid_estim['date'])
    covid_estim = covid_estim.set_index('date').sort_index()
    # Rename to match our compartments
    covid_estim = covid_estim.rename(columns={'infections': 'asymp',
                                              'severe': 'extreme',
                                              'symptomatic': 'mild'
                                              })

    hhs = pd.read_csv(hhs_path)
    hhs = hhs[hhs['state'] == state_abbrev]
    assert len(hhs) == len(hhs.date.unique())

    hhs.loc[:, 'date'] = pd.to_datetime(hhs['date'])

    hhs = hhs.rename(columns={'adult_icu_bed_covid_utilization_numerator': 'icu_count',
                              'inpatient_beds_used_covid': 'general_ward_count'})

    # we have previous day admissions, so add 1 to date
    gen = hhs.copy()
    gen.loc[:, 'date'] = gen['date'] + pd.DateOffset(days=1)
    gen = gen.set_index('date').sort_index()
    gen = gen.rename(columns={'previous_day_admission_adult_covid_confirmed': 'general_ward_in'})

    hhs = hhs.set_index('date').sort_index()

    owid = pd.read_csv(owid_path)
    owid = owid[owid['location'] == state]
    owid.loc[:, 'date'] = pd.to_datetime(owid['date'])
    owid = owid.set_index('date').sort_index()
    # There are NaN's in the vaccine timeseries, so linearly interpolate
    owid.loc[:, 'people_fully_vaccinated_per_hundred'] = owid[['people_fully_vaccinated_per_hundred']].interpolate(
        method='linear')
    owid['vax_pct'] = owid[['people_fully_vaccinated_per_hundred']] * 0.01

    df = pd.merge(hhs[['icu_count', 'general_ward_count', 'deaths_covid']],
                  covid_estim[['asymp', 'extreme', 'mild', 'Rt']],
                  how='outer',
                  left_index=True, right_index=True).merge(
        gen[['general_ward_in']], how='outer', left_index=True, right_index=True
    ).merge(owid[['vax_pct']], how='outer', left_index=True, right_index=True)

    return df

def create_warmup(df, warmup_start, warmup_end, vax_asymp_risk, vax_mild_risk, vax_gen_risk, vax_icu_risk):
    """Create arrays of warmup data using a dataframe with a datetime index. Incorrectly splits on vax. status.

    Args:
        df (pd.DataFrame): Pandas dataframe with a column for everything we need
        warmup_start (str): Date string in YYYYMMDD format
        warmup_end (str): Date string in YYYYMMDD format
        vax_asymp_risk (float): Vaccine efficacy at preventing asymptomatic cases
        vax_mild_risk (float):  Vaccine efficacy at preventing mild cases
        vax_extreme_risk (float): Vaccine efficacy at preventing extreme cases
    Returns:
        (warmup_asymp, warmup_mild, warmup_extreme) (dict{int:np.array}, dict{int:np.array}, dict{int:np.array}):
            A tuple of 3 dictionaries, keyed on vaccination status (0 = not vaccinated, 1 = vaccinated) with values
            corresponding to the number of new entries into that compartment during the warmup period each day.
    """

    warmup_infected = {}
    warmup_asymp = {}
    warmup_mild = {}
    warmup_gen = {}
    count_gen = {}
    count_icu = {}

    not_vaxxed = 0
    vaxxed = 1

    # vaccines protect from infection/asymp/mild/extreme/general
    # multiply population vaccinated by (1-protection %) to get vaccinated
    # TODO: This math is wrong
    warmup_asymp[vaxxed] = (df.loc[warmup_start:warmup_end, 'vax_pct'] * (1 - vax_asymp_risk) * \
                            df.loc[warmup_start:warmup_end, 'asymp']).values
    warmup_asymp[not_vaxxed] = df.loc[warmup_start:warmup_end, 'asymp'].values - warmup_asymp[vaxxed]

    warmup_mild[vaxxed] = (df.loc[warmup_start:warmup_end, 'vax_pct'] * (1 - vax_mild_risk) * \
                           df.loc[warmup_start:warmup_end, 'mild']).values
    warmup_mild[not_vaxxed] = df.loc[warmup_start:warmup_end, 'mild'].values - warmup_mild[vaxxed]

    warmup_gen[vaxxed] = (df.loc[warmup_start:warmup_end, 'vax_pct'] * (1 - vax_gen_risk) * \
                              df.loc[warmup_start:warmup_end, 'general_ward_in']).values
    warmup_gen[not_vaxxed] = df.loc[warmup_start:warmup_end, 'general_ward_in'].values - warmup_gen[vaxxed]

    count_gen[vaxxed] = (df.loc[warmup_end, 'vax_pct'] * (1 - vax_gen_risk) * \
                          df.loc[warmup_end, 'general_ward_count'])
    count_gen[not_vaxxed] = df.loc[warmup_end, 'general_ward_count'] - count_gen[vaxxed]

    count_icu[vaxxed] = (df.loc[warmup_end, 'vax_pct'] * (1 - vax_gen_risk) * \
                         df.loc[warmup_end, 'icu_count'])
    count_icu[not_vaxxed] = df.loc[warmup_end, 'icu_count'] - count_icu[vaxxed]

    return warmup_asymp, warmup_mild, count_gen, count_icu

