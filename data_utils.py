from datetime import datetime, timedelta
import pandas as pd

# ---------------------------------
# Calendar Utils
# ---------------------------------

def __fill_holidays(df, warehouses, holidays):
    for item in holidays:
        dates, holiday_name = item
        generated_dates = [datetime.strptime(
            date, '%m/%d/%Y').strftime('%Y-%m-%d') for date in dates]
        for generated_date in generated_dates:
            df.loc[(df['warehouse'].isin(warehouses)) & (
                df['date'] == generated_date), 'holiday'] = 1
            df.loc[(df['warehouse'].isin(warehouses)) & (
                df['date'] == generated_date), 'holiday_name'] = holiday_name
    return df


def add_holidays(calendar: pd.DataFrame):
    czech_holiday = [
        (['03/31/2024', '04/09/2023', '04/17/2022',
          '04/04/2021', '04/12/2020'], 'Easter Day'),
        (['05/12/2024', '05/10/2020', '05/09/2021',
          '05/08/2022', '05/14/2023'], "Mother Day")]
    calendar = __fill_holidays(calendar, warehouses=[
        'Prague_1', 'Prague_2', 'Prague_3'], holidays=czech_holiday)

    brno_holiday = [
        (['03/31/2024', '04/09/2023', '04/17/2022',
          '04/04/2021', '04/12/2020'], 'Easter Day'),
        (['05/12/2024', '05/10/2020', '05/09/2021',
          '05/08/2022', '05/14/2023'], "Mother Day")]
    calendar = __fill_holidays(calendar, warehouses=[
        'Brno_1'], holidays=brno_holiday)

    munich_holidays = [
        (['03/30/2024', '04/08/2023', '04/16/2022',
          '04/03/2021'], 'Holy Saturday'),
        (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day')]
    calendar = __fill_holidays(calendar, warehouses=[
        'Munich_1'], holidays=munich_holidays)

    frank_holidays = [
        (['03/30/2024', '04/08/2023', '04/16/2022',
          '04/03/2021'], 'Holy Saturday'),
        (['05/12/2024', '05/14/2023', '05/08/2022', '05/09/2021'], 'Mother Day')]
    calendar = __fill_holidays(calendar, warehouses=[
        'Frankfurt_1'], holidays=frank_holidays)

    calendar.loc[(calendar['holiday'] == 1) & (calendar['holiday_name'].isna()),
                 'holiday_name'] = 'Easter Monday'
    datesx = ['03/31/2024', '04/09/2023',
              '04/17/2022', '04/04/2021', '04/12/2020']
    holidaysx = [datetime.strptime(
        date, '%m/%d/%Y') - timedelta(days=1) for date in datesx]
    warehouses = ['Prague_1', 'Prague_2', 'Prague_3']
    calendar.loc[(calendar['date'].isin(holidaysx)) & (
        calendar['warehouse'].isin(warehouses)), 'holiday'] = 1

    return calendar


def __process_minical(df):
    df = df.sort_values('date').reset_index(drop=True)
    # days until next holiday
    df['next_holiday_date'] = df.loc[df['holiday'] == 1, 'date'].shift(-1)
    df['next_holiday_date'] = df['next_holiday_date'].bfill()
    df['days_to_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
    df.drop(columns=['next_holiday_date'], inplace=True)

    # days until shops closed
    df['next_shops_closed_date'] = df.loc[df['shops_closed']
                                          == 1, 'date'].shift(-1)
    df['next_shops_closed_date'] = df['next_shops_closed_date'].bfill()
    df['days_to_shops_closed'] = (
        df['next_shops_closed_date'] - df['date']).dt.days
    df.drop(columns=['next_shops_closed_date'], inplace=True)

    df['day_after_closing'] = (
        (df['shops_closed'] == 0) & (df['shops_closed'].shift(1) == 1)
    ).astype(int)
    df['holiday_name'] = df['holiday_name'].fillna('No Holiday')
    return df


def process_calendar(calendar: pd.DataFrame):
    calendar['date'] = pd.to_datetime(calendar['date'])
    # calendar['events'] = calendar[['holiday', 'shops_closed',
    #                               'winter_school_holidays', 'school_holidays']].sum(axis=1)
    warehouses = ['Frankfurt_1', 'Munich_1', 'Budapest_1',
                  'Brno_1', 'Prague_1', 'Prague_2', 'Prague_3']
    dfs = [calendar.query(
        f'date >= "2020-08-01 00:00:00" and warehouse == "{warehouse}"') for warehouse in warehouses]
    extended = pd.concat([__process_minical(df) for df in dfs]).sort_values(
        'date').reset_index(drop=True)
    return extended


def one_day_calendar(calendar: pd.DataFrame):
    numeric = calendar.select_dtypes('number').columns.to_list()
    numeric_cal = calendar[[
        'date'] + numeric].groupby('date').mean().reset_index().drop_duplicates()

    holidays = calendar[['date', 'holiday_name']].copy()
    dates = holidays[['date']].drop_duplicates()
    real_holidays = holidays.loc[holidays['holiday_name'] != 'No Holiday', [
        'date', 'holiday_name']].groupby('date').first().reset_index()
    holidays = dates.merge(real_holidays, how='left',
                           on='date').fillna('No Holiday')

    cal = numeric_cal.merge(holidays, how='left', on='date')
    cal['date'] = pd.to_datetime(cal['date'])

    return cal

# ---------------------------------
# Inventory Utils
# ---------------------------------

def process_inventory(inventory: pd.DataFrame):
    inventory['kind'] = inventory['name'].apply(lambda x: x.split('_')[0])
    inventory.drop('name', axis=1, inplace=True)
    inventory.rename(
        columns={f'L{i}_category_name_en': f'L{i}' for i in range(1, 5)}, inplace=True)

    inventory.loc[(inventory['kind'] == 'Snack') & (
        inventory['L1'] == 'Bakery'), 'kind'] = 'Bakery_Snack'
    inventory.loc[inventory['kind'] == 'Bell pepper', 'kind'] = 'Bell Pepper'
    inventory.loc[inventory['kind'] ==
                  'Brussels sprout', 'kind'] = 'Brussels Sprout'
    inventory.loc[inventory['kind'] == 'Breadcrumbs', 'kind'] = 'Breadcrumb'

    return inventory
