import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)

import pandas as pd
import numpy as np

def parse_datetime_column(df, date_column='DATETIME'):
    """
    Converts the specified datetime column to a consistent datetime format.
    Handles both standard datetime strings and numeric time representations.
    """
    def convert_numeric_to_datetime(x):
        # If numeric, treat as hours and add to a base date
        if isinstance(x, (int, float)):
            return pd.Timestamp("2017-01-01") + pd.to_timedelta(x, unit='h')
        return pd.NaT

    # First, attempt to convert standard datetime strings
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # For any remaining NaT values (likely from numeric entries), handle as numeric
    numeric_mask = df[date_column].isna() & df[date_column].astype(str).str.isnumeric()
    df.loc[numeric_mask, date_column] = df.loc[numeric_mask, date_column].apply(convert_numeric_to_datetime)

    # Fill NaT values with nearest valid dates
    df[date_column] = df[date_column].fillna(method='ffill').fillna(method='bfill')
    return df

def identify_attacks(test_data):
    """
    Identifies attack intervals in test_data and returns a DataFrame summarizing these intervals.
    """
    attacks = test_data.loc[test_data['ATT_FLAG'] == 1]
    prev_datetime = attacks.index[0]
    start = prev_datetime
    count_attacks = 0

    attack_intervals = pd.DataFrame(columns=['Name', 'Start', 'End', 'Replay_Copy'])
    for index, _ in attacks.iterrows():
        if count_attacks == 3:
            count_attacks += 1
        if (index - prev_datetime > 1):
            count_attacks += 1
            interval = pd.DataFrame([['attack_' + str(count_attacks), start, prev_datetime, start - (prev_datetime - start) - 200]],
                                    columns=['Name', 'Start', 'End', 'Replay_Copy'])
            attack_intervals = pd.concat([attack_intervals, interval], ignore_index=True)
            start = index
        prev_datetime = index
    count_attacks += 1
    interval = pd.DataFrame([['attack_' + str(count_attacks), start, prev_datetime, start - (prev_datetime - start) - 200]],
                            columns=['Name', 'Start', 'End', 'Replay_Copy'])
    attack_intervals = pd.concat([attack_intervals, interval], ignore_index=True)

    print('_________________________________ATTACK INTERVALS___________________________________\n')
    print(attack_intervals)
    print('____________________________________________________________________________________')
    return attack_intervals

def spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, att_num, constraints=None):
    """
    Applies a spoofing technique to the test_data based on attack intervals and returns spoofed data.
    """
    df2 = pd.DataFrame()
    if dataset == 'WADI':
        row = attack_intervals.iloc[att_num - 1 if att_num < 5 else att_num - 2]
    elif dataset == 'BATADAL':
        row = attack_intervals.iloc[att_num - 1 if att_num < 8 else att_num - 8]
    df = pd.DataFrame(columns=test_data.columns)
    df = spoofing_technique(df, row, eavesdropped_data, attack_intervals, constraints)
    df['ATT_FLAG'] = 1
    return df

def replay(df, row, eavesdropped_data, attack_intervals, *args):
    """
    Applies replay attack to input data.
    """
    df = pd.concat([df, eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy'] + (row['End'] - row['Start'])]])[test_data.columns.tolist()]
    return df

def constrained_replay(df, row, eavesdropped_data, attack_intervals, *args):
    """
    Constrained version of the replay attack.
    """
    constraints = args[0]
    check_constraints(constraints)
    try:
        test_data[constraints[0]] = eavesdropped_data[constraints[0]].loc[row['Replay_Copy']:row['Replay_Copy'] + (row['End'] - row['Start']) + 1].values
    except:
        try:
            test_data[constraints[0]] = eavesdropped_data[constraints[0]].loc[row['Replay_Copy']:row['Replay_Copy'] + (row['End'] - row['Start'])].values
        except:
            test_data[constraints[0]] = eavesdropped_data[constraints[0]].loc[row['Replay_Copy']:row['Replay_Copy'] + (row['End'] - row['Start']) - 1].values
    return test_data

def check_constraints(constraints):
    if constraints is None:
        print('Provide constraints')
        import sys
        sys.exit()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', type=str, default=['BATADAL'])
parser.add_argument('-c', '--constraint_setting', nargs='+', type=str, default=['best'])

args = parser.parse_args()
dataset = args.data[0]
constraints_setting = args.constraint_setting[0]

if __name__ == "__main__":
    if dataset == 'BATADAL':
        list_of_constraints = ['PLC_1', 'PLC_2', 'PLC_3', 'PLC_4', 'PLC_5', 'PLC_6', 'PLC_7', 'PLC_8', 'PLC_9'] if constraints_setting == 'topology' else [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
    elif dataset == 'WADI':
        list_of_constraints = ['PLC_1', 'PLC_2'] if constraints_setting == 'topology' else [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]

    data_folder = f'../../Data/{dataset}'
    for i in list_of_constraints:
        test_data = pd.read_csv(f'{data_folder}/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
        test_data = parse_datetime_column(test_data, date_column='DATETIME')
        test_data.set_index('DATETIME', inplace=True)

        eavesdropped_data = pd.read_csv(f'{data_folder}/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
        eavesdropped_data = parse_datetime_column(eavesdropped_data, date_column='DATETIME')
        eavesdropped_data.set_index('DATETIME', inplace=True)

        constraints = []
        attack_intervals = identify_attacks(test_data)

        if dataset == 'BATADAL': 
            for att_num in range(1, 8): 
                s = open(f'../{"Black_Box_Attack" if constraints_setting == "topology" else "Whitebox_Attack"}/constraints/{dataset}/{"constraint_PLC" if constraints_setting == "topology" else f"constraint_variables_attack_{att_num}"}.txt').read()
                dictionary = eval(s)
                constraints.append(dictionary[i])

                test_data = pd.read_csv(f'../../Data/BATADAL/attack_{att_num}_from_test_dataset.csv').drop(columns=['Unnamed: 0'], axis=1)
                test_data = parse_datetime_column(test_data, date_column='DATETIME')
                test_data.set_index('DATETIME', inplace=True)

                spoofed_data = spoof(constrained_replay, attack_intervals, eavesdropped_data, test_data, att_num, constraints)
                output_path = f'./results/BATADAL/{"constrained_PLC/constrained_" + str(i) + "_attack_" + str(att_num) if constraints_setting == "topology" else f"attack_{att_num}_replay_max_{i}"}.csv'
                spoofed_data.to_csv(output_path)

        elif dataset == 'WADI':
            for att_num in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                s = open(f'../{"Black_Box_Attack" if constraints_setting == "topology" else "Whitebox_Attack"}/constraints/{dataset}/{"constraint_PLC" if constraints_setting == "topology" else f"constraint_variables_attack_{att_num}"}.txt').read()
                dictionary = eval(s)
                constraints.append(dictionary[i])

                test_data = pd.read_csv(f'../../Data/{dataset}/attack_{att_num}_from_test_dataset.csv')
                test_data = parse_datetime_column(test_data, date_column='DATETIME')
                test_data.set_index('DATETIME', inplace=True)

                spoofed_data = spoof(constrained_replay, attack_intervals, eavesdropped_data, test_data, att_num, constraints)
                output_path = f'./results/{dataset}/{"constrained_PLC/constrained_" + str(i) + "_attack_" + str(att_num) if constraints_setting == "topology" else f"attack_{att_num}_replay_max_{i}"}.csv'
                spoofed_data.to_csv(output_path)
