import pandas as pd
import numpy as np
import datetime
pd.set_option('display.max_columns', 500)

def parse_datetime_column(df, date_column='DATETIME'):
    """
    Attempts to parse the specified datetime column to ensure consistency in date format.
    Converts inconsistent formats to datetime and fills NaT values.
    """
    # Convert and fill NaT values
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df[date_column] = df[date_column].fillna(method='ffill').fillna(method='bfill')
    return df

def identify_attacks(test_data):
    """
    Given the test_data identifies the attack intervals and creates a pandas DataFrame where those spoofing is going to be applied.
    
    Returns
    -------
    DataFrame
        summary of the attack intervals
    """
    # find attacks among data
    attacks = test_data.loc[test_data['ATT_FLAG'] == 1]
    prev_datetime = attacks.index[0]
    start = prev_datetime
    count_attacks = 0

    # find attacks bounds
    attack_intervals = pd.DataFrame(columns=['Name', 'Start', 'End', 'Replay_Copy'])
    for index, _ in attacks.iterrows():
        if count_attacks == 3:
            count_attacks += 1
        if (index - prev_datetime > 1):
            count_attacks += 1
            interval = pd.DataFrame([['attack_' + str(count_attacks), start, prev_datetime, (start - (prev_datetime - start)) - 200]],
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
    Given a spoofing_technique to be applied, the attack_intervals, eavesdropped_data and test_data, it builds the dataset containing sensor spoofing.
    
    Returns
    -------
    DataFrame
        Dataset with spoofed sensor readings.
    """
    df2 = pd.DataFrame()
    if dataset == 'WADI':
        if att_num < 5:
            row = attack_intervals.iloc[att_num - 1]
        else:
            row = attack_intervals.iloc[att_num - 2]
    if dataset == 'BATADAL':
        if att_num < 8:
            row = attack_intervals.iloc[att_num - 1]
        else:
            row = attack_intervals.iloc[att_num - 8]
    df = pd.DataFrame(columns=test_data.columns)
    df = spoofing_technique(df, row, eavesdropped_data, attack_intervals, constraints)
    df['ATT_FLAG'] = 1
    
    return df

def replay(df, row, eavesdropped_data, attack_intervals, *args):
    """
    Applies replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    df = pd.concat([df, eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy'] + (row['End'] - row['Start'])]])[test_data.columns.tolist()]
    return df

def constrained_replay(df, row, eavesdropped_data, attack_intervals, *args):
    """
    Constrained version of the replay attack
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
print(args.data)

dataset = args.data[0]

if __name__ == "__main__":
    constraints_setting = args.constraint_setting[0]
    if dataset == 'BATADAL':
        if constraints_setting == 'topology':
            list_of_constraints = ['PLC_1', 'PLC_2', 'PLC_3', 'PLC_4', 'PLC_5', 'PLC_6', 'PLC_7', 'PLC_8', 'PLC_9']
        else:
            list_of_constraints = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
    if dataset == 'WADI':
        if constraints_setting == 'topology':
            list_of_constraints = ['PLC_1', 'PLC_2']
        else:
            list_of_constraints = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]

    data_folder = '../../Data/' + dataset

    for i in list_of_constraints:
        if dataset == 'BATADAL':
            # 파일 읽기 (parse_dates 및 date_parser 제거)
            test_data = pd.read_csv(data_folder + '/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
            test_data = parse_datetime_column(test_data, date_column='DATETIME')
            test_data.set_index('DATETIME', inplace=True)

            eavesdropped_data = pd.read_csv(data_folder + "/test_dataset_1.csv").drop(columns=['Unnamed: 0'], axis=1)
            eavesdropped_data = parse_datetime_column(eavesdropped_data, date_column='DATETIME')
            eavesdropped_data.set_index('DATETIME', inplace=True)

        if dataset == 'WADI':
            test_data = pd.read_csv(data_folder + '/attacks_october_clean_with_label.csv')
            test_data = parse_datetime_column(test_data, date_column='DATETIME')
            test_data.set_index('DATETIME', inplace=True)

            eavesdropped_data = pd.read_csv(data_folder + "/train_dataset.csv")
            eavesdropped_data = parse_datetime_column(eavesdropped_data, date_column='DATETIME')
            eavesdropped_data.set_index('DATETIME', inplace=True)

        constraints = []
        actuator_columns = eavesdropped_data.filter(regex=("STATUS")).columns.tolist()

        spoofing_technique = constrained_replay
        attack_intervals = identify_attacks(test_data)

        if dataset == 'BATADAL': 
            for att_num in range(1, 8): 
                if constraints_setting == 'topology':
                    s = open('../Black_Box_Attack/constraints/' + dataset + '/constraint_PLC.txt', 'r').read()
                else:
                    s = open('../Whitebox_Attack/constraints/' + dataset + '/constraint_variables_attack_' + str(att_num) + '.txt', 'r').read()
                dictionary = eval(s)
                constraints.append(dictionary[i])

                print('ATT Num:', att_num)
                # 파일 읽기 및 날짜 변환 적용
                test_data = pd.read_csv(
                    '../../Data/BATADAL/attack_' + str(att_num) + '_from_test_dataset.csv'
                ).drop(columns=['Unnamed: 0'], axis=1)
                test_data = parse_datetime_column(test_data, date_column='DATETIME')
                test_data.set_index('DATETIME', inplace=True)

                spoofed_data = spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, att_num, constraints)
                output_path = './results/BATADAL/attack_' + str(att_num) + '_replay_max_' + str(i) + '.csv'
                if constraints_setting == 'topology':
                    output_path = './results/BATADAL/constrained_PLC/constrained_' + str(i) + '_attack_' + str(att_num) + '.csv'
                spoofed_data.to_csv(output_path)

        if dataset == 'WADI':
            for att_num in [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
                if constraints_setting == 'topology':
                    s = open('../Black_Box_Attack/constraints/' + dataset + '/constraint_PLC.txt', 'r').read()
                else:
                    s = open('../Whitebox_Attack/constraints/' + dataset + '/constraint_variables_attack_' + str(att_num) + '.txt', 'r').read()
                
                dictionary = eval(s)
                constraints.append(dictionary[i])

                print('ATT Num:', att_num)
                # 파일 읽기 및 날짜 변환 적용
                test_data = pd.read_csv(
                    '../../Data/' + dataset + '/attack_' + str(att_num) + '_from_test_dataset.csv'
                )
                test_data = parse_datetime_column(test_data, date_column='DATETIME')
                test_data.set_index('DATETIME', inplace=True)

                spoofed_data = spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, att_num, constraints)

                output_path = './results/' + dataset + '/attack_' + str(att_num) + '_replay_max_' + str(i) + '.csv'
                if constraints_setting == 'topology':
                    output_path = './results/' + dataset + '/constrained_PLC/constrained_' + str(i) + '_attack_' + str(att_num) + '.csv'

                spoofed_data.to_csv(output_path)
