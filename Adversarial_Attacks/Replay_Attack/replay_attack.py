import pandas as pd
import numpy as np
import argparse

pd.set_option('display.max_columns', 500)

# Define argument parser and parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', type=str, default=['BATADAL'])
parser.add_argument('-c', '--constraint_setting', nargs='+', type=str, default=['best'])
args = parser.parse_args()
dataset = args.data[0]
constraints_setting = args.constraint_setting[0]

def identify_attacks(test_data):
    """
    Identifies attack intervals in test_data and returns a DataFrame summarizing these intervals.
    """
    attacks = test_data.loc[test_data['ATT_FLAG'] == 1]
    if attacks.empty:
        print("Warning: No attacks found in test_data.")
        return pd.DataFrame(columns=['Name', 'Start', 'End', 'Replay_Copy'])

    prev_timestamp = attacks.index[0]
    start = prev_timestamp
    count_attacks = 0
    attack_intervals = pd.DataFrame(columns=['Name', 'Start', 'End', 'Replay_Copy'])

    for index, _ in attacks.iterrows():
        if count_attacks == 3:
            count_attacks += 1
        if (index - prev_timestamp) > 1:  # Set back to 1 if using smaller intervals
            count_attacks += 1
            interval = pd.DataFrame([[f'attack_{count_attacks}', start, prev_timestamp,
                                      start - (prev_timestamp - start) - 200]],
                                    columns=['Name', 'Start', 'End', 'Replay_Copy'])
            if not interval.empty:
                attack_intervals = pd.concat([attack_intervals, interval], ignore_index=True)
            start = index
        prev_timestamp = index

    count_attacks += 1
    interval = pd.DataFrame([[f'attack_{count_attacks}', start, prev_timestamp,
                              start - (prev_timestamp - start) - 200]],
                            columns=['Name', 'Start', 'End', 'Replay_Copy'])
    if not interval.empty:
        attack_intervals = pd.concat([attack_intervals, interval], ignore_index=True)

    print('_________________________________ATTACK INTERVALS___________________________________\n')
    print(attack_intervals)
    print('____________________________________________________________________________________')
    return attack_intervals

def spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, att_num, constraints=None):
    """
    Applies a spoofing technique to the test_data based on attack intervals and returns spoofed data.
    """
    if att_num > len(attack_intervals):
        print(f"Warning: Attack number {att_num} is out of bounds for available intervals.")
        return test_data

    row = attack_intervals.iloc[att_num - 1]
    df = pd.DataFrame(columns=test_data.columns)
    df = spoofing_technique(df, row, eavesdropped_data, attack_intervals, constraints)
    df['ATT_FLAG'] = 1
    return df

def constrained_replay(df, row, eavesdropped_data, attack_intervals, constraints):
    """
    Constrained version of the replay attack using integer timestamps.
    """
    if any(isinstance(c, list) for c in constraints):
        constraints = [item for sublist in constraints for item in sublist]

    check_constraints(constraints)
    end_idx = int(row['Replay_Copy']) + int(row['End'] - row['Start']) + 1

    for constraint in constraints:
        if constraint not in eavesdropped_data.columns:
            print(f"Warning: Constraint '{constraint}' not in eavesdropped data columns.")
            continue

        replay_range = eavesdropped_data[constraint].iloc[int(row['Replay_Copy']):end_idx]
        if not replay_range.empty:
            df[constraint] = replay_range.values
        else:
            print(f"Warning: Replay range from {row['Replay_Copy']} to {end_idx} is empty for constraint '{constraint}'. Skipping assignment.")
    return df

def check_constraints(constraints):
    if constraints is None:
        print('Provide constraints')
        import sys
        sys.exit()

if __name__ == "__main__":
    if dataset == 'BATADAL':
        list_of_constraints = ['PLC_1', 'PLC_2', 'PLC_3', 'PLC_4', 'PLC_5', 'PLC_6', 'PLC_7', 'PLC_8', 'PLC_9'] if constraints_setting == 'topology' else [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
    elif dataset == 'WADI':
        list_of_constraints = ['PLC_1', 'PLC_2'] if constraints_setting == 'topology' else [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]

    data_folder = f'../../Data/{dataset}'
    for i in list_of_constraints:
        test_data = pd.read_csv(f'{data_folder}/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
        test_data.set_index('DATETIME', inplace=True)

        eavesdropped_data = pd.read_csv(f'{data_folder}/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
        eavesdropped_data.set_index('DATETIME', inplace=True)

        constraints = []
        attack_intervals = identify_attacks(test_data)

        if dataset == 'BATADAL': 
            for att_num in range(1, 8): 
                s = open(f'../{"Black_Box_Attack" if constraints_setting == "topology" else "Whitebox_Attack"}/constraints/{dataset}/{"constraint_PLC" if constraints_setting == "topology" else f"constraint_variables_attack_{att_num}"}.txt').read()
                dictionary = eval(s)
                constraints.append(dictionary[i])

                test_data = pd.read_csv(f'../../Data/BATADAL/attack_{att_num}_from_test_dataset.csv').drop(columns=['Unnamed: 0'], axis=1)
                test_data.set_index('DATETIME', inplace=True)

                spoofed_data = spoof(constrained_replay, attack_intervals, eavesdropped_data, test_data, att_num, constraints)

                # Check if spoofed_data is empty before saving
                output_path = f'./results/BATADAL/{"constrained_PLC/constrained_" + str(i) + "_attack_" + str(att_num) if constraints_setting == "topology" else f"attack_{att_num}_replay_max_{i}"}.csv'
                if not spoofed_data.empty:
                    spoofed_data.to_csv(output_path)
                else:
                    print(f"Warning: N
