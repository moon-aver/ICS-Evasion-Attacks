import pandas as pd
import numpy as np
import argparse

pd.set_option('display.max_columns', 500)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', nargs='+', type=str, default=['BATADAL'])
args = parser.parse_args()
dataset = args.data[0]

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
        if (index - prev_timestamp) > 24:  # Adjust threshold as needed
            count_attacks += 1
            interval = pd.DataFrame([[f'attack_{count_attacks}', start, prev_timestamp,
                                      start - (prev_timestamp - start) - 200]],
                                    columns=['Name', 'Start', 'End', 'Replay_Copy'])
            attack_intervals = pd.concat([attack_intervals, interval], ignore_index=True)
            start = index
        prev_timestamp = index

    count_attacks += 1
    interval = pd.DataFrame([[f'attack_{count_attacks}', start, prev_timestamp,
                              start - (prev_timestamp - start) - 200]],
                            columns=['Name', 'Start', 'End', 'Replay_Copy'])
    attack_intervals = pd.concat([attack_intervals, interval], ignore_index=True)

    print('_________________________________ATTACK INTERVALS___________________________________\n')
    print(attack_intervals)
    print('____________________________________________________________________________________')
    return attack_intervals

def spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, att_num):
    """
    Applies a spoofing technique to the test_data based on attack intervals and returns spoofed data.
    """
    if att_num > len(attack_intervals):
        print(f"Warning: Attack number {att_num} is out of bounds for available intervals.")
        return test_data

    row = attack_intervals.iloc[att_num - 1]
    df = pd.DataFrame(columns=test_data.columns)
    df = spoofing_technique(df, row, eavesdropped_data, attack_intervals)
    df['ATT_FLAG'] = 1
    return df

def unconstrained_replay(df, row, eavesdropped_data, attack_intervals):
    """
    Unconstrained version of the replay attack to copy all columns.
    """
    start_idx = int(row['Replay_Copy'])
    end_idx = start_idx + int(row['End'] - row['Start']) + 1
    replay_range = eavesdropped_data.iloc[start_idx:end_idx]
    df = pd.concat([df, replay_range], ignore_index=True)
    return df

if __name__ == "__main__":
    data_folder = f'../../Data/{dataset}'
    test_data = pd.read_csv(f'{data_folder}/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
    test_data.set_index('DATETIME', inplace=True)
    eavesdropped_data = pd.read_csv(f'{data_folder}/test_dataset_1.csv').drop(columns=['Unnamed: 0'], axis=1)
    eavesdropped_data.set_index('DATETIME', inplace=True)

    attack_intervals = identify_attacks(test_data)

    for att_num in range(1, 8):
        test_data = pd.read_csv(f'../../Data/BATADAL/attack_{att_num}_from_test_dataset.csv').drop(columns=['Unnamed: 0'], axis=1)
        test_data.set_index('DATETIME', inplace=True)

        spoofed_data = spoof(unconstrained_replay, attack_intervals, eavesdropped_data, test_data, att_num)
        output_path = f'./results/BATADAL/attack_{att_num}_replay_max_all.csv'
        spoofed_data.to_csv(output_path)
