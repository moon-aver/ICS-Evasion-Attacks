import pandas as pd
import numpy as np
import datetime
pd.set_option('display.max_columns', 500)



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
    prev_datetime = attacks.index[0]  # find first timing
    start = prev_datetime
    count_attacks = 0

    # find attacks bounds
    attack_intervals = pd.DataFrame(
        columns=['Name', 'Start', 'End', 'Replay_Copy'])
    for index, _ in attacks.iterrows():
        if count_attacks == 3:
            count_attacks = count_attacks + 1
        if (index - prev_datetime > 1):
            count_attacks = count_attacks + 1
            interval = pd.DataFrame([['attack_'+str(count_attacks), start, prev_datetime, (start - (prev_datetime - start)) - 200]]
                                    , columns=['Name', 'Start', 'End', 'Replay_Copy'], index = [count_attacks])
            pd.concat([attack_intervals,interval])
            start = index
        prev_datetime = index
    count_attacks = count_attacks + 1
    interval = pd.DataFrame([['attack_'+str(count_attacks), start, prev_datetime, start - (
        prev_datetime - start) - 200]]
                            , columns=['Name', 'Start', 'End', 'Replay_Copy'], index = [count_attacks])
    pd.concat([attack_intervals,interval])

    print('_________________________________ATTACK INTERVALS___________________________________\n')
    print(attack_intervals)
    print('____________________________________________________________________________________')
    return attack_intervals


def spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data,att_num, constraints=None ):
    
    """
    
    given a spoofing_technique to be applied, the attack_intervals, eavesdropped_data and test_data, it builds the dataset containing sensor spoofing.
    
    Returns
    -------
    DataFrame
        Dataset with spoofed sensor readings.
    """
    df2 = pd.DataFrame()
    if dataset == 'WADI':
        if att_num < 5:
            row = attack_intervals.iloc[att_num-1]
        else:
            row = attack_intervals.iloc[att_num-2]
    if dataset == 'BATADAL':
        if att_num < 8:
            row = attack_intervals.iloc[att_num-1]
        else:
            row = attack_intervals.iloc[att_num-8]
    df = pd.DataFrame(columns=test_data.columns)
    df = spoofing_technique(df,row, eavesdropped_data, attack_intervals, constraints)
    df.ATT_FLAG = 1
    
    return df


def replay(df, row, eavesdropped_data, attack_intervals, *args):
    """
    
    applies replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    df.concat([df,eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy']+(
        row['End']-(row['Start']))]])[test_data.columns.tolist()]  # append replayed row    
    return df



def constrained_replay(df, row, eavesdropped_data, attack_intervals, *args):
    
    """
    constrained version of the replay attack
    """

    constraints = args[0]
    check_constraints(constraints)
    print(len(eavesdropped_data[constraints[0]].loc[row['Replay_Copy']
        :row['Replay_Copy']+(row['End']-(row['Start']))+1]))
    print(len(test_data))
    try:
        test_data[constraints[0]] = eavesdropped_data[constraints[0]].loc[row['Replay_Copy']
            :row['Replay_Copy']+(row['End']-(row['Start']))+1].values
    except:
        try:
            test_data[constraints[0]] = eavesdropped_data[constraints[0]].loc[row['Replay_Copy']
                :row['Replay_Copy']+(row['End']-(row['Start']))].values
        except:
            test_data[constraints[0]] = eavesdropped_data[constraints[0]].loc[row['Replay_Copy']
                :row['Replay_Copy']+(row['End']-(row['Start']))-1].values
    return test_data

def check_constraints(constraints):
    if constraints == None:
        print('Provide constraints')
        import sys
        sys.exit()
    else:
        pass

def parse_datetime_column(data_frame, column_name='DATETIME'):
    # 날짜 형식을 검사하여 적절한 처리를 선택
    if pd.api.types.is_numeric_dtype(data_frame[column_name]):
        # 숫자 형식이면 기준 시간에 시간을 더한다고 가정 (예: 분 단위)
        start_time = datetime.datetime(2024, 1, 1)  # 기준 시간 설정
        data_frame[column_name] = data_frame[column_name].apply(lambda x: start_time + datetime.timedelta(minutes=x))
    else:
        try:
            # 시도: 표준 날짜/시간 형식으로 파싱
            data_frame[column_name] = pd.to_datetime(data_frame[column_name])
        except ValueError:
            # 파싱 실패 시 경고 출력 및 원본 데이터 유지
            print("Warning: DATETIME column could not be parsed as standard date format. It will be used as is.")
    return data_frame

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
            list_of_constraints =  ['PLC_1', 'PLC_2','PLC_3','PLC_4','PLC_5','PLC_6','PLC_7','PLC_8','PLC_9']
        else:
            list_of_constraints = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
    if dataset == 'WADI':
        if constraints_setting == 'topology':
            list_of_constraints =  ['PLC_1', 'PLC_2']
        else:
            list_of_constraints = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 60, 70, 80]
            
    data_folder = '../../Data/'+dataset

    for i in list_of_constraints:
        if dataset == 'BATADAL':
            test_data = pd.read_csv(data_folder+'/test_dataset_1.csv')
            eavesdropped_data = pd.read_csv(data_folder+"/test_dataset_1.csv")
            test_data = test_data.drop(columns=['Unnamed: 0'], axis=1)
            eavesdropped_data = eavesdropped_data.drop(columns=['Unnamed: 0'], axis=1)
        if dataset == 'WADI':
            test_data = pd.read_csv(data_folder+'/attacks_october_clean_with_label.csv')
            eavesdropped_data = pd.read_csv(data_folder+"/train_dataset.csv")

        constraints = pd.DataFrame()

        actuator_columns = eavesdropped_data.filter(
            regex=("STATUS")).columns.tolist()

        spoofing_technique = constrained_replay
        attack_intervals = identify_attacks(test_data)

        if dataset == 'BATADAL': 
            for att_num in [1,2,3,4,5,6,7]: 
                if constraints_setting == 'topology':
                    s = open('../Black_Box_Attack/constraints/'+dataset+'/constraint_PLC.txt', 'r').read()
                else:
                    s = open('../Whitebox_Attack/constraints/'+dataset+'/constraint_variables_attack_'+str(att_num)+'.txt', 'r').read()
                dictionary =  eval(s)
                print(dictionary)
                constraints = pd.concat([constraints, pd.DataFrame([dictionary[i]])], ignore_index=True)
                
                print(constraints)

                print('ATT Num: '+str(att_num))
                # test_data =  pd.read_csv('../../Data/BATADAL/attack_'+str(att_num)+'_from_test_dataset.csv', index_col=['DATETIME'], parse_dates=True)
                # CSV 파일 읽기 (날짜 파싱 없이)
                test_data = pd.read_csv('../../Data/BATADAL/attack_'+str(att_num)+'_from_test_dataset.csv')                
                # DATETIME 열 파싱
                test_data = parse_datetime_column(test_data)
                test_data = test_data.drop(columns=['Unnamed: 0'], axis=1, errors='ignore')
                
                # attack_intervals의 범위 내에 있는지 검증
                if att_num-1 < len(attack_intervals):
                    # 범위 내에서 spoof 함수 호출 및 결과 저장
                    spoofed_data = spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, att_num, constraints)
                    if constraints_setting == 'topology':
                        spoofed_data.to_csv('./results/BATADAL/constrained_PLC/constrained_'+str(i)+'_attack_'+str(att_num)+'.csv')
                    else:
                        spoofed_data.to_csv('./results/BATADAL/attack_'+str(att_num)+'_replay_max_'+str(i)+'.csv')
                else:
                    # 범위를 벗어날 경우 경고 메시지 출력
                    print(f"Warning: Attack number {att_num} is out of bounds. No data processed for att_num {att_num}.")

                test_data = test_data.drop(columns=['Unnamed: 0'], axis=1, errors='ignore')
                spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data, att_num, constraints )
                if constraints_setting == 'topology':
                    spoofed_data.to_csv('./results/BATADAL/constrained_PLC/constrained_'+str(i)+'_attack_'+str(att_num)+'.csv')
                else:
                    spoofed_data.to_csv('./results/BATADAL/attack_'+str(att_num)+'_replay_max_'+str(i)+'.csv')
                
            test_data = pd.read_csv('../../Data/BATADAL/test_dataset_2.csv')#, parse_dates=True)  # , dayfirst=True)
            eavesdropped_data = pd.read_csv("../../Data/BATADAL/test_dataset_2.csv")#, parse_dates=True)  # ,  dayfirst=True)
            
            test_data = test_data.drop(columns=['Unnamed: 0'], axis=1)
            eavesdropped_data = eavesdropped_data.drop(columns=['Unnamed: 0'], axis=1)
            constraints=[]

            actuator_columns = test_data.filter(
                regex=("STATUS")).columns.tolist()

            spoofing_technique = constrained_replay
            attack_intervals = identify_attacks(test_data)
                
            for att_num in [8,9,10,11,12,13,14]: 
                if constraints_setting == 'topology':
                    s = open('../Black_Box_Attack/constraints/'+dataset+'/constraint_PLC.txt', 'r').read()
                else:
                    s = open('../Whitebox_Attack/constraints/'+dataset+'/constraint_variables_attack_'+str(att_num)+'.txt', 'r').read()
                dictionary =  eval(s)
                constraints = pd.concat([constraints, pd.DataFrame([dictionary[i]])], ignore_index=True)
                
                dictionary =  eval(s)
                constraints = pd.concat([constraints, pd.DataFrame([dictionary[i]])], ignore_index=True)
                print('ATT Num: '+str(att_num))
                test_data =  pd.read_csv('../../Data/BATADAL/attack_'+str(att_num)+'_from_test_dataset.csv', index_col=['DATETIME'], parse_dates=True)
                test_data = test_data.drop(columns=['Unnamed: 0'], axis=1)
                spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data, att_num, constraints )
                if constraints_setting == 'topology':
                    spoofed_data.to_csv('./results/BATADAL/constrained_PLC/constrained_'+str(i)+'_attack_'+str(att_num)+'.csv')
                else:
                    spoofed_data.to_csv('./results/BATADAL/attack_'+str(att_num)+'_replay_max_'+str(i)+'.csv')
        if dataset == 'WADI':
              for att_num in [1,2,3,5,6,7,8,9,10,11,12,13,14,15]: #
                if constraints_setting == 'topology':
                    s = open('../Black_Box_Attack/constraints/'+dataset+'/constraint_PLC.txt', 'r').read()
                else:
                    s = open('../Whitebox_Attack/constraints/'+dataset+'/constraint_variables_attack_'+str(att_num)+'.txt', 'r').read()
                dictionary =  eval(s)
                #print(dictionary)
                constraints = pd.concat([constraints, pd.DataFrame([dictionary[i]])], ignore_index=True)
                
                #print(constraints)

                print('ATT Num: '+str(att_num))
                test_data =  pd.read_csv('../../Data/'+dataset+'/attack_'+str(att_num)+'_from_test_dataset.csv', index_col=['DATETIME'], parse_dates=True)
                #test_data = test_data.drop(columns=['Unnamed: 0'], axis=1)
                spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data, att_num, constraints )
                if constraints_setting == 'topology':
                    spoofed_data.to_csv('./results/'+dataset+'/constrained_PLC/constrained_'+str(i)+'_attack_'+str(att_num)+'.csv')
                else:
                    spoofed_data.to_csv('./results/'+dataset+'/attack_'+str(att_num)+'_replay_max_'+str(i)+'.csv')
