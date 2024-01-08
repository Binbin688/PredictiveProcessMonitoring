import csv
import re

import numpy as np

# data_name_list = ['BPIC12', 'BPIC12_W_', 'BPIC12_WC', 'BPIC13_I', 'BPIC13_P', 'BPIC20_D', 'BPIC20_Pe', 'BPIC20_Pr',
#                   'BPIC20_I', 'BPIC20_R', 'Helpdesk', 'Receipt']
data_name_list = ['Receipt']
write_add = 'f_result_dual_view.csv'
with open(write_add, 'a', encoding='utf-8-sig') as f:
    f.write(f'Data address,Accuracy,Precision,Recall,F-score,AUC\n')
f.close()
for data_name in data_name_list:
    data_set_add = data_name + '.csv'
    result_add = 'result_' + data_name + '_dual_view.csv'
    acc_list, pr_list, re_list, f_list, auc_list = [], [], [], [], []
    with open(result_add, 'r', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(re.findall(r'\d+', str(row))) != 0:
                metrics = re.findall(r'\d+.\d+', str(row))
                acc_list.append(float(metrics[1]))
                pr_list.append(float(metrics[2]))
                re_list.append(float(metrics[3]))
                f_list.append(float(metrics[4]))
                auc_list.append(float(metrics[5]))
    f.close()

    print(f'Data address: {data_set_add}, Accuracy: {sum(acc_list) / len(acc_list)}, '
          f'Precision {sum(pr_list) / len(pr_list)}, Recall: {sum(re_list) / len(re_list)}, '
          f'F-score: {sum(f_list) / len(f_list)}, AUC: {sum(auc_list) / len(auc_list)}\n')

    with open(write_add, 'a', encoding='utf-8-sig') as f:
        f.write(f'{data_set_add},'
                f'{round((sum(acc_list) / len(acc_list)), 4)}±{round(np.std(acc_list), 4)},'
                f'{round((sum(pr_list) / len(pr_list)), 4)}±{round(np.std(pr_list), 4)},'
                f'{round((sum(re_list) / len(re_list)), 4)}±{round(np.std(re_list), 4)},'
                f'{round((sum(f_list) / len(f_list)), 4)}±{round(np.std(f_list), 4)}\n')
    f.close()
