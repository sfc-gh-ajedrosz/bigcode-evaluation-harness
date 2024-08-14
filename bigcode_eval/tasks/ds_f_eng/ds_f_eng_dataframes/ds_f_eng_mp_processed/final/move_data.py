pth = "/Users/mpietruszka/Repos/aj-bigcode/bigcode_eval/tasks/ds_f_eng/ds_f_eng_dataframes/ds_f_eng_mp_processed/final"
dest_pth = "/Users/mpietruszka/Repos/aj-bigcode/bigcode_eval/tasks/ds_f_eng/ds_f_eng_dataframes/"

import os
import shutil
import copy

all_dirs = os.listdir(pth)
for a in all_dirs:
    new = os.path.join(pth, a)
    input_csv_path = os.path.join(new, "data.csv")
    final_csv_path = os.path.join(dest_pth, a + '.csv')
    if os.path.exists(input_csv_path):
        shutil.copy(input_csv_path, final_csv_path)
    #os.path.cp(input_csv_path, final_csv_path)


