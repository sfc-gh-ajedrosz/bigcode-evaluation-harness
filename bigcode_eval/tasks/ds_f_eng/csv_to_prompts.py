import os.path
import shutil
import subprocess
from pathlib import Path
import json
import os

import jsonargparse
import pandas as pd

from bigcode_eval.tasks.custom_metrics.kernels import kernels

TASK_HEAD = "You will be shown a dataset from a Kaggle competition. The competition is to build a classifier or regressor over this dataset such that it has the highest accuracy possible. Your task is to preprocess this dataset so that an existing model can improve its accuracy. You will be shown a general description of the dataset including a description of the task. This will be followed by a list of features and their descriptions. These features are available in a pandas DataFrame. You should output a single function in Python 3.7+ that will take this dataframe as input and return a modified dataframe with new features. New features can be formed by e.g. keeping the old features unchanged, creating interactions between features, preprocessing features, removing features."
TASK_TAIL = "Now please write a single function in Python 3.7+ which takes dataframes and returns dataframes with new features."
DESCRIPTION_COLUMN = "DESCRIPTION_UPDATED"
FEATURES_COLUMN = "FEATURES_UPDATED"
DATAFRAME_HEADER_COLUMN = "head_n30"
REF_COLUMN = "ref"
SELECTED_COLUMN = "decision"
SELECTED_VALUE = "accept"


def series_to_prompt(series: pd.Series) -> str:
    return f"{TASK_HEAD}\n\nDataset description: {series[DESCRIPTION_COLUMN]}\n\nDataframe features and their descriptions:\n{series[FEATURES_COLUMN]}\n\nDataframe header:\n{series[DATAFRAME_HEADER_COLUMN]}\n\n{TASK_TAIL}"

def series_to_chat(series: pd.Series, df) -> list:
    def _user_turn(_series: pd.Series) -> str:
        return f"Dataset description: {series[DESCRIPTION_COLUMN]}\n\nDataframe features and their descriptions:\n{series[FEATURES_COLUMN]}\n\nDataframe header:\n{series[DATAFRAME_HEADER_COLUMN]}\n\n{TASK_TAIL}"

    def _model_turn(code: str) -> str:
        return '```\n' + code.strip() + '\n```'

    keys = list(kernels.keys())
    refs = [k.replace('__', '/').replace('.csv', '') for k in keys]

    return [
       {"role": "system", "text": TASK_HEAD},
       {"role": "user", "text": _user_turn(df[df["ref"] == refs[0]])},
       {"role": "model", "text": _model_turn(kernels[keys[0]])},
       {"role": "user", "text": _user_turn(df[df["ref"] == refs[1]])},
       {"role": "model", "text": _model_turn(kernels[keys[1]])},
       {"role": "user", "text": _user_turn(df[df["ref"] == refs[2]])},
       {"role": "model", "text": _model_turn(kernels[keys[2]])},
       {"role": "user", "text": _user_turn(series)}
    ]

def placeholder_series_to_prompt() -> str:
    return f"{TASK_HEAD}"


def unzip(zip_path, extract_to):
    import zipfile
    import os

    # Specify the ZIP file and the extraction target directory

    # Ensure the target directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Open and extract all files from the ZIP archive
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Files extracted to {extract_to}")

    if len(os.listdir(extract_to)) == 1:
        # just rename
        mv_src = f"{extract_to}/{os.listdir(extract_to)[0]}"
        tgt_src = f"{extract_to}/data.csv"
        shutil.move(mv_src, tgt_src)
    else:
        # choose and rename
        return None


def main(dataframe_path: Path, output_path: Path, run_placeholder: bool) -> None:
    MAX_NUM_ROWS = 120

    print(f"Reading from {dataframe_path}")
    df = pd.read_csv(dataframe_path)

    print(f"Saving to {output_path}")
    with open(output_path, "w") as f_out:
        for _, row in df.iterrows():
            # if row[SELECTED_COLUMN] != SELECTED_VALUE:
            #     continue
            if run_placeholder:
                prompt = placeholder_series_to_prompt()
                proper_path_name = row[REF_COLUMN].replace("/", "__")
                DATASETS_STORED_PATH = '/Users/mpietruszka/Repos/ds-f-eng/auto-feature-engineering/final 8'
                is_dataset_available = os.path.exists(f"{DATASETS_STORED_PATH}/{proper_path_name}/data.csv")
                if is_dataset_available:
                    pass
                else:
                    # download and unzip
                    subprocess.run(f'kaggle datasets download -d {row[REF_COLUMN]}', shell=True)
                    # move from downloaded to the one desired
                    import shutil
                    current_directory = os.getcwd()
                    filename = row[REF_COLUMN].split("/")[-1]
                    source_file = os.path.join(current_directory, filename + '.zip')

                    destination_file = os.path.join(DATASETS_STORED_PATH, filename)

                    shutil.copy(source_file, destination_file + '.zip')
                    extract_to = os.path.join(DATASETS_STORED_PATH,proper_path_name)

                    result = unzip(destination_file + '.zip', extract_to)
                    if result is None:
                        continue
            else:
                prompt = series_to_prompt(row)
                chat = series_to_chat(row, df)
            ref = row[REF_COLUMN]

            csv_name = ref.replace('/', '__') + '.csv'
            if csv_name in kernels.keys():
                print('Skipping ', ref, '(few-shot example)')
                continue

            out_line = json.dumps({'dataframe_id': ref, 'prompt': prompt, 'chat': chat})
            f_out.write(f"{out_line}\n")
            MAX_NUM_ROWS -= 1
            if MAX_NUM_ROWS == 0:
                break


if __name__ == "__main__":
    jsonargparse.CLI(main, as_positional=False)
