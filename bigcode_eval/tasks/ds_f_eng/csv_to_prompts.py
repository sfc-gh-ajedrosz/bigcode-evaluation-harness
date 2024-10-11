import os.path
import shutil
import subprocess
from pathlib import Path
from collections import Counter
import json
from copy import deepcopy
import os

import jsonargparse
import pandas as pd

from bigcode_eval.tasks.custom_metrics.kernels import kernels
from bigcode_eval.tasks.custom_metrics.ds_f_eng_eval_3_refactored import DataProcessor

TASK_HEAD = '''You will be shown a dataset from a Kaggle competition. Your task is to preprocess this dataset so that it can improve the accuracy of an existing XGBoost model. After an example (denoted by ### Example), you will be provided with a problem to Solve (### Problem).

Both consist of:
- A general description of the dataset and the task.
- A list of features, their descriptions, and sample values. These features are in a pandas DataFrame.

Your task is to generate a single function in Python 3.7+ using pandas 2.1.3 and scikit-learn 1.5.2. This function will:
- Take the dataset and return a modified dataset with transformed and new features.
- Keep original features unchanged while adding new features through interactions, transformations, or preprocessing.
- Handle missing values, encode categorical variables, and scale/normalize features if necessary.
- Remove irrelevant or redundant features if needed.
- Importantly, use world knowledge to add new features, e.g., define dictionaries that map existing features to the new ones.

The function must follow this signature:
```python
def transform(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame) -> (pd.DataFrame, pd.Series, pd.DataFrame):
    """
        Transforms the train and test datasets to improve model accuracy by applying preprocessing and feature engineering.

    Parameters:
    - train_x: DataFrame with training features.
    - train_y: Series with training target labels.
    - test_x: DataFrame with test features.

    Returns:
    - Transformed train_x, train_y, and test_x DataFrames.
    """
```

Remember to not change or encode values in train_y. This parameter is provided only for subsampling consistant with train_x.'''

EXAMPLE_PREFIX =  "### Example\n"

ANSWER_PREFIX = '### Problem\n'

ANSWER_SUFFIX = '''

Now write the code wrapped between ```python and ```.

'''

EXAMPLE_INTERFIX = '''This code was written by an expert data scientist working to improve predictions.

''' # Code:

## NEW

# EXAMPLE_PREFIX =  ""
# ANSWER_PREFIX = ''
# ANSWER_SUFFIX = ''
# EXAMPLE_SUFFIX = '''

# Now write the code wrapped between ```python and ```.'''

# ANSWER_SUFFIX = EXAMPLE_SUFFIX
# EXAMPLE_INTERFIX = ''


# TASK_TAIL = "Write your complete function implementation here, including any necessary imports."
TASK_TAIL = "" # Python function:" # Now write a python function wrapped between ```python and ```."

DESCRIPTION_COLUMN = "DESCRIPTION_UPDATED"
FEATURES_COLUMN = "FEATURES_UPDATED"
DATAFRAME_HEADER_COLUMN = "head_n300"
REF_COLUMN = "ref"
SELECTED_COLUMN = "decision"
SELECTED_VALUE = "accept"

def series_to_prompt(series: pd.Series) -> str:
    return f"{TASK_HEAD}Dataset description: {series[DESCRIPTION_COLUMN]}\n\nDataframe features and their descriptions:\n{series[FEATURES_COLUMN]}\n\nDataframe header:\n{series[DATAFRAME_HEADER_COLUMN]}{TASK_TAIL}"

N = 10

def series_to_chat(series: pd.Series, df, train_x) -> list:
    def _user_turn(_series: pd.Series, data) -> str:

        def _ensure(o):
            try:
                o = eval(o)
            except:
                o = json.loads(o)
            return o

        def _cast(o):
            s = _cast_type(o)
            s = _ensure(s)

            if isinstance(s, list):
                s = s[:N]

            # s = '\n\n'.join([_listize(x) for x in s])

            s = json.dumps(s)
            s = f'```json\n{s}\n```'
            return s

        def _cast_type(o):
            try:
                return o.values[0]
            except:
                return o
        
        if data is None:
            
            csv_name = _series.ref.values[0].replace('/', '__') + '.csv'
            dp = DataProcessor(
                Path('/Users/lborchmann/Documents/bench/bigcode-evaluation-harness/final_13'),
                timeout=600, allow_no_transform=True, target_column=_series.target_col.values[0]
            )
            data = dp.load_dataframe(csv_name, next(_series.iterrows())[1])

        train_data = data.train_x
        train_data_y = data.train_target


        # New    
        descriptions = _ensure(_cast_type(_series[FEATURES_COLUMN]))
        # descriptions = {k.replace('Consumer Rating', 'ConsumerRating').replace('Used/New', 'Used\\/New').replace('Consumer Reviews', 'ConsumerReviews').replace('Check-in service', 'Checkin service').replace('Departure/Arrival', 'Departure\\/Arrival'): v for k, v in descriptions.items()}
        keys = sorted(descriptions.keys())
        header = _ensure(_cast_type(_series[DATAFRAME_HEADER_COLUMN]))

        assert len(set(keys) - set(header[0].keys())) == 0

        def _render(v):
            if isinstance(v, str):
                v = f'"{v}"'
            elif isinstance(v, int):
                return str(v)
            elif isinstance(v, float):
                return str(round(v, 4))
            elif v is None:
                return 'nan'
            return str(v)

        for k in keys:
            if k in header[0]:
                if k in train_data:
                    all_values = train_data[k].values.tolist()
                    dtype = train_data[k].dtype
                elif k == train_data_y.name:
                    all_values = train_data_y.values.tolist()
                    dtype = train_data_y.dtype
                else:
                    all_values = [header[x][k] for x in range(len(header))]
                
                frequent = [v for v, _ in Counter(all_values).most_common()]
                descriptions[k] = descriptions[k].strip().rstrip('.') + f'. Most frequent values ({dtype}): [' + ', '.join([_render(frequent[x]) for x in range(min(len(frequent), N))]) + ']'
        
        features = '\n'.join([f'- *{k}*: {v}' for k, v in {k: descriptions[k] for k in keys}.items()])
        # features = f'```json\n{json.dumps({k: descriptions[k] for k in keys})}\n```'

        return f"Dataset description: {_cast_type(_series[DESCRIPTION_COLUMN])}\n\nDataframe features and their descriptions:\n{features}"

        # return f"Dataset description:\n{_cast_type(_series[DESCRIPTION_COLUMN])}\n\nDataframe features and their descriptions:\n{_cast(_series[FEATURES_COLUMN])}\n\nDataframe header:\n{_cast(_series[DATAFRAME_HEADER_COLUMN])}\n\n" #  {TASK_TAIL}"

    def _model_turn(code: str) -> str:
        # return code.strip()
        return '```python\n' + code.strip() + '\n```'

    keys = list(kernels.keys())
    refs = [k.replace('__', '/').replace('.csv', '') for k in keys]

    # import pdb
    # pdb.set_trace()

    return [
        {"role": "system", "text": TASK_HEAD},
        #    {"role": "user", "text": EXAMPLE_PREFIX + _user_turn(df[df["ref"] == refs[0]], None) + EXAMPLE_SUFFIX},
        #    {"role": "model", "text": ANSWER_PREFIX + _model_turn(kernels[keys[0]]) + ANSWER_SUFFIX},
        #    {"role": "user", "text": EXAMPLE_PREFIX + _user_turn(df[df["ref"] == refs[2]], None) + EXAMPLE_SUFFIX},
        #    {"role": "model", "text": ANSWER_PREFIX + _model_turn(kernels[keys[2]]) + ANSWER_SUFFIX},

    #    {"role": "user", "text": EXAMPLE_PREFIX + _user_turn(df[df["ref"] == refs[1]], None)},
    #    {"role": "model", "text": EXAMPLE_INTERFIX + _model_turn(kernels[keys[1]])},
    #    {"role": "user", "text": ANSWER_PREFIX + _user_turn(series, train_x) + ANSWER_SUFFIX}

       {"role": "user", "text": EXAMPLE_PREFIX + _user_turn(df[df["ref"] == refs[1]], None) + '\n\n' + EXAMPLE_INTERFIX + _model_turn(kernels[keys[1]]) + '\n\n' + ANSWER_PREFIX + _user_turn(series, train_x) + ANSWER_SUFFIX.rstrip()}

    #    {"role": "user", "text": EXAMPLE_PREFIX + _user_turn(df[df["ref"] == refs[1]], None) + EXAMPLE_INTERFIX + _model_turn(kernels[keys[1]]) + '\n\n' + ANSWER_PREFIX + _user_turn(series, train_x) + ANSWER_SUFFIX.rstrip()}
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
    df = pd.read_csv(dataframe_path, encoding='utf-8')

    print(f"Saving to {output_path}")
    with open(output_path, "w") as f_out:
        for _, row in df.iterrows():
            # if row[SELECTED_COLUMN] != SELECTED_VALUE:
            #     continue
            ref = row[REF_COLUMN]
            csv_name = ref.replace('/', '__') + '.csv'

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

                dp = DataProcessor(
                    Path('/Users/lborchmann/Documents/bench/bigcode-evaluation-harness/final_12'),
                    timeout=600, allow_no_transform=True, target_column=row.target_col
                )

                data = dp.load_dataframe(csv_name, row)
                chat = series_to_chat(row, df, data)


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
