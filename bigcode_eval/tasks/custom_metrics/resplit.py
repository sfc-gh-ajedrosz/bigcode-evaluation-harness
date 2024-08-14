import pandas as pd
import os

# Define paths
pth = "/Users/mpietruszka/Repos/aj-bigcode/bigcode_eval/tasks/ds_f_eng/ds_f_eng_dataframes/ds_f_eng_mp_processed/final"
dest_pth = "/Users/mpietruszka/Repos/aj-bigcode/bigcode_eval/tasks/ds_f_eng/ds_f_eng_dataframes/"

# Define constants
TOTAL_MAX_SAMPLES = 1000
MAX_TRAIN_SAMPLES = int(TOTAL_MAX_SAMPLES * 0.9)        # TODO: this should not be limited, as optimistically all models can handle all of the datasets
MAX_TEST_SAMPLES = (TOTAL_MAX_SAMPLES - MAX_TRAIN_SAMPLES) * 10

# Get list of all directories in the path
all_dirs = os.listdir(pth)

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()
logged_sizes = []
# Loop through each directory
for a in all_dirs:
    new = os.path.join(pth, a)
    input_csv_path = os.path.join(new, "data.csv")
    final_csv_path = os.path.join(dest_pth, a + '.csv')

    if os.path.exists(input_csv_path):
        # Read the CSV file
        df = pd.read_csv(input_csv_path)
        num_train_samples = sum(df["ds_f_eng__split"] == "train")
        num_test_samples = sum(df["ds_f_eng__split"] == "test")
        logged_sizes.append((a, num_train_samples, num_test_samples))
        print(logged_sizes[-1])
        # Limit the number of train samples
        if num_train_samples > MAX_TRAIN_SAMPLES:
            train_df = df[df["ds_f_eng__split"] == "train"].sample(MAX_TRAIN_SAMPLES, random_state=42)
        else:
            train_df = df[df["ds_f_eng__split"] == "train"]

        # Limit the number of test samples
        if num_test_samples > MAX_TEST_SAMPLES:
            test_df = df[df["ds_f_eng__split"] == "test"].sample(MAX_TEST_SAMPLES, random_state=42)
        else:
            test_df = df[df["ds_f_eng__split"] == "test"]

        # Concatenate the train and test DataFrames
        final_df = pd.concat([train_df, test_df])

        # Save the final DataFrame to a CSV file
        #final_df.to_csv(final_csv_path, index=False)
        # Append the current DataFrame to the all_data DataFrame
        # all_data = pd.concat([all_data, final_df])

# Save the combined data to a joint CSV file
# joint_csv_path = os.path.join(dest_pth, 'joint_dataset.csv')
# all_data.to_csv(joint_csv_path, index=False)
