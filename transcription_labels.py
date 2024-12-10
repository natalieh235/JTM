
import pandas as pd
import os
# Paths to the train and test metadata files
data_dir = "./data/small_transcription"
train_metadata_path = data_dir + "/train_small.csv"
test_metadata_path = data_dir + "/test_small.csv"
output_metadata_path = data_dir + "/traintest_musicnet_metadata.csv"

all_metadata = data_dir + "/musicnet_metadata.csv"
all_metadata_pd = pd.read_csv(all_metadata)

piano_metadata = all_metadata_pd[all_metadata_pd['ensemble'] == "Solo Piano"]
piano_metadata.to_csv(f'{data_dir}/piano_music_metadata.csv', index=False)


if not os.path.exists(output_metadata_path):
    # Load train and test metadata
    train_metadata = pd.read_csv(train_metadata_path)
    test_metadata = pd.read_csv(test_metadata_path)

    # Combine train and test metadata
    combined_metadata = pd.concat([train_metadata, test_metadata], ignore_index=True)

    # Save the combined metadata to a new file
    combined_metadata.to_csv(output_metadata_path, index=False)

    print(f"Combined metadata saved to {output_metadata_path}")

frames = []
meta = pd.read_csv(r"./data/musicnet_metadata.csv")
length_meta = pd.read_csv(test_metadata_path)

# print(length_meta)
csv_paths = "../musicnet/musicnet/train_labels"
for file in os.listdir(csv_paths):
    # print('file', file)
    file_id = int(file.split(".")[0])
    if file.split(".")[1] == 'csv':
        if meta[meta['id'] == file_id]['ensemble'].values[0] == "Solo Piano":
            df = pd.read_csv(f'{csv_paths}/{file}')
        else:
            continue

        # print(length_meta[length_meta['id'] == file_id])
        if length_meta[length_meta['id'] == file_id].empty:
          # print('not in train file')
          continue

        # print(df)
        print('made it', file)
        df['id'] = file.split(".")[0]
        df['length'] = length_meta[length_meta['id'] == int(file.split(".")[0])]['length'].values[0]
        frames.append(df)

master_df = pd.concat(frames)

master_df['start_time'] = master_df['start_time'] * 160 / 441

master_df['end_time'] = master_df['end_time'] * 160 / 441

master_df['id'] = master_df['id'].astype('category')
# master_df['inst'] = master_df['instrument'].astype('category')
master_df['id_cat'] = master_df['id'].cat.codes
# master_df['instrument_cat'] = master_df['instrument'].cat.codes
master_df.to_csv(f'{data_dir}/metadata_transcript_test.csv', index=False)