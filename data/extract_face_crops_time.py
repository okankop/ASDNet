import glob
import os
import numpy as np
import pandas as pd
import cv2
import random


def generate_mini_dataset(video_dir, output_dir, df, balanced=False):
    # Assumes there is always more negatives than positives.
    df_neg = df[df['label_id'] == 0]
    df_pos = df[df['label_id'] == 1]
    instances_neg = df_neg['instance_id'].unique().tolist()
    instances_pos = df_pos['instance_id'].unique().tolist()

    if balanced:
        random.seed(17)
        instances_neg = random.sample(instances_neg, len(instances_pos))
        df_neg = df_neg[df['instance_id'].isin(instances_neg)]

    print(len(instances_pos), len(instances_neg))
    print(len(df_pos), len(df_neg))
    balanced_df = pd.concat([df_pos, df_neg]).reset_index(drop=True)
    balanced_df = balanced_df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
    entity_list = balanced_df['entity_id'].unique().tolist()
    balanced_gb = balanced_df.groupby('entity_id')

    # Make sure directory exists
    for l in balanced_df['label'].unique().tolist():
        d = os.path.join(output_dir, l)
        if not os.path.isdir(d):
            os.makedirs(d)

    for entity_idx, instance in enumerate(entity_list):
        instance_data = balanced_gb.get_group(instance)

        video_key = instance_data.iloc[0]['video_id']
        entity_id = instance_data.iloc[0]['entity_id']
        video_file = glob.glob(os.path.join(video_dir, '{}.*'.format(video_key)))[0]

        V = cv2.VideoCapture(video_file)

        # Make sure directory exists
        instance_dir = os.path.join(os.path.join(output_dir, entity_id))
        if not os.path.isdir(instance_dir):
            os.makedirs(instance_dir)

        j = 0
        for _, row in instance_data.iterrows():
            image_filename = os.path.join(instance_dir, str(row['frame_timestamp'])+'.jpg')
            if os.path.exists(image_filename):
                print('skip', image_filename)
                continue

            V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)

            # Load frame and get dimensions
            _, frame = V.read()
            h = np.size(frame, 0)
            w = np.size(frame, 1)

            # Crop face
            crop_x1 = int(row['entity_box_x1'] * w)
            crop_y1 = int(row['entity_box_y1'] * h)
            crop_x2 = int(row['entity_box_x2'] * w)
            crop_y2 = int(row['entity_box_y2'] * h)
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2, :]

            j = j+1

            cv2.imwrite(image_filename, face_crop)


if __name__ == '__main__':
    ava_video_dir = '/usr/home/kop/datasets/AVA_ActiveSpeaker/test_videos'
    output_dir = '/usr/home/kop/datasets/AVA_ActiveSpeaker/instance_crops_time_test'
    csv_file = '/usr/home/kop/active-speakers-context/ava_activespeaker_test_augmented.csv'

    df = pd.read_csv(csv_file, engine='python')
    train_subset_dir = os.path.join(output_dir, 'test')
    generate_mini_dataset(ava_video_dir, train_subset_dir, df, balanced=False)

    print(':::ALL DONE:::')
