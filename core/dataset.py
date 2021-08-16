import os
import math
import glob
import time
import random
import torch

from PIL import Image
from torch.utils import data
from torchvision.transforms import RandomCrop, ColorJitter

import numpy as np
import core.io as io
import core.clip_utils as cu
import multiprocessing as mp

class AV_Enc_BaseDataset(data.Dataset):
    def __init__(self):
        # Cached data
        self.entity_data = {}
        self.speech_data = {}
        self.entity_list = []

        #Reproducibilty
        random.seed(42)
        np.random.seed(0)

    def _postprocess_speech_label(self, speech_label):
        speech_label = int(speech_label)
        if speech_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
            speech_label = 0
        return speech_label

    def _postprocess_entity_label(self, entity_label):
        entity_label = int(entity_label)
        if entity_label == 2:  # Remember 2 = SPEAKING_NOT_AUDIBLE
            entity_label = 0
        return entity_label

    def _cache_entity_data(self, csv_file_path):
        entity_set = set()

        csv_data = io.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            entity_id = csv_row[-3]
            timestamp = csv_row[1]

            speech_label = self._postprocess_speech_label(csv_row[-2])
            entity_label = self._postprocess_entity_label(csv_row[-2])
            minimal_entity_data = (entity_id, timestamp, entity_label)

            # Store minimal entity data
            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}
            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
                entity_set.add((video_id, entity_id))
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

            #Store speech meta-data
            if video_id not in self.speech_data.keys():
                self.speech_data[video_id] = {}
            if timestamp not in self.speech_data[video_id].keys():
                self.speech_data[video_id][timestamp] = speech_label

            #max operation yields if someone is speaking.
            new_speech_label = max(self.speech_data[video_id][timestamp], speech_label)
            self.speech_data[video_id][timestamp] = new_speech_label

        return entity_set

    def _cache_entity_data_forward(self, csv_file_path, target_video):
        entity_list = list()

        csv_data = io.csv_to_list(csv_file_path)
        csv_data.pop(0)  # CSV header
        for csv_row in csv_data:
            video_id = csv_row[0]
            if video_id != target_video:
                continue

            entity_id = csv_row[-3]
            timestamp = csv_row[1]
            entity_label = self._postprocess_entity_label(csv_row[-2])

            entity_list.append((video_id, entity_id, timestamp))
            minimal_entity_data = (entity_id, timestamp, entity_label) # sfate to ingore label here

            if video_id not in self.entity_data.keys():
                self.entity_data[video_id] = {}

            if entity_id not in self.entity_data[video_id].keys():
                self.entity_data[video_id][entity_id] = []
            self.entity_data[video_id][entity_id].append(minimal_entity_data)

        return entity_list

    def _entity_list_postprocessing(self, entity_set):
        print('Initial', len(entity_set))

        # filter out missing data on disk
        all_disk_data = set(os.listdir(self.video_root))
        for video_id, entity_id in entity_set.copy():
            if entity_id not in all_disk_data:
                entity_set.remove((video_id, entity_id))
        print('Pruned not in disk', len(entity_set))
        self.entity_list = sorted(list(entity_set))


class AV_Enc_Dataset(AV_Enc_BaseDataset):
    def __init__(self, audio_root, video_root, csv_file_path, clip_lenght,
                 target_size, video_transform=None, do_video_augment=False):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment

        # Clip arguments
        self.clip_lenght = clip_lenght
        self.half_clip_length = math.floor(self.clip_lenght/2)
        self.target_size = target_size

        entity_set = self._cache_entity_data(csv_file_path)
        self._entity_list_postprocessing(entity_set)

    def __len__(self):
        return int(len(self.entity_list)/1)

    def __getitem__(self, index):
        #Get meta-data
        video_id, entity_id = self.entity_list[index]
        entity_metadata = self.entity_data[video_id][entity_id]

        audio_offset = float(entity_metadata[0][1])
        mid_index = random.randint(0, len(entity_metadata)-1)
        midone = entity_metadata[mid_index]
        target = int(midone[-1])
        target_audio = self.speech_data[video_id][midone[1]]

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index, self.half_clip_length)
        clip_meta_data = clip_meta_data[:self.clip_lenght]

        video_data, audio_data = io.load_av_clip_from_metadata(clip_meta_data,
                                 self.video_root, self.audio_root, audio_offset,
                                 self.target_size)

        if self.do_video_augment:
            # random flip
            if bool(random.getrandbits(1)):
                video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

            # random color jitter
            color_transforms = ColorJitter.get_params(brightness=[0.6,1.4], contrast=[0.6,1.4], saturation=[0.6,1.4], hue=None)
            video_data = [color_transforms(s) for s in video_data]

            # random crop
            width, height = video_data[0].size
            f = random.uniform(0.5, 1)
            i, j, h, w = RandomCrop.get_params(video_data[0], output_size=(int(height*f), int(width*f)))
            video_data = [s.crop(box=(j, i, w, h)) for s in video_data]

        if self.video_transform is not None:
            video_data = [self.video_transform(vd) for vd in video_data]

        video_data = torch.stack(video_data, 0).permute(1, 0, 2, 3) 
        return (np.float32(audio_data), video_data), target, target_audio


class AV_Enc_Forward_Dataset(AV_Enc_BaseDataset):
    def __init__(self, target_video, audio_root, video_root, csv_file_path, clip_lenght,
                 target_size, video_transform=None, do_video_augment=False):
        super().__init__()

        # Data directories
        self.audio_root = audio_root
        self.video_root = video_root

        # Post-processing
        self.video_transform = video_transform
        self.do_video_augment = do_video_augment
        self.target_video = target_video

        # Clip arguments
        self.clip_lenght = clip_lenght
        self.half_clip_length = math.floor(self.clip_lenght/2)
        self.target_size = target_size

        self.entity_list = self._cache_entity_data_forward(csv_file_path, self.target_video )
        print('len(self.entity_list)', len(self.entity_list))

    def _where_is_ts(self, entity_metadata, ts):
        for idx, val in enumerate(entity_metadata):
            if val[1] == ts:
                return idx

        raise Exception('time stamp not found')

    def __len__(self):
        return int(len(self.entity_list))

    def __getitem__(self, index):
        #Get meta-data
        video_id, entity_id, ts = self.entity_list[index]
        entity_metadata = self.entity_data[video_id][entity_id]

        audio_offset = float(entity_metadata[0][1])
        mid_index = self._where_is_ts(entity_metadata, ts)
        midone = entity_metadata[mid_index]
        gt = midone[-1]

        clip_meta_data = cu.generate_clip_meta(entity_metadata, mid_index, self.half_clip_length)
        clip_meta_data = clip_meta_data[:self.clip_lenght] 

        video_data, audio_data = io.load_av_clip_from_metadata(clip_meta_data,
                                 self.video_root, self.audio_root, audio_offset,
                                 self.target_size)

        if self.do_video_augment:
            # random flip
            if bool(random.getrandbits(1)):
                video_data = [s.transpose(Image.FLIP_LEFT_RIGHT) for s in video_data]

            # random crop
            width, height = video_data[0].size
            f = random.uniform(0.5, 1)
            i, j, h, w = RandomCrop.get_params(video_data[0], output_size=(int(height*f), int(width*f)))
            video_data = [s.crop(box=(j, i, w, h)) for s in video_data]

        if self.video_transform is not None:
            video_data = [self.video_transform(vd) for vd in video_data]

        # video_data = torch.cat(video_data, dim=0)
        video_data = torch.stack(video_data, 0).permute(1, 0, 2, 3) ######################### MODIFICATION
        return np.float32(audio_data), video_data, video_id, ts, entity_id, gt



class TM_ISRM_BaseDataset(data.Dataset):
    def get_speaker_context(self, ts_to_entity, video_id, target_entity_id,
                            center_ts, num_speakers):
        context_entities = list(ts_to_entity[video_id][center_ts])
        random.shuffle(context_entities)
        context_entities.remove(target_entity_id)

        if not context_entities:  # nos mamamos la lista
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < num_speakers:
                context_entities.append("zero_entity")
        elif len(context_entities) < num_speakers:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            while len(context_entities) < num_speakers:
                context_entities.append("zero_entity")
        else:
            context_entities.insert(0, target_entity_id)  # make sure is at 0
            context_entities = context_entities[:num_speakers]

        return context_entities

    def _decode_feature_data_from_csv(self, feature_data):
        feature_data = feature_data[1:-1]
        feature_data = feature_data.split(',')
        return np.asarray([float(fd) for fd in feature_data])

    def get_time_context(self, entity_data, video_id, target_entity_id,
                         center_ts, half_time_length, stride):
        if not target_entity_id == "zero_entity":
            all_ts = list(entity_data[video_id][target_entity_id].keys())
            center_ts_idx = all_ts.index(str(center_ts))

            start = center_ts_idx-(half_time_length*stride)
            end = center_ts_idx+((half_time_length+1)*stride)
            selected_ts_idx = list(range(start, end, stride))
            selected_ts = []
            for idx in selected_ts_idx:
                if idx < 0:
                    idx = 0
                if idx >= len(all_ts):
                    idx = len(all_ts)-1
                selected_ts.append(all_ts[idx])
            selected_ts = selected_ts[:half_time_length*2]
        else:
            selected_ts = list(range(0, half_time_length*2+1)) # Some random timestamps
            selected_ts = selected_ts[:half_time_length*2]

        return selected_ts

    def get_time_indexed_feature(self, video_id, entity_id, selectd_ts):
        time_features = []
        if not entity_id == "zero_entity":
            for ts in selectd_ts:
                time_features.append(self.entity_data[video_id][entity_id][ts][0])
        else:
            for ts in selectd_ts:
                time_features.append(np.zeros(672))

        return np.asarray(time_features)

    def _cache_feature_file(self, csv_file):
        entity_data = {}
        feature_list = []
        ts_to_entity = {}

        print('load feature data', csv_file)
        csv_data = io.csv_to_list(csv_file)
        for csv_row in csv_data:
            video_id = csv_row[0]
            ts = csv_row[1]
            entity_id = csv_row[2]
            features = self._decode_feature_data_from_csv(csv_row[-1])
            label = int(float(csv_row[3]))

            # entity_data
            if video_id not in entity_data.keys():
                entity_data[video_id] = {}
            if entity_id not in entity_data[video_id].keys():
                entity_data[video_id][entity_id] = {}
            if ts not in entity_data[video_id][entity_id].keys():
                entity_data[video_id][entity_id][ts] = []
            entity_data[video_id][entity_id][ts] = (features, label)
            feature_list.append((video_id, entity_id, ts))

            # ts_to_entity
            if video_id not in ts_to_entity.keys():
                ts_to_entity[video_id] = {}
            if ts not in ts_to_entity[video_id].keys():
                ts_to_entity[video_id][ts] = []
            ts_to_entity[video_id][ts].append(entity_id)

        print('loaded ', len(feature_list), ' features')
        return entity_data, feature_list, ts_to_entity


class TM_ISRM_Dataset(TM_ISRM_BaseDataset):
    def __init__(self, csv_file_path, seq_length, time_stride, num_speakers):
        # Space config
        self.seq_length = seq_length
        self.time_stride = time_stride
        self.num_speakers = num_speakers
        self.half_time_length = math.floor(self.seq_length/2)

        # In memory data
        self.feature_list = []
        self.ts_to_entity = {}
        self.entity_data = {}

        # Load metadata
        self._cache_feature_data(csv_file_path)

    # Parallel load of feature files
    def _cache_feature_data(self, dataset_dir):
        pool = mp.Pool(int(mp.cpu_count()/2))
        files = glob.glob(dataset_dir + '/*.csv')
        results = pool.map(self._cache_feature_file, files)
        pool.close()

        for r_set in results:
            e_data, f_list, ts_ent = r_set
            print('unpack ', len(f_list))
            self.entity_data.update(e_data)
            self.feature_list.extend(f_list)
            self.ts_to_entity.update(ts_ent)

    def __len__(self):
        return int(len(self.feature_list))

    def __getitem__(self, index):
        video_id, target_entity_id, center_ts = self.feature_list[index]
        entity_context = self.get_speaker_context(self.ts_to_entity, video_id,
                                                  target_entity_id, center_ts,
                                                  self.num_speakers)

        target = self.entity_data[video_id][target_entity_id][center_ts][1]
        feature_set = np.zeros((self.num_speakers, self.seq_length, 512+160))
        for idx, ctx_entity in enumerate(entity_context):
            time_context = self.get_time_context(self.entity_data,
                                                 video_id,
                                                 ctx_entity, center_ts,
                                                 self.half_time_length,
                                                 self.time_stride)
            features = self.get_time_indexed_feature(video_id, ctx_entity,
                                                     time_context)
            feature_set[idx, ...] = features

        feature_set = np.asarray(feature_set)
        feature_set = np.swapaxes(feature_set, 0, 2)
        return np.float32(feature_set), target


class TM_ISRM_Forward_Dataset(TM_ISRM_BaseDataset):
    def __init__(self, csv_file_path, seq_length, time_stride, num_speakers):
        # Space config
        self.seq_length = seq_length
        self.time_stride = time_stride
        self.num_speakers = num_speakers
        self.half_time_length = math.floor(self.seq_length/2)

        # In memory data
        self.feature_list = []
        self.ts_to_entity = {}
        self.entity_data = {}

        # Single video metdadata
        self.entity_data, self.feature_list, self.ts_to_entity =   self._cache_feature_file(csv_file_path)


    def __len__(self):
        return int(len(self.feature_list))

    def __getitem__(self, index):
        video_id, target_entity_id, center_ts = self.feature_list[index]
        entity_context = self.get_speaker_context(self.ts_to_entity, video_id,
                                                  target_entity_id, center_ts,
                                                  self.num_speakers)

        feature_set = np.zeros((self.num_speakers, self.seq_length, 512+160))
        for idx, ctx_entity in enumerate(entity_context):
            time_context = self.get_time_context(self.entity_data,
                                                 video_id,
                                                 ctx_entity, center_ts,
                                                 self.half_time_length,
                                                 self.time_stride)
            features = self.get_time_indexed_feature(video_id, ctx_entity,
                                                     time_context)
            feature_set[idx, ...] = features

        feature_set = np.asarray(feature_set)
        feature_set = np.swapaxes(feature_set, 0, 2)
        return np.float32(feature_set), video_id, center_ts, target_entity_id