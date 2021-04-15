import os
import soundfile
import math
import numpy as np
import torch
import logging
import librosa
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    """Face Landmarks dataset."""
    __white_list_formats = {'flac'}

    def __init__(self, data_list, classes, batch_size, dim, samplerate, frames, channels, funct, allow_shuffle=True):
        'Initialization'
        self.data_list = data_list
        self.classes = classes
        self.batch_size = batch_size
        self.frames = frames
        self.samplerate = samplerate
        self.funct = funct
        self.dim = dim
        self.channels = channels
        self.allow_shuffle = allow_shuffle
        self.shuffle_dataset()
        
    def set_funct(self, funct):
        self.funct = funct
        
    def get_classes(self):
        return self.classes

    def shuffle_dataset(self):
        if self.allow_shuffle:
            np.random.shuffle(self.data_list)
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if index < self.__len__():
            batch = self.data_list[index * self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs

            tempy = [e[2] for e in batch]
            tempX = [[e[0], e[1]] for e in batch]

            # Generate data
            x_ret = self.__data_generation(tempX)
            y_ret = tempy
            x_ret = torch.from_numpy(x_ret)
            # todo: 
            # you have to make y_ret to kategorical i guess
            # y_ret = torch.FloatTensor(y_ret)
            y_ret = torch.LongTensor(y_ret)
            return x_ret, y_ret
        else:
            self.shuffle_dataset()
            raise IndexError("that's enough!")

    def __data_generation(self, x_data):
        'Generates data containing batch_size samples'

        # Generate data
        wav_batch = []
        for idx, element in enumerate(x_data):
            # Store sample
            file_path = element[0]
            sample_idx = element[1]

            #wav_array, _ = soundfile.read(file_path, int(self.frames*self.samplerate), int(sample_idx), fill_value=0)
            
            wav_array,_ = librosa.load(file_path, sr=self.samplerate, offset=int(sample_idx), duration=self.frames)
            if len(wav_array) < int(self.samplerate * self.frames):
                pad = int(self.samplerate * self.frames) - len(wav_array)
                pad = np.zeros(pad,dtype=np.float32)
                wav_array = np.append(wav_array,pad)
            wav_batch.append(wav_array)
        x_tmp = self.funct(wav_batch)
        if self.channels != -1:
            x_tmp = np.reshape(x_tmp, (*self.dim, self.channels))
        x_ret = x_tmp
        return x_ret

    
    def __repr__(self):
        return "not implemented"

class CustomDataGenerator():
    'Generates data for pytorch'
    __white_list_formats = {'flac'}

    def __init__(self, data_path, class_folders, slice_length, samplerate, trainingdata_amount, batch_size, channels, funct, hop_in_data, debug=False, rewrite_npy=False, allow_shuffle=True):
        self.debug = debug
        self.rewrite_npy = rewrite_npy
        self.data_path = data_path
        self.class_folders = class_folders
        self.slice_length = slice_length
        self.samplerate = samplerate
        self.trainingdata_amount = trainingdata_amount
        self.batch_size = batch_size
        self.funct = funct
        self.channels = channels
        self.hop_in_data = hop_in_data
        self.dim = self.get_current_dimension()
        self.allow_shuffle = allow_shuffle

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARN)

    def get_current_dimension(self):
        y = np.zeros((self.batch_size, int(self.slice_length * self.samplerate)),dtype=np.float32)
        ret = self.funct(y)
        return np.shape(ret)
    
    def get_current_dimension_bak(self):
        y = np.zeros(int(self.slice_length * self.samplerate))
        ret = self.funct(y)
        return np.shape(ret)

    def get_generators(self):
        classes = self._get_classes(self.class_folders, self.data_path)
        data_train = self._get_data_dict(classes,self.data_path, self.rewrite_npy, self.samplerate, self.hop_in_data)
        data_train, data_test = self.prepare_training_and_test_data(data_train, self.trainingdata_amount)
        train_gen = BirdDataset(data_train, classes, self.batch_size, self.dim, self.samplerate, self.slice_length, self.channels, self.funct, self.allow_shuffle)
        test_gen = BirdDataset(data_test, classes, self.batch_size, self.dim, self.samplerate, self.slice_length, self.channels, self.funct, self.allow_shuffle)
        return train_gen, test_gen

    def _get_classes(self, class_folders:str, data_path:str):
        class_list = []
        for class_folder in class_folders:
            folder_path = os.path.join(data_path, class_folder)
            if os.path.isdir(folder_path):
                class_list.append(class_folder)
        return class_list

    def _get_data_dict(self, class_list:list, data_path:str, rewrite_npy:bool, samplerate:int, hop_in_data:int):
        data_dict = dict()
        for class_folder in class_list:
            folder_path = os.path.join(data_path, class_folder)
            meta_folder = os.path.join(folder_path, "meta")
            if not os.path.exists(meta_folder):
                os.mkdir(meta_folder)
            file_np = os.path.join(meta_folder, "Datalist.npy")
            logging.debug("looking for: " + file_np)
            if not os.path.isfile(file_np) or rewrite_npy:
                if rewrite_npy:
                    logging.debug("Refreshing Datalist.npy")
                else:
                    logging.debug("Datalist.npy does not exist for " + class_folder)
                data_dict[class_folder] = []
                class_idx = class_list.index(class_folder)
                logging.debug("class_idx:" + str(class_idx))
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path) and file_path.endswith("flac"):
                        file_size = os.path.getsize(file_path)
                        if file_size > 1000:
                            info = soundfile.info(file_path)
                            if info.samplerate == samplerate:
                                x = [[file_path, i] for i in range(0,int(info.duration),hop_in_data)]
                                data_dict[class_folder].extend(x)
                            else:
                                raise RuntimeError('samplerate mismatch')
                logging.debug("creating " + file_np)
                np.save(file_np, data_dict[class_folder])
                [ele.append(class_idx) for ele in data_dict[class_folder]]
            else:
                class_idx = class_list.index(class_folder)
                logging.debug("Datalist.npy exist for " + class_folder)
                tmp = np.load(file_np)
                data_dict[class_folder] = tmp.tolist()
                [ele.append(class_idx) for ele in data_dict[class_folder]]
        return data_dict

    def prepare_training_and_test_data(self, data_dict, trainingdata_amount):
        """ Prepares the trainings- and test- dataset
            trainingdata_amount: between 0 and 1.0; specifies the amount of trainingsdata in percentage
        """
        train_data = []
        test_data = []
        for class_key in dict(data_dict).keys():
            class_data = data_dict[class_key]
            data_len = len(class_data)
            idx_train_end = int(data_len * trainingdata_amount)
            train_data.extend(class_data[:idx_train_end])
            test_data.extend(class_data[idx_train_end:])
        return train_data, test_data    