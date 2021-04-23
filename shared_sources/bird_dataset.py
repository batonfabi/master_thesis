import os
import soundfile
import math
import numpy as np
import torch
import logging
import librosa
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    """
    Implements torch.utils.data.Dataset  to provide sliced audio data in a given format

    ...

    Methods
    -------
    set_function(function)
        sets function which is used to convert the slices of audio data into a given type

    get_classes()
        returns known training classes
    
    shuffle_dataset()
        schuffles the dataset
    """

    __white_list_formats = {'flac'}

    def __init__(self, data, classes, batch_size, dim, samplerate, slice_length, channels, funct, allow_shuffle=True):
        """
            Parameters
            -----------

            data
                list of trainingdata tuples in the format ['path to file', start_pos_for_window , class_idx]
            classes: 
                list of classes
            batch_size: 
            dim: 
                dimension to reshape outputdata
            samplerate: 
                samplerate of audiofiles
            slice_length:
                length of slices for audiosamples in seconds
            channels: 
                num of channels of outputdata 
            funct:
                function to transform output data, for example to transform the data to mel spectograms 
            allow_shuffle (bool, optional):
                is shuffeling allowed 
        """
        self._data = data
        self._classes = classes
        self._batch_size = batch_size
        self._slice_length = slice_length
        self._samplerate = samplerate
        self._funct = funct
        self._dim = dim
        self._channels = channels
        self._allow_shuffle = allow_shuffle
        self.shuffle_dataset()
        
    def set_funct(self, funct):
        self._funct = funct
        
    def get_classes(self):
        return self._classes

    def shuffle_dataset(self):
        if self._allow_shuffle:
            np.random.shuffle(self._data)
        
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self._data) / self._batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        """

        if index < self.__len__():
            batch = self._data[index * self._batch_size:(index+1)*self._batch_size]
            tempy = [e[2] for e in batch]
            tempX = [[e[0], e[1]] for e in batch]

            x_ret = self.__data_generation(tempX)
            y_ret = tempy
            x_ret = torch.from_numpy(x_ret)
            
            # todo: 
            # you have to make y_ret to kategorical i guess if you want to use y as label
            # for gans its not needed

            y_ret = torch.LongTensor(y_ret)
            return x_ret, y_ret
        else:
            self.shuffle_dataset()
            raise IndexError("that's enough!")

    def __data_generation(self, x_data):
        """
        Generates data containing batch_size samples
        """
        wav_batch = []
        for idx, element in enumerate(x_data):
            file_path = element[0]
            sample_idx = element[1]
            wav_array,_ = librosa.load(file_path, sr=self._samplerate, offset=int(sample_idx), duration=self._slice_length)
            if len(wav_array) < int(self._samplerate * self._slice_length):
                pad = int(self._samplerate * self._slice_length) - len(wav_array)
                pad = np.zeros(pad,dtype=np.float32)
                wav_array = np.append(wav_array,pad)
            wav_batch.append(wav_array)
        x_tmp = self._funct(wav_batch)
        if self._channels != -1:
            x_tmp = np.reshape(x_tmp, (*self._dim, self._channels))
        x_ret = x_tmp
        return x_ret

    
    def __repr__(self):
        return "not implemented"

class CustomDataGenerator():
    """
    creates training and test dataset from the class BirdDataset 
    ...

    Methods
    -------
    get_current_dimension()
        retunrs shape of output data

    get_generators()
        retunrs traindata_generator and testdata_generator 
    
    """
    'Generates data for pytorch'
    __white_list_formats = {'flac'}

    def __init__(self, data_path, class_folders, slice_length, samplerate, trainingdata_amount, batch_size, channels, funct, hop_in_data, debug=False, rewrite_npy=False, allow_shuffle=True):
        """
            Parameters
            -----------

            data_path 
                root path to data
            class_folders
                classes to use inside the root path. Classes have to be seperated by folders
            slice_length:
                length of slices for audiosamples in seconds
            samplerate
                samplerate of audofiles 
            trainingdata_amount 
                percentage of data that should be used as trainingdta 1 for 100% 
            batch_size

            channels: 
                num of channels of outputdata 
            funct:
                function to transform output data, for example to transform the data to mel spectograms 
            allow_shuffle (bool, optional):
                is shuffeling allowed 
            hop_in_data
                windowsize of moving window
            debug 
                if true  debug output will be printed
            rewrite_npya
                if true the saved folder analysis from classes will be overwritten 
        """
        self._debug = debug
        self._rewrite_npy = rewrite_npy
        self._data_path = data_path
        self._class_folders = class_folders
        self._slice_length = slice_length
        self._samplerate = samplerate
        self._trainingdata_amount = trainingdata_amount
        self._batch_size = batch_size
        self._funct = funct
        self._channels = channels
        self._hop_in_data = hop_in_data
        self._dim = self.get_current_dimension()
        self._allow_shuffle = allow_shuffle

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARN)

    def get_current_dimension(self):
        """
        get shape of outputdata
        """
        y = np.zeros((self._batch_size, int(self._slice_length * self._samplerate)),dtype=np.float32)
        ret = self._funct(y)
        return np.shape(ret)

    def get_generators(self):
        """
        returns trainingdata generator and testdata generator
        """
        classes = self._get_classes(self._class_folders, self._data_path)
        data_train = self._get_data_dict(classes,self._data_path, self._rewrite_npy, self._samplerate, self._hop_in_data)
        data_train, data_test = self._prepare_training_and_test_data(data_train, self._trainingdata_amount)
        train_gen = BirdDataset(data_train, classes, self._batch_size, self._dim, self._samplerate, self._slice_length, self._channels, self._funct, self._allow_shuffle)
        test_gen = BirdDataset(data_test, classes, self._batch_size, self._dim, self._samplerate, self._slice_length, self._channels, self._funct, self._allow_shuffle)
        return train_gen, test_gen

    def _get_classes(self, class_folders:str, data_path:str):
        """
        returns list of training classes  
        """
        class_list = []
        for class_folder in class_folders:
            folder_path = os.path.join(data_path, class_folder)
            if os.path.isdir(folder_path):
                class_list.append(class_folder)
        return class_list

    def _get_data_dict(self, class_list:list, data_path:str, rewrite_npy:bool, samplerate:int, hop_in_data:int):
        """
        analyses trainingdata   
        """
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

    def _prepare_training_and_test_data(self, data_dict, trainingdata_amount):
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