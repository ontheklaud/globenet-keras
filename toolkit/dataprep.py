from os import path, walk, makedirs

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf


def load_scene_list(path_src=None, prefix='', postfix='', exclusion_list=None):
    paths = dict()

    for root, dirs, files in walk(path_src, topdown=False):
        for name in files:

            # old school name split
            # key = name.replace(prefix, '').replace(postfix, '')
            nameonly, ext = path.splitext(name)
            if postfix.lower() == ext.lower():
                key = nameonly.replace(prefix, '')

                # check exclusion list
                if exclusion_list is not None and key in exclusion_list:
                    pass
                else:
                    path_profile = path.join(root, name)
                    paths.update({key: path_profile})
            else:
                continue

    keys = list(paths.keys())
    keys.sort()

    return keys, paths


class Dataset_SatReg(object):

    path_scene = None
    path_track = None
    img_shape = None
    prefix = None
    postfix = None

    rescale_img_max = None
    rescale_track = None
    list_exclusion = None
    batch_size = None

    dict_track = None

    # dataset_train = None
    # dataset_test = None

    it_train = None
    it_test = None

    def __init__(self, path_scene=None, path_track=None, img_shape=None, prefix='', postfix='',
                 rescale_img_max=1, rescale_track=None, list_exclusion=None, batch_size=1):

        self.path_scene = path_scene
        self.path_track = path_track
        self.img_shape = img_shape
        self.prefix = prefix
        self.postfix = postfix

        self.rescale_img_max = rescale_img_max
        self.rescale_track = rescale_track
        self.list_exclusion = list_exclusion if list_exclusion is not None else list()
        self.batch_size = batch_size

        # set-up

        self.setup()

        # fin
        return

    def setup(self):

        # read tracks
        self.read_tracks()

        # read image keys
        keys, paths = load_scene_list(path_src=self.path_scene,
                                      prefix=self.prefix, postfix=self.postfix, exclusion_list=self.list_exclusion)

        kf = KFold(n_splits=10, shuffle=True)
        train_idx, test_idx = next(kf.split(X=keys))
        list_train_key = [keys[idx] for idx in train_idx]
        list_test_key = [keys[idx] for idx in test_idx]

        # dict_train_kv = dict()
        # dict_test_kv = dict()
        # for key in list_train_key:
        #     dict_train_kv.update({
        #         key: dict({"key": paths[key], "track": self.dict_track[key]})
        #     })

        # for key in list_test_key:
        #     dict_test_kv.update({key: tuple([paths[key], self.dict_track[key]])})

        dataset_train = tf.data.Dataset.from_tensor_slices((
            # list_train_key,
            [paths[key] for key in list_train_key],
            [self.dict_track[key] for key in list_train_key]
        ))

        dataset_test = tf.data.Dataset.from_tensor_slices((
            # list_test_key,
            [paths[key] for key in list_test_key],
            [self.dict_track[key] for key in list_test_key]
        ))

        dataset_train = dataset_train.map(
            lambda path, track: tuple(tf.py_func(
                self._read_func,
                [path, track, self.img_shape, self.rescale_img_max],
                [tf.float32, tf.float32, tf.float32]
            ))
        ).shuffle(buffer_size=self.batch_size*50).batch(batch_size=self.batch_size)

        dataset_test = dataset_test.map(
            lambda path, track: tuple(tf.py_func(
                self._read_func,
                [path, track, self.img_shape, self.rescale_img_max],
                [tf.float32, tf.float32, tf.float32]
            ))
        ).batch(batch_size=self.batch_size).prefetch(buffer_size=self.batch_size*50)

        self.it_train = dataset_train.make_initializable_iterator()
        self.it_test = dataset_test.make_initializable_iterator()

        print("KFold:", len(train_idx), len(test_idx))

        # fin
        return

    @staticmethod
    def _read_func(path_b, track, img_shape, rescale_img_max):

        # key = str(key_b, encoding="utf-8")
        path = str(path_b, encoding="utf-8")

        scene = np.true_divide(
            np.fromfile(path, dtype=np.uint16).reshape(img_shape),
            rescale_img_max).astype(np.float32)

        lat, long = track

        # fin
        return scene, lat, long

    def read_tracks(self):

        lat_min = self.rescale_track[0]
        long_min = self.rescale_track[1]
        lat_max = self.rescale_track[2]
        long_max = self.rescale_track[3]
        lat_bound = lat_max - lat_min
        long_bound = long_max - long_min

        f_handle = open(self.path_track, mode='r')

        dict_track = dict()
        for _line in f_handle:

            # factors: KEY SCALE LATITUDE LONGITUDE AIRPRESSURE WINDSPEED
            # only use factors[0, 2, 3]
            factors = _line.strip().split('\t')
            key = factors[0]
            value = [float(x) for x in factors[2:4]]
            value[0] = np.true_divide(value[0] - lat_min, lat_bound)
            value[1] = np.true_divide(value[1] - long_min, long_bound)

            print("LINE:", key, value)
            dict_track.update({key: value})


        # clean-up
        f_handle.close()

        self.dict_track = dict_track

        # fin
        return
