from os import path, walk, makedirs
import random
from math import ceil

import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf

from env import FLAGS
from toolkit.dirty import read_list, write_list, resolve_coord_in, resolve_coord_in_v2


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

    dict_track = None

    def __init__(self, path_scene=None, path_track=None, img_shape=None, prefix='', postfix='',
                 rescale_img_max=1, rescale_track=None, list_exclusion=None):

        self.path_scene = path_scene
        self.path_track = path_track
        self.img_shape = img_shape
        self.prefix = prefix
        self.postfix = postfix

        self.rescale_img_max = rescale_img_max
        self.rescale_track = rescale_track
        self.list_exclusion = list_exclusion if list_exclusion is not None else list()

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

        dict_train_kv = dict()
        dict_test_kv = dict()
        for key in list_train_key:
            dict_train_kv.update({key: tuple([paths[key], self.dict_track[key]])})
        for key in list_test_key:
            dict_test_kv.update({key: tuple([paths[key], self.dict_track[key]])})

        dataset_train = tf.data.Dataset.from_tensor_slices((list(dict_train_kv.keys()), list(dict_train_kv.values())))
        # dataset_test = tf.data.Dataset.from_tensor_slices(list(list(dict_test_kv.values())))

        # dataset_train = dataset_train.map()
        # dataset_test = dataset_test.map()

        print("KFold:", len(train_idx), len(test_idx))

        # fin
        return

    @staticmethod
    def _read_func(path_b, track, img_shape, rescale_img_max):

        # key = str(key_b, encoding="utf-8")
        path = str(path_b, encoding="utf-8")

        scene = np.true_divide(
            np.fromfile(path, dtype=np.uint16).reshape(img_shape),
            rescale_img_max)

        # fin
        return scene, track

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


class Dataset_PRegv3(object):

    _in_shape = None
    _batch_size = -1
    _output_size = -1

    _scene_path = None
    _scene_key_list = None
    _scene_resolve_list = None

    _track_path = None
    _track_key_list = None
    _track_resolve_list = None
    _track_path_exception = None

    _mutual_key_list = None
    _train_list = None
    _valid_list = None
    _test_list = None
    _fold_path = None
    _use_all_fold = False
    _kf = -1
    _kf_train = None
    _kf_valid = None
    _kf_test = None
    _preset_fold = False
    _use_valid = False

    _input_norm = False
    _input_norm_scale = 0

    # batch dataset readout
    _read_key_pos = 0

    # statistics
    _concurrent = -1
    _stat_multiple_item = None
    _multiple_rev_dict = None
    _false_label_list = None

    lat_mean  = 0
    lat_devi  = 0
    long_mean = 0
    long_devi = 0
    min_list = None
    max_list = None
    mean_list = None
    devi_list = None

    def __init__(self, path_scene=None, path_track=None, path_track_exception=None,
                 path_fold=None, preset_fold=False, use_all_fold=False, kfold=10,
                 use_valid=False, in_shape=None,
                 norm_policy='minmax', min_list=None, max_list=None,
                 input_norm=True, input_norm_scale=1,
                 batch_size=1, output_size=FLAGS.output_length, concurrent=0):

        self._in_shape = in_shape

        self._scene_path = path_scene
        self._track_path = path_track
        self._track_path_exception = path_track_exception
        self._batch_size = batch_size
        self._output_size = output_size

        # for coordinate normalization
        self.min_list = np.array(min_list, dtype=np.float32)
        self.max_list = np.array(max_list, dtype=np.float32)

        if norm_policy == 'minmax':
            self.mean_list = self.min_list
            self.devi_list = self.max_list - self.min_list
        elif norm_policy == 'posneg':
            self.mean_list = np.true_divide(self.max_list + self.mean_list, 2)
            self.devi_list = self.max_list - self.mean_list

        # for input normalization
        self._input_norm = input_norm
        self._input_norm_scale = input_norm_scale

        # concurrency
        self._concurrent = concurrent

        # pre-build fold
        self._preset_fold = preset_fold
        self._use_all_fold = use_all_fold
        self._kf = kfold
        self._fold_path = path_fold if path_fold is not None else None
        self._use_valid = use_valid

        # pring coordinates settings
        # print("[I] Dataset_PRegv3.__init__(): [coord settings] lat->(%.1f, %.1f), long->(%.1f, %.1f)"
        #       % (self.lat_mean, self.lat_devi, self.long_mean, self.long_devi))
        info_norm_str = "[I] Dataset_PRegv3.__init__():\nmin:{0}\nmax:{1}\nmean:{2}\ndevi:{3}"
        print(info_norm_str.format(self.min_list, self.max_list, self.mean_list, self.devi_list))

        # build data model
        self.build()

    # data-set loading
    def read_binary(self, dpath=None, norm_max=1024):

        # resolve = np.fromfile(file=dpath, dtype='u2').reshape(self._in_shape)
        resolve = np.fromfile(file=dpath, dtype=np.uint16).reshape(self._in_shape)

        if self._input_norm:
            larger_than = resolve > norm_max - 1
            resolve[larger_than] = norm_max - 1
            # set input normalization scaling
            resolve = np.true_divide(resolve, norm_max) * self._input_norm_scale
        else:
            pass

        return resolve

    def load_track_labels(self):

        track_dict = dict()
        multiple_counts = dict()

        # load exception list
        exception_list = list()
        with open(self._track_path_exception) as f:
            for line in f:
                key = line.strip().split()[0]
                exception_list.append(key)
        exception_list.sort()

        # load track list without exceptions
        with open(self._track_path) as f:
            for line in f:
                arr = line.strip().split()
                key = arr[0]

                # skip exception item
                if key in exception_list:
                    continue
                else:
                    pass

                # overflow occurred when prediction (>=1.0)
                # for lat, scope: -12~68 (based on COMS Map)
                # for long scope: 50~206 (based on COMS Map)

                input_list = np.array(arr[1:5], dtype=np.float32)

                norm_list = resolve_coord_in_v2(list_input=input_list,
                                                list_mean=self.mean_list,
                                                list_devi=self.devi_list)

                # lat_norm, long_norm = resolve_coord_in(
                #     lat=float(arr[2]), long=float(arr[3]),
                #     latm=self.lat_mean, latd=self.lat_devi,
                #     longm=self.long_mean, longd=self.long_devi
                # )
                # point_instance = [1, lat_norm, long_norm]

                point_inst = np.concatenate([[1], norm_list])

                # update valid item
                if key not in track_dict:
                    track_dict.update({key: list()})
                else:
                    pass
                # track_dict[key].append(point_instance)
                track_dict[key].append(point_inst)

                # counting items in scene
                if key not in multiple_counts:
                    multiple_counts.update({key:0})
                else:
                    pass
                multiple_counts[key] += 1

        # reverse resolve keys for sorting
        multiple_rev_dict = dict()
        for key in multiple_counts.keys():
            v = multiple_counts[key]
            if v in multiple_rev_dict:
                pass
            else:
                multiple_rev_dict.update({v: list()})
            multiple_rev_dict[v].append(key)

        self._multiple_rev_dict = multiple_rev_dict
        self._stat_multiple_item = [len(multiple_rev_dict[key]) for key in multiple_rev_dict.keys()]
        print("[I] Dataset_PRegv3.load_track_labels(): get_counts:", self._stat_multiple_item)

        # finalize available key lists
        self._track_key_list = track_dict.keys()
        self._track_resolve_list = track_dict

    def output_filler(self, output_pre, skip_conf=False):

        # set number of fills to match output size
        num2fill = self._output_size - len(output_pre)
        conf_zeros_fill = np.zeros(num2fill).reshape([-1, 1])

        # fill zeros on filler
        lat_fill = np.ones(num2fill).reshape([-1, 1])
        long_fill = np.ones(num2fill).reshape([-1, 1])

        # merge valid coordinates and filler
        output_list = list()
        output_list.append(output_pre)
        if num2fill > 0:
            dummy_stack = np.dstack([conf_zeros_fill, lat_fill, long_fill]).reshape([num2fill, -1])
            output_list.append(dummy_stack)
        else:
            pass

        fill2concat = np.concatenate(output_list)
        if skip_conf:
            fill2concat = fill2concat[:, 1:]
        else:
            pass

        return fill2concat

    def output_filler_v2(self, output_pre, size=1, skip_conf=False):

        # set number of fills to match output size
        num2fill = self._output_size - len(output_pre)

        # merge valid coordinates and filler
        output_list = list()
        output_list.append(output_pre)
        if num2fill > 0:
            dummy_stack = np.zeros(shape=[num2fill, size])
            output_list.append(dummy_stack)
        else:
            pass

        fill2concat = np.concatenate(output_list)
        if skip_conf:
            fill2concat = fill2concat[:, 1:]
        else:
            pass

        return fill2concat

    def resolve_false_label(self):

        self._false_label_list = list()

        for key in self._scene_key_list:
            if key not in self._track_key_list:
                self._false_label_list.append(key)
            else:
                continue

        print("[I] Dataset_PRegv3.resolve_false_label(): false label:", len(self._false_label_list))

    def set_fold(self, n_splits=10):

        # kFold
        kf = KFold(n_splits=n_splits, shuffle=True)

        # resolved folds
        train_fold = list()
        valid_fold = list()
        test_fold = list()

        # params initialization
        X_train_full, X_train, X_valid, X_test = None, None, None, None

        # train+valid vs test
        mutual_key_list_np = np.array(self._mutual_key_list)
        for train, test in kf.split(self._mutual_key_list):
            X_train_full, X_test = mutual_key_list_np[train], mutual_key_list_np[test]
            # break for single split
            # break

            if self._use_valid:
                # train vs valid
                for train, valid in kf.split(X_train_full):
                    X_train, X_valid = X_train_full[train], X_train_full[valid]
                    # break for single split
                    break
            else:
                X_train = X_train_full
                X_valid = X_test

            train_fold.append(X_train)
            valid_fold.append(X_valid)
            test_fold.append(X_test)

            # break

        # save all folds
        if self._use_all_fold:
            self._kf_train = train_fold
            self._kf_valid = valid_fold
            self._kf_test = test_fold
        else:
            pass

        # always use first fold while not loading custom batch
        self._train_list = train_fold[0]
        self._valid_list = valid_fold[0]
        self._test_list = test_fold[0]

        print("[I] Dataset_PRegv3.set_fold(): total fold size: %d" % len(test_fold))
        print("[I] Dataset_PRegv3.set_fold(): train: %d, valid: %d, test: %d" %
              (len(self._train_list), len(self._valid_list), len(self._test_list)))

    def load_fold(self):

        self._train_list = read_list(path=path.join(self._fold_path, 'train.lst'), sort=True, as_ndarray=True)
        self._valid_list = read_list(path=path.join(self._fold_path, 'valid.lst'), sort=True, as_ndarray=True)
        self._test_list = read_list(path=path.join(self._fold_path, 'test.lst'), sort=True, as_ndarray=True)

        print("[I] Dataset_PRegv3.load_fold(): fold loading from %s completed" % self._fold_path)
        print("[I] Dataset_PRegv3.load_fold(): Statistics: Train:{0}/Valid:{1}/Test:{2}".format(
            len(self._train_list), len(self._valid_list), len(self._test_list)
        ))

    def save_all_fold(self):

        for i in range(len(self._kf_test)):

            fold_path = path.join(self._fold_path, "fold%02d" % (i + 1))
            if not path.exists(fold_path):
                makedirs(fold_path)
            else:
                pass

            # save fold list into file
            write_list(path=path.join(fold_path, 'train.lst'), item_list=self._kf_train[i])
            write_list(path=path.join(fold_path, 'valid.lst'), item_list=self._kf_valid[i])
            write_list(path=path.join(fold_path, 'test.lst'), item_list=self._kf_test[i])

        print("[I] Dataset_PRegv3.save_all_fold(): finished saving all folds")
        print("[I] Dataset_PRegv3.save_all_fold(): all files saved at:", self._fold_path)

    def build(self):

        # load list of available sensory images
        self._scene_key_list, self._scene_resolve_list = \
            load_scene_list(path_src=self._scene_path, prefix='', postfix='.bin')

        # load track labels
        self.load_track_labels()

        print("[I] Dataset_PRegv3.build(): loaded img keys:", len(self._scene_key_list), "/ loaded tracks:", len(self._track_key_list))

        if self._concurrent == 0:
            # key setting v2: based on pre-filtered image directory
            # use entire scenes for model
            self._mutual_key_list = self._scene_key_list
        else:
            # key setting v3: based on simultaneous option
            # use exact count of items in scene

            # resolve false labels
            self.resolve_false_label()
            # pick mutual key list (scene vs track)
            self._mutual_key_list = self._multiple_rev_dict[self._concurrent] + self._false_label_list

            print("[I] Dataset_PRegv3.build(): loaded section:", len(self._multiple_rev_dict[self._concurrent]),
                  "/ extra:", len(self._false_label_list))

        self._mutual_key_list = np.array(self._mutual_key_list)

        print("[I] Dataset_PRegv3.build(): final keys:", len(self._mutual_key_list))

        if self._preset_fold:
            self.load_fold()
        else:
            # set kFolds (single time)
            self.set_fold(n_splits=self._kf)

    def resolve_track(self, key, skip_conf=False):

        if key in self._track_resolve_list:
            resolve = self._track_resolve_list[key]
        else:
            resolve = list()

        # print('resolve:', resolve)

        fill_size = len(self.min_list)
        resolved = self.output_filler_v2(resolve, size=fill_size, skip_conf=skip_conf)

        return resolved

    def get_next_batch(self, mode='train', out_as_np=False, skip_conf=False, fit_split=1, scrap=-1, start_idx=-1):

        key_list = None

        # pick random keys (within batch size) based on pick mode
        if mode == 'train':
            key_list = self._train_list
        elif mode == 'valid':
            key_list = self._valid_list
        elif mode == 'test':
            key_list = self._test_list
        elif mode == 'full':
            key_list = self._mutual_key_list

        if mode == 'train':
            batch_keys = random.sample(key_list.tolist(), self._batch_size)
        elif mode == 'valid' or mode == 'test':
            bottom = self._read_key_pos * self._batch_size
            top = bottom + self._batch_size
            max_len = len(key_list)
            top = max_len if top > max_len else top
            # print("key read from [%d:%d]/%d" % (bottom, top, max_len))
            batch_keys = key_list[bottom:top]
            self._read_key_pos = self._read_key_pos + 1 if top < max_len else 0

            # ugly case - trouble when multi-gpu environment
            # activate only when _read_key_pos is reversed
            # add aux random samples for fit EVEN multi-split division
            if top >= max_len and len(key_list) % fit_split != 0:
                remainder = np.remainder(len(key_list), fit_split)
                to_fill = fit_split - remainder

                # print('[D] prep2.get_next_batch: list_size:{0}, remainder: {1}, to_fill: {2}'.format(
                #     len(batch_keys), remainder, to_fill
                # ))
                # if there exists any aux things to concatenate
                if to_fill > 0:
                    remainder_sample = random.sample(key_list.tolist(), to_fill)
                    batch_keys = np.concatenate([batch_keys, remainder_sample])
                else:
                    pass
                print('[D] prep2.get_next_batch: revised final_list_size:{0}'.format(len(batch_keys)))
            else:
                pass
        else:
            batch_keys = None

        targets = [self._scene_resolve_list[key] for key in batch_keys]
        scenes = [self.read_binary(dpath=scene_path) for scene_path in targets]
        tracks = [self.resolve_track(key, skip_conf=skip_conf) for key in batch_keys]

        if out_as_np:
            scenes = np.array(scenes)
            tracks = np.array(tracks)
        else:
            pass

        # print('scrap:', scrap)
        if scrap > 0 and start_idx > -1:
            tracks = tracks[:,:,start_idx:start_idx+scrap]
        else:
            pass

        return scenes, tracks

    def get_next_batch_with_shuffled_labels(self):

        scenes, tracks = self.get_next_batch()

        for track in tracks:
            np.random.shuffle(track)

        return scenes, tracks

    def get_iter_size(self, mode='train'):

        resolve = 0

        if mode == 'train':
            resolve = ceil(len(self._train_list) / self._batch_size)
        elif mode == 'valid':
            resolve = ceil(len(self._valid_list) / self._batch_size)
        elif mode == 'test':
            resolve = ceil(len(self._test_list) / self._batch_size)
        elif mode == 'full':
            resolve = ceil(len(self._mutual_key_list) / self._batch_size)

        return resolve

    def get_test_set(self):
        return self._test_list

    def resolve_keys(self, keys=None):

        targets = [self._scene_resolve_list[key] for key in keys]
        scenes = [self.read_binary(dpath=scene_path) for scene_path in targets]
        tracks = [self.resolve_track(key) for key in keys]

        return scenes, tracks

    def resolve_coord_out_v2(self, list_input, long_index=2, long_overflow_fix=True):

        # print('list_input:', list_input.shape, list_input)

        list_denorm = (list_input * self.devi_list) + self.mean_list
        # print('list_denorm:', list_denorm)
        long_denorm = list_denorm[:,:,long_index]
        # print('long_denorm:', long_denorm)

        # latitude may not overflow, but only longitude
        if long_overflow_fix:
            long_denorm = [self.fix_longi_overflow(x) for x in long_denorm]
        else:
            pass

        list_denorm[:,:,long_index] = long_denorm

        return list_denorm

    def fix_longi_overflow(self, long_denorm):

        if abs(long_denorm) > float(180):
            long_sign = long_denorm / abs(long_denorm)
            long_rslv = -long_sign * (float(360) - abs(long_denorm))
            long_denorm_fix = long_rslv
        else:
            long_denorm_fix = long_denorm

        return long_denorm_fix
