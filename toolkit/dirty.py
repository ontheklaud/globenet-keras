import os
import random

import numpy as np
from geopy.distance import vincenty, great_circle
import tensorflow as tf


def resolve_coord_in(lat, long, latm, latd, longm, longd):

    lat_norm = (lat - latm)/latd
    long_norm = (long - longm)/longd

    return lat_norm, long_norm

def resolve_coord_in_v2(list_input, list_mean, list_devi):

    norm_list = np.true_divide((list_input - list_mean), list_devi)

    return norm_list

def resolve_coord_out(lat, long, latm, latd, longm, longd, long_overflow_fix=True):

    lat_denorm = (lat * latd) + latm
    long_denorm = (long * longd) + longm

    long_sign = long_denorm / abs(long_denorm)

    # latitude may not overflow, but only longitude
    if long_overflow_fix and abs(long_denorm) > float(180):
        long_rslv = -long_sign * (float(360) - abs(long_denorm))
        long_denorm = long_rslv
        # print('[D] lat_rslv fixed from %.1f to %.1f' % (long_denorm, long_rslv))
    else:
        # do nothing
        pass

    return lat_denorm, long_denorm

def resolve_coord_out_v2(list_output, list_mean, list_devi, long_idx=2, long_overflow_fix=True):

    denorm_list = (list_output * list_devi) + list_mean

    # longitude overflow issue on basic data

    long_denorm = denorm_list[long_idx]

    long_sign = long_denorm / abs(long_denorm)
    # latitude may not overflow, but only longitude
    if long_overflow_fix and abs(long_denorm) > float(180):
        long_rslv = -long_sign * (float(360) - abs(long_denorm))
        long_denorm = long_rslv

        denorm_list[long_idx] = long_denorm
        # print('[D] lat_rslv fixed from %.1f to %.1f' % (long_denorm, long_rslv))
    else:
        # do nothing
        pass

    return denorm_list

def latlong_distance(coord_a=None, coord_b=None, type='great_circle'):
    resolve = -1

    try:
        if type is 'vincenty':
            resolve = vincenty(coord_a, coord_b).kilometers
        elif type is 'great_circle':
            resolve = great_circle(coord_a, coord_b).kilometers
    except ValueError:
        pass

    return resolve


def read_list(path=None, sort=True, as_ndarray=False):
    resolve = list()
    with open(path) as f:
        for line in f:
            resolve.append(line.strip())

    if sort:
        resolve.sort()
    else:
        pass

    if as_ndarray:
        resolve = np.array(resolve)
    else:
        pass

    return resolve


def write_list(path=None, filemode='w', item_list=None):
    with open(path, filemode) as f:
        for item in item_list:
            f.write(item + '\n')


def summary_filters(namespace, tensor, trigger=True, num_of_filters=1):
    if trigger:
        try:
            candidates = random.sample(range(tensor.get_shape()[-1].value), num_of_filters)
            for idx in candidates:
                tf.summary.image(namespace+'/'+str(idx), tensor[:1, :, :, idx:idx+1], max_outputs=1)
        except ValueError:
            print('failed to summary %d filter(s) from tensor %s' % (num_of_filters, namespace))
            pass
    else:
        pass


def np_mean(x=None):
    return np.mean(x)
