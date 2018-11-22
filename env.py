from os import getcwd
from os.path import join

import tensorflow as tf
from tensorflow.python.client import device_lib

# set FLAGS for tensorflow app flags
flags = tf.flags
FLAGS = tf.flags.FLAGS

DATASET_ROOT = join(getcwd(), "data")
flags.DEFINE_string('dataset_root', DATASET_ROOT, 'dataset root')
flags.DEFINE_string('path_scene', join(DATASET_ROOT, 'ndarray_Typhoons_resize'), 'path_scene')
flags.DEFINE_string('path_track', join(DATASET_ROOT, 'track_Typhoons.txt'), 'path_track')
flags.DEFINE_string('path_track_exception', join(DATASET_ROOT, 'track_Typhoons_NX.txt'), 'path_track_exception')
flags.DEFINE_string('summaries_dir', 'summary', 'summaries directory')
flags.DEFINE_string('ckpt_dir', 'ckpt', 'checkpoint directory')
flags.DEFINE_integer('output_length', 1, 'output_length')
flags.DEFINE_float('learning_rate', 1e-5, 'learning_rate')
flags.DEFINE_integer('epochs', 30, 'epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_integer('summary_step', 100, 'summary_step')

flags.DEFINE_string('model_name', 'model_name', 'model name')
flags.DEFINE_boolean('preset_fold', False, 'whether to use prebuilt fold set')
flags.DEFINE_string('path_fold', 'folds', 'path to Kfold')
flags.DEFINE_boolean('use_valid', False, 'whether to use discrete valid set')
flags.DEFINE_string('optimizer', 'AdamOptimizer', 'optimizer')
flags.DEFINE_string('conv_act_policy', 'relu', 'activation policy on conv')
flags.DEFINE_string('flat_act_policy', 'tanh', 'activation policy on dense')
flags.DEFINE_string('loss_policy', 'mse', 'loss policy')

flags.DEFINE_boolean('batch_norm', False, 'Batch Normalization')
flags.DEFINE_string('network_type', 'cplx', 'network topology type')
flags.DEFINE_integer('save_max_to_keep', 0, 'tf.train.Saver max_to_keep')

flags.DEFINE_boolean('allow_soft_placement', False, 'ConfigProto_allow_soft_placement')
flags.DEFINE_boolean('log_device_placement', False, 'ConfigProto_log_device_placement')

# session config
tf_sess_config = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
tf_sess_config.gpu_options.allow_growth = True

SESS_CFG = tf_sess_config

SECT_SIZE = 21

# Y_MAX = 1544
# X_MAX = 1934
Y_MAX = 768
X_MAX = 768

IN_SHAPE = (Y_MAX, X_MAX, 4)
PH_IN_SHAPE = (None, Y_MAX, X_MAX, 4)
PH_OUT_SHAPE_BBOX = (None, SECT_SIZE, SECT_SIZE, 2)
PH_OUT_SHAPE_BREG = (None, FLAGS.output_length, 3)
PH_OUT_POINT = (None, FLAGS.output_length, 2)
PH_OUT_INDEX = (None, FLAGS.output_length, 2)


HE5FIG = ['ir1', 'ir2', 'swir', 'wv']


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
