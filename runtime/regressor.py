from os import path, makedirs
import time

import numpy as np
import tensorflow as tf

from model.imglinreg import ImgLinReg
from model.imglinregcplx import ImgLinRegCplx
from env import FLAGS, IN_SHAPE, PH_IN_SHAPE, PH_OUT_INDEX, SESS_CFG
from toolkit.dataprep import Dataset_SatReg


def main(argv):

    sess = tf.Session(config=SESS_CFG)

    summaries_dir = path.abspath(FLAGS.summaries_dir)
    fold_dir = path.abspath(FLAGS.path_fold)
    ckpt_dir = path.abspath(FLAGS.ckpt_dir)
    if not path.exists(ckpt_dir):
        print('[E] Cannot find ckpt_dir')
        makedirs(ckpt_dir)
    else:
        pass

    rescale_track = [-12, 68, 50, 206]
    dataset = Dataset_SatReg(path_scene=FLAGS.path_scene, path_track=FLAGS.path_track,
                             img_shape=IN_SHAPE, postfix=".bin", rescale_img_max=1023, rescale_track=rescale_track,
                             list_exclusion=None, batch_size=FLAGS.batch_size)

    X = tf.placeholder(dtype=tf.float32, shape=PH_IN_SHAPE)
    Y = tf.placeholder(dtype=tf.float32, shape=PH_OUT_INDEX)

    if FLAGS.network_type == 'simple':
        Network = ImgLinReg
    elif FLAGS.network_type == 'cplx':
        Network = ImgLinRegCplx
    else:
        Network = None

    network = Network(x=X, y=Y,
                      conv_act_policy=FLAGS.conv_act_policy, flat_act_policy=FLAGS.flat_act_policy,
                      batch_norm=FLAGS.batch_norm, optimizer=getattr(tf.train, FLAGS.optimizer),
                      learning_rate=FLAGS.learning_rate)

    tb_writer = tf.summary.FileWriter(path.join(summaries_dir, 'train'), sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver(max_to_keep=FLAGS.save_max_to_keep)
    save_path = path.join(ckpt_dir, FLAGS.model_name)

    latest_ckpt_step = 0

    # load checkpoint iff there exists correct checkpoint
    ckpt_cond1 = tf.train.checkpoint_exists(checkpoint_prefix=ckpt_dir)
    ckpt_cond2 = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir) is not None
    if ckpt_cond1 and ckpt_cond2:
        # in-line checkpoint restoration
        latest_path = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)

        latest_ckpt_name = path.basename(latest_path)
        latest_ckpt_step = int(latest_ckpt_name.replace(FLAGS.model_name + '-', '')) + 1

        print('[I] {:.1f} : started restoring Model {:s} from {:s}'.format(time.time(), FLAGS.model_name, latest_ckpt_name))
        saver.restore(sess=sess, save_path=latest_path)

        print('[I] {:.1f} : Model {:s} restored from {:s} successfully!'.format(time.time(), FLAGS.model_name, latest_ckpt_name))
    else:
        print('[I] failed to load pre-trained model')

    # logger for validation error evaluation
    valid_error_log_path = path.join(ckpt_dir, "error_valid.log")
    valid_error_writer = open(valid_error_log_path, 'w')

    # logger for test error evaluation
    test_error_log_path = path.join(ckpt_dir, "error_test.log")
    test_error_writer = open(test_error_log_path, 'w')

    tasks = tuple(["train", "test"])

    it_global = 0
    for epoch in range(FLAGS.epochs):

        # Perform training
        time_begin = time.time()
        print("[{:.1f}|epoch {:04d}] INIT".format(time_begin, epoch + 1))

        for task in tasks:

            if task == "train":
                net_step = network.train()
                data_task = dataset.it_train
            elif task == "valid" or task == "test":
                net_step = network.valid()
                data_task = dataset.it_test
            else:
                net_step = None
                data_task = None

            sess.run(data_task.initializer)
            next_elem_feed = data_task.get_next()

            it = 0
            time_it_start = time.time()
            print("[{:.1f}|epoch {:04d}] {:s}".format(time_it_start, epoch + 1, task.upper()))
            task_errors = list()
            while True:
                try:
                    feed_scene, lat, long = sess.run(next_elem_feed)
                    feed_track = np.hstack([lat[:, np.newaxis], long[:, np.newaxis]])
                    feed_track = feed_track[:, np.newaxis, :]

                    feed_dict = dict({X: feed_scene, Y: feed_track})
                    output = sess.run(net_step, feed_dict=feed_dict)
                    loss = output[1] if task == "train" else output

                    if task == "train":
                        print("[{:.1f}|epoch {:04d}|iter {:04d}]: RMSE {:.3f}".format(
                            time.time(), epoch + 1, it + 1, loss))
                    else:
                        pass

                    task_errors.append(loss)

                    it += 1
                    if task == "train":
                        if it_global % FLAGS.summary_step == 0:
                            tb_writer.add_summary(output[-1], it_global)
                        else:
                            pass
                        it_global += 1
                    else:
                        pass
                except tf.errors.OutOfRangeError:
                    break
            time_it_end = time.time()
            print("[{:.1f}|epoch {:04d}] {:s} mean error: RMSE {:.3f}, StdDev: {:.3f}".format(
                time_it_end, epoch + 1, task.upper(), np.mean(task_errors), np.std(task_errors)))
            print("[{:.1f}|epoch {:04d}] {:s} elapsed time: {:.1f} s".format(
                time_it_end, epoch + 1, task.upper(), time_it_end-time_it_start))

            if task == "valid":
                valid_error_writer.write("{:f},{:f}\n".format(np.mean(task_errors), np.std(task_errors)))
                valid_error_writer.flush()
            elif task == "test":
                test_error_writer.write("{:f},{:f}\n".format(np.mean(task_errors), np.std(task_errors)))
                test_error_writer.flush()
            else:
                pass

        # save model per 1 epoch
        saved_path = saver.save(sess, save_path, global_step=epoch + latest_ckpt_step)
        print("Model %s-%d saved as %s" % (FLAGS.model_name, epoch + latest_ckpt_step, saved_path))

    # close valid/test_error_writer instance
    valid_error_writer.close()
    test_error_writer.close()

    # close tf session
    sess.close()


if __name__ == '__main__':
    # execute
    tf.app.run()
