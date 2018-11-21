import tensorflow as tf
from keras.layers import Dense, Reshape, Flatten, Conv2D

from toolkit.dirty import summary_filters
from toolkit.loss import root_mean_sqrt_error
from toolkit.synapse import MaxPool, ResolvedSyn, InceptionUnit


class ImgLinRegCplx(object):

    _x = None
    _y = None
    _opt = None
    _conv_act_policy = None
    _flat_act_policy = None
    _batch_norm = False
    _net = None
    _summary = None
    _loss = None
    _step = None

    def __init__(self, x=None, y=None, conv_act_policy='relu', flat_act_policy='sigmoid',
                 batch_norm=False, optimizer=None, learning_rate=1e-5):
        if x is not None or y is not None:
            # set input/output placeholder
            self._x = x
            self._y = y

            # set activation policy (conv/flat)
            self._conv_act_policy = conv_act_policy
            self._flat_act_policy = flat_act_policy
            print("[I] ImgLinRegCplx.__init__(): ConvActivation set as %s" % conv_act_policy)
            print("[I] ImgLinRegCplx.__init__(): FlatActivation set as %s" % flat_act_policy)

            # set batch normalization feature
            self._batch_norm = batch_norm

            # optimizer initialization w/ learning rate
            if optimizer is not None and issubclass(optimizer, tf.train.Optimizer):
                opt_resolve = optimizer
            else:
                opt_resolve = tf.train.GradientDescentOptimizer
            self._opt = opt_resolve(learning_rate=learning_rate)
            print("[I] ImgLinRegCplx.__init__(): optimizer set as %s" % opt_resolve.__name__)

            # start topology setup
            self.setup()

        else:
            # if there is no input, do nothing
            pass

    def setup(self):

        x = self._x
        y = self._y

        print(x)
        net = ResolvedSyn(x=Conv2D(filters=16, kernel_size=[3, 3], padding='same')(x),
                          type=self._conv_act_policy)
        net = MaxPool(x=net)
        summary_filters(namespace='conv1', tensor=net, num_of_filters=4)

        net = ResolvedSyn(x=Conv2D(filters=64, kernel_size=[3, 3], padding='same')(net),
                          type=self._conv_act_policy)
        net = MaxPool(x=net)
        summary_filters(namespace='conv2', tensor=net, num_of_filters=8)

        net = InceptionUnit(x=net, scale=16, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception1')
        net = InceptionUnit(x=net, scale=32, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception2')
        net = InceptionUnit(x=net, scale=48, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception3')
        net = InceptionUnit(x=net, scale=64, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception4')
        net = InceptionUnit(x=net, scale=80, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception5')
        net = InceptionUnit(x=net, scale=96, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception6')
        net = InceptionUnit(x=net, scale=128, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception7')
        net = InceptionUnit(x=net, scale=192, batch_norm=self._batch_norm,
                            activation=self._conv_act_policy, namespace='inception8')

        net = Flatten()(net)
        net = ResolvedSyn(x=Dense(units=8192)(net), type=self._flat_act_policy)
        net = ResolvedSyn(x=Dense(units=1024)(net), type=self._flat_act_policy)
        net = Dense(units=128, activation=self._flat_act_policy)(net)
        net = Dense(units=2, activation='sigmoid')(net)
        # net = Reshape([1, 2])(net)
        self._net = net

        rmse = root_mean_sqrt_error(labels=y, predictions=net)
        tf.summary.scalar('rmse', rmse)
        self._loss = rmse

        self._step = self._opt.minimize(self._loss)
        self._summary = tf.summary.merge_all()

    def train(self):
        return [self._step, self._loss, self._net, self._summary]

    def valid(self):
        return self._loss

    def feed(self):
        return self._net
