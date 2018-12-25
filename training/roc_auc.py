import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
from functools import reduce
from chainer import function
import six
import os
import numpy as np
from chainermn import CommunicatorBase
from chainer import configuration
import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer.training.extensions import Evaluator
from sklearn import metrics

import chainer.functions as F

from chainer import reporter as reporter_module


def _get_1d_numpy_array(v):
    """Convert array or Variable to 1d numpy array
    Args:
        v (numpy.ndarray or cupy.ndarray or chainer.Variable): array to be
            converted to 1d numpy array
    Returns (numpy.ndarray): Raveled 1d numpy array
    """
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()


def _to_list(a):
    """convert value `a` to list
    Args:
        a: value to be convert to `list`
    Returns (list):
    """
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


def plot_roc(y_true, y_score, out_name):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score)
    auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % auc)
    plt.legend()
    plt.title('ROC curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.grid(True)
    plt.savefig(out_name)


class SingleAUCEvaluator(Evaluator):
    """Evaluator which calculates ROC AUC score
    Note that this Evaluator is only applicable to binary classification task.
    Args:
        iterator: Dataset iterator for the dataset to calculate ROC AUC score.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.
    """

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None, rank=None):
        super(SingleAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_roc(trainer=trainer)

        reporter_module.report(result)
        return result
        self.rank = rank

    def evaluate_roc(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = []
        t_total = []
        length = it.dataset.__len__()
        batchsize = it.batch_size
        length = length // batchsize
        from tqdm import tqdm
        pbar = tqdm(total=length)
        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func(*in_arrays[:-1])
                t = in_arrays[-1]
            y_data = cuda.to_cpu(y.data)
            t_data = cuda.to_cpu(t)
            y_total.extend(y_data)
            t_total.extend(t_data)
            pbar.update(1)
        y_total = numpy.concatenate(y_total).ravel()
        t_total = numpy.concatenate(t_total).ravel()
        index = numpy.where(t_total != -1)[0]
        y_total = y_total[index]
        t_total = t_total[index]
        d = {'label': t_total, 'score': y_total}
        np.save('validation_' + str(self.rank), **d)
        observation = {}
        with reporter.report_scope(observation):
            roc_auc = metrics.roc_auc_score(t_total, F.sigmoid(y_total).data)
            with reporter.report_scope(observation):
                reporter.report({'roc_auc_': roc_auc}, self._targets['main'])
                reporter.report({'loss': F.sigmoid_cross_entropy(y_total, t_total).data},
                                self._targets['main'])
                reporter.report({'accuracy': F.binary_accuracy(y_total, t_total).data}, self._targets['main'])
        return observation


class ROCAUCEvaluator(Evaluator):
    """Evaluator which calculates ROC AUC score
    Note that this Evaluator is only applicable to binary classification task.
    Args:
        iterator: Dataset iterator for the dataset to calculate ROC AUC score.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.
    """

    def __init__(self, iterator, target, comm, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None):
        super(ROCAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.rank = comm.rank
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.comm = comm

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_roc(trainer=trainer)

        reporter_module.report(result)
        return result

    def evaluate_roc(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = []
        t_total = []

        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func(*in_arrays[:-1])
                t = in_arrays[-1]
            y_data = _get_1d_numpy_array(F.softmax(y)[:, 1])
            # y_data = _get_1d_numpy_array(F.sigmoid(y)[:, 0])
            t_data = _get_1d_numpy_array(t)
            y_total.append(y_data)
            t_total.append(t_data)
        y_total = numpy.concatenate(y_total).ravel()
        t_total = numpy.concatenate(t_total).ravel()
        index = numpy.where(t_total != -1)[0]
        y_total = y_total[index]
        t_total = t_total[index]
        updater = trainer.updater
        epoch = updater.epoch
        out_dir = trainer.out
        out_name = os.path.join(out_dir, str(epoch) + 'epoch_roc.pdf')
        gather_data = self.comm.gather(np.vstack([t_total, y_total]))
        if self.rank == 0:
            gather_data = reduce(lambda x, y: np.hstack([x, y]), gather_data)
            gather_t = np.array(gather_data[0])
            gather_y = np.array(gather_data[1])
            plot_roc(y_true=gather_t, y_score=gather_y, out_name=out_name)
        roc_auc = metrics.roc_auc_score(t_total, y_total)
        observation = {}
        with reporter.report_scope(observation):
            reporter.report({'roc_auc': roc_auc}, self._targets['main'])
        return observation


class Multi_label_ROCAUCEvaluator(Evaluator):
    """Evaluator which calculates ROC AUC score
    Note that this Evaluator is only applicable to binary classification task.
    Args:
        iterator: Dataset iterator for the dataset to calculate ROC AUC score.
            It can also be a dictionary of iterators. If this is just an
            iterator, the iterator is registered by the name ``'main'``.
        target: Link object or a dictionary of links to evaluate. If this is
            just a link object, the link is registered by the name ``'main'``.
        converter: Converter function to build input arrays and true label.
            :func:`~chainer.dataset.concat_examples` is used by default.
            It is expected to return input arrays of the form
            `[x_0, ..., x_n, t]`, where `x_0, ..., x_n` are the inputs to
            the evaluation function and `t` is the true label.
        device: Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        eval_hook: Function to prepare for each evaluation process. It is
            called at the beginning of the evaluation. The evaluator extension
            object is passed at each call.
        eval_func: Evaluation function called at each iteration. The target
            link to evaluate as a callable is used by default.
        name (str): name of this extension. When `name` is None,
            `default_name='validation'` which is defined in super class
            `Evaluator` is used as extension name. This name affects to the
            reported key name.
        pos_labels (int or list): labels of the positive class, other classes
            are considered as negative.
        ignore_labels (int or list or None): labels to be ignored.
            `None` is used to not ignore all labels.
    Attributes:
        converter: Converter function.
        device: Device to which the training data is sent.
        eval_hook: Function to prepare for each evaluation process.
        eval_func: Evaluation function called at each iteration.
        pos_labels (list): labels of the positive class
        ignore_labels (list): labels to be ignored.
    """

    def __init__(self, iterator, target, comm, label_name, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, name=None,
                 pos_labels=1, ignore_labels=None):
        super(Multi_label_ROCAUCEvaluator, self).__init__(
            iterator, target, converter=converter, device=device,
            eval_hook=eval_hook, eval_func=eval_func)
        self.rank = comm.rank
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.comm = comm
        self.label_name = label_name

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_roc(trainer=trainer)

        reporter_module.report(result)
        return result

    def evaluate_roc(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        y_total = np.array([]).reshape([0, len(self.label_name)])
        t_total = np.array([]).reshape([0, len(self.label_name)])
        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                y = eval_func(*in_arrays[:-1])
                t = in_arrays[-1]
            # y = F.sigmoid(y)
            y_data = cuda.to_cpu(y.data)
            t_data = cuda.to_cpu(t)
            y_total = np.vstack([y_total, y_data])
            t_total = np.vstack([t_total, t_data])
        updater = trainer.updater
        epoch = updater.iteration
        out_dir = trainer.out
        observation = {}
        for label_index, label in enumerate(self.label_name):
            y = y_total[:, label_index]
            t = t_total[:, label_index]
            index = numpy.where(t != -1)[0]
            y = y[index]
            t = t[index]
            out_name = os.path.join(out_dir, str(epoch) + 'iteration_' + label + '_roc.pdf')
            gather_data = self.comm.gather(np.vstack([t, y]))
            if self.rank == 0:
                gather_data = reduce(lambda x, y: np.hstack([x, y]), gather_data)
                gather_t = np.array(gather_data[0], dtype=np.int)
                gather_y = np.array(gather_data[1], dtype=np.float32)

                plot_roc(y_true=gather_t, y_score=F.sigmoid(gather_y).data, out_name=out_name)
                roc_auc = metrics.roc_auc_score(gather_t, F.sigmoid(gather_y).data)
                with reporter.report_scope(observation):
                    reporter.report({'roc_auc_' + label: roc_auc}, self._targets['main'])
                    reporter.report({'loss': F.sigmoid_cross_entropy(gather_y, gather_t).data},
                                    self._targets['main'])
                    reporter.report({'accuracy': F.binary_accuracy(gather_y, gather_t).data}, self._targets['main'])
        return observation
