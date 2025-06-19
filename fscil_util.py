import os
import torch
import pickle
import shutil
import random
import numpy as np
import torch.nn as nn
import os.path as osp
from functools import partial
from collections import OrderedDict, defaultdict
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import f1_score, confusion_matrix


class AverageMeter:
    """Compute and store the average and current value.

    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self, ema=False):
        """
        Args:
            ema (bool, optional): apply exponential moving average.
        """
        self.ema = ema
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()

        self.val = val
        self.sum += val * n
        self.count += n

        if self.ema:
            self.avg = self.avg * 0.9 + self.val * 0.1
        else:
            self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.

    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter=" "):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                "Input to MetricMeter.update() must be a dictionary"
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(f"{name} {meter.val:.4f} ({meter.avg:.4f})")
        return self.delimiter.join(output_str)


class _BaseWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        last_epoch=-1,
        verbose=False
    ):
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

    pass


class ConstantWarmupScheduler(_BaseWarmupScheduler):

    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch,
        cons_lr,
        last_epoch=-1,
        verbose=False
    ):
        self.cons_lr = cons_lr
        super().__init__(
            optimizer, successor, warmup_epoch, last_epoch, verbose
        )

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]

    pass


class Evaluator(object):
    """Evaluator for classification."""

    def __init__(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        pass

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        pass

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        pass

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(self._y_true, self._y_pred, average="macro", labels=np.unique(self._y_true))

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["total"] = self._total
        results["correct"] = self._correct

        print(f"=> result * total: {self._total:,} * correct: {self._correct:,} * "
              f"accuracy: {acc:.1f}% * error: {err:.1f}% * macro_f1: {macro_f1:.1f}%")
        return results

    pass


class EvaluatorPart(object):
    """Evaluator for classification."""

    def __init__(self, split_1=60, split_2=65):
        self.split_1 = split_1
        self.split_2 = split_2
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        pass

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        pass

    def process(self, mo, gt):
        which = (self.split_1 <= gt) & (gt < self.split_2)
        mo = mo[which]
        gt = gt[which]
        if mo is not None and len(mo) > 0:
            pred = mo.max(1)[1]
            matches = pred.eq(gt).float()
            self._correct += int(matches.sum().item())
            self._total += gt.shape[0]

            self._y_true.extend(gt.data.cpu().numpy().tolist())
            self._y_pred.extend(pred.data.cpu().numpy().tolist())
            pass
        pass

    def evaluate(self):
        results = OrderedDict()
        acc, err, macro_f1 = 0.0, 0.0, 0.0
        if self._total > 0:
            acc = 100.0 * self._correct / self._total
            err = 100.0 - acc
            macro_f1 = 100.0 * f1_score(self._y_true, self._y_pred, average="macro", labels=np.unique(self._y_true))
            pass

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1
        results["total"] = self._total
        results["correct"] = self._correct

        print(f"=> result * total: {self._total:,} * correct: {self._correct:,} * "
              f"accuracy: {acc:.1f}% * error: {err:.1f}% * macro_f1: {macro_f1:.1f}%")
        return results

    pass


class UtilTool(object):

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        pass

    @staticmethod
    def compute_accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for
        the specified values of k.

        Args:
            output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
            target (torch.LongTensor): ground truth labels with shape (batch_size).
            topk (tuple, optional): accuracy at top-k will be computed. For example,
                topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

        Returns:
            list: accuracy at top-k.
        """
        maxk = max(topk)
        batch_size = target.size(0)

        if isinstance(output, (tuple, list)):
            output = output[0]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(100.0 / batch_size)
            res.append(acc)

        return res

    @staticmethod
    def load_checkpoint(fpath):
        r"""Load checkpoint.

        ``UnicodeDecodeError`` can be well handled, which means
        python2-saved files can be read from python3.

        Args:
            fpath (str): path to checkpoint.

        Returns:
            dict

        Examples::
            >>> fpath = 'log/my_model/model.pth.tar-10'
            >>> checkpoint = load_checkpoint(fpath)
        """
        if fpath is None:
            raise ValueError("File path is None")

        if not osp.exists(fpath):
            raise FileNotFoundError('File is not found at "{}"'.format(fpath))

        map_location = None if torch.cuda.is_available() else "cpu"

        try:
            checkpoint = torch.load(fpath, map_location=map_location)

        except UnicodeDecodeError:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
            checkpoint = torch.load(
                fpath, pickle_module=pickle, map_location=map_location
            )

        except Exception:
            print('Unable to load checkpoint from "{}"'.format(fpath))
            raise

        return checkpoint

    @staticmethod
    def save_checkpoint(state, save_dir, is_best=False, remove_module_from_keys=True, model_name=""):
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        if remove_module_from_keys:
            # remove 'module.' in state_dict's keys
            state_dict = state["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            state["state_dict"] = new_state_dict

        # save model
        epoch = state["epoch"]
        if not model_name:
            model_name = "model.pth.tar-" + str(epoch)
        fpath = osp.join(save_dir, model_name)
        torch.save(state, fpath)
        print(f"Checkpoint saved to {fpath}")

        # save current model name
        checkpoint_file = osp.join(save_dir, "checkpoint")
        checkpoint = open(checkpoint_file, "w+")
        checkpoint.write("{}\n".format(osp.basename(fpath)))
        checkpoint.close()

        if is_best:
            best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
            shutil.copy(fpath, best_fpath)
            print('Best checkpoint saved to "{}"'.format(best_fpath))
        pass

    pass
