import torch
import torch.optim as optim
from torch.autograd import Variable
import pickle
import itertools
from utils.mixup import shuffle_minibatch
import numpy as np
import torch.nn as nn
import os


class Trainer:
    def __init__(self, args, model, criterion, logger):
        self.args = args
        self.decay = 1
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.SGD(
            model.parameters(),
            args.learn_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True)
        #self.optimizer = optim.Adam(model.parameters())
        self.nGPU = args.nGPU
        self.learn_rate = args.learn_rate
        self.architecture = args.model
        self.data_size = args.data_size

    def train(self, epoch, train_loader):
        n_batches = len(train_loader)

        acc_avg = 0
        acc_top3_avg = 0

        loss_avg = 0
        total = 0

        model = self.model
        model.train()
        self.learning_rate(epoch)

        for i, (input_tensor, target) in itertools.islice(train_loader,stop=self.args.data_size):

            # print(target)
            if self.args.mixup:
                # input_tensor = np.transpose(input_tensor, (0, 2, 3, 1))
                input_tensor, target = shuffle_minibatch(input_tensor, target, self.args, mixup=True)
                # input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
            # print(target)
            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            target_var = Variable(target)

            output = model(input_var)

            if self.args.mixup:
                m = nn.LogSoftmax(dim=1)

                loss = -m(output) * target
                loss = torch.sum(loss) / 128

                _, target = torch.max(target.data, 1)
            else:
                loss = self.criterion(output, target_var)

            acc, acc_top3 = self.accuracy(output.data, target, (1, 3))

            acc_avg += acc * batch_size
            acc_top3_avg += acc_top3 * batch_size

            loss_avg += loss.item() * batch_size
            total += batch_size

            print("| Epoch[%d] [%d/%d]  Loss %1.4f  Acc %6.3f  Acc-Top3 %6.3f LR %1.8f" % (
                epoch,
                i + 1,
                n_batches,
                loss.item(),
                acc,
                acc_top3,
                self.decay * self.learn_rate))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_avg /= total
        acc_avg /= total
        acc_top3_avg /= total

        print("\n=> Epoch[%d]  Loss: %1.4f  Acc %6.3f  Acc-Top3 %6.3f\n" % (
            epoch,
            loss_avg,
            acc_avg,
            acc_top3_avg))

        torch.cuda.empty_cache()

        summary = dict()

        summary['acc'] = acc_avg
        summary['acc-top3'] = acc_top3_avg
        summary['loss'] = loss_avg

        return summary

    def test(self, epoch, test_loader):

        targets = []
        outputs = []

        n_batches = len(test_loader)

        acc_avg = 0
        acc_top3_avg = 0

        total = 0

        model = self.model
        model.eval()
        out_f = open('results.txt', 'w')
        for i, (input_tensor, target) in enumerate(test_loader):

            if self.nGPU > 0:
                input_tensor = input_tensor.cuda()
                target = target.cuda(async=True)

            batch_size = target.size(0)
            input_var = Variable(input_tensor)
            # target_var = Variable(target)

            output = model(input_var)

            # if self.args.save_result:
            #     _, predictions = output.topk(3, dim=1, )
            #     for filename, prediction in zip(filenames, predictions):
            #         out_f.write(str(os.path.basename(filename)).split('.')[0]+','+','.join([str(int(x)) for x in prediction])+'\n')


            acc, acc_top3 = self.accuracy(output.data, target, (1, 3))

            acc_avg += acc * batch_size
            acc_top3_avg += acc_top3 * batch_size

            total += batch_size

            print("| Test[%d] [%d/%d]   Acc %6.3f  Acc-Top3 %6.3f" % (
                epoch,
                i + 1,
                n_batches,
                acc,
                acc_top3))

        acc_avg /= total
        acc_top3_avg /= total

        print("\n=> Test[%d]  Acc %6.3f  Acc-Top3 %6.3f\n" % (
            epoch,
            acc_avg,
            acc_top3_avg))

        torch.cuda.empty_cache()


        summary = dict()

        summary['acc'] = acc_avg
        summary['acc-top3'] = acc_top3_avg
        return summary

    # def accuracy(self, output, target):
    #
    #     batch_size = target.size(0)
    #
    #     _, pred = torch.max(output, 1)
    #
    #     correct = pred.eq(target).float().sum(0)
    #
    #     correct.mul_(100. / batch_size)
    #
    #     return correct[0]

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # print(correct_k)

            res.append(correct_k.mul_(100.0 / batch_size)[0])
        return res

    def learning_rate(self, epoch):
        self.decay = 0.1 ** ((epoch - 1) // self.args.decay)
        learn_rate = self.learn_rate * self.decay
        if learn_rate < 1e-7:
            learn_rate = 1e-7
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learn_rate
