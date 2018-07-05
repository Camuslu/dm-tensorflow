import numpy as np

def prec(correctCount, predictedCount):
    if predictedCount == 0:
        return np.nan
    return np.round(correctCount/predictedCount*100.0, 2)

def recall(correctCount, trueCount):
    if trueCount == 0:
        return np.nan
    return np.round(correctCount/trueCount*100.0, 2)

def f1(prec, recall):
    if prec ==0 and recall == 0:
        return np.nan
    return np.round(2*prec*recall/(prec+recall), 2)

class Evaluator(object):

    def __init__(self, values):
        self.n = len(values)
        self.correct = 0
        self.predicted = 0
        self.true = 0
        self.noprediction = 0
        self.total = 0
        self.corrects = np.zeros(self.n)
        self.predicteds = np.zeros(self.n)
        self.trues = np.zeros(self.n)
        self.values = values

    def evaluate(self, target: np.array, predicted: np.array):
        print(target.shape)
        print(predicted.shape)
        compare = ((target == predicted) & predicted)
        self.correct += np.count_nonzero(compare)
        self.predicted += np.count_nonzero(predicted)
        self.true += np.count_nonzero(target)
        self.noprediction += np.count_nonzero((np.add.reduce(target, 1) > 0) == 0)
        self.total += target.shape[0]
        self.corrects += np.add.reduce(compare, 0)
        self.predicteds += np.add.reduce(predicted, 0)
        self.trues += np.add.reduce(target, 0)

    def result(self):
        recall = self.correct / self.true
        prec = self.correct / self.predicted
        f1 = 2 * (prec * recall) / (prec + recall)
        recalls = self.corrects / self.trues
        precs = self.corrects / self.predicteds
        f1s = 2 * (precs * recalls) / (precs + recalls)
        percent_filtered = self.noprediction / self.total
        out = 'Category\tPrec\tRecall\tF1\n'
        out += 'TOTAL\t{:0.4f}\t{:0.4f}\t{:0.4f}\treduction\t{:0.4f}\n'\
            .format(prec, recall, f1, percent_filtered)
        for i in range(self.n):
            out += '{}\t{:0.4f}\t{:0.4f}\t{:0.4f}\n' \
                .format(self.values[i], precs[i], recalls[i], f1s[i])
        return out

class Evaluator(object):

    def __init__(self, values):
        self.n = len(values)
        self.correct = 0
        self.predicted = 0
        self.true = 0
        self.noprediction = 0
        self.total = 0
        self.corrects = np.zeros(self.n)
        self.predicteds = np.zeros(self.n)
        self.trues = np.zeros(self.n)
        self.values = values

    def evaluate(self, target: np.array, predicted: np.array):
        print(target.shape)
        print(predicted.shape)
        compare = ((target == predicted) & predicted)
        self.correct += np.count_nonzero(compare)
        self.predicted += np.count_nonzero(predicted)
        self.true += np.count_nonzero(target)
        self.noprediction += np.count_nonzero((np.add.reduce(target, 1) > 0) == 0)
        self.total += target.shape[0]
        self.corrects += np.add.reduce(compare, 0)
        self.predicteds += np.add.reduce(predicted, 0)
        self.trues += np.add.reduce(target, 0)

    def result(self):
        recall = self.correct / self.true
        prec = self.correct / self.predicted
        f1 = 2 * (prec * recall) / (prec + recall)
        recalls = self.corrects / self.trues
        precs = self.corrects / self.predicteds
        f1s = 2 * (precs * recalls) / (precs + recalls)
        percent_filtered = self.noprediction / self.total
        out = 'Category\tPrec\tRecall\tF1\n'
        out += 'TOTAL\t{:0.4f}\t{:0.4f}\t{:0.4f}\treduction\t{:0.4f}\n'\
            .format(prec, recall, f1, percent_filtered)
        for i in range(self.n):
            out += '{}\t{:0.4f}\t{:0.4f}\t{:0.4f}\n' \
                .format(self.values[i], precs[i], recalls[i], f1s[i])
        return out

