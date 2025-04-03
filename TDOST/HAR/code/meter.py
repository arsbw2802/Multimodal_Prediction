import numpy as np


class RunningMeter:
    def __init__(self, args):
        # Tracking at a per epoch level
        self.loss = {'train': [], 'val': [], 'test': [], 'tl': []}
        self.accuracy = {'train': [], 'val': [], 'test': [], 'tl': []}
        self.f1_score = {'train': [], 'val': [], 'test': [], 'tl': []}
        self.f1_score_weighted = {'train': [], 'val': [], 'test': [], 'tl': []}
        self.confusion_matrix = {'train': [], 'val': [], 'test': [], 'tl': []}

        self.recall = {'train': [], 'val': [], 'test': [], 'tl': []}
        self.recall_weighted = {'train': [], 'val': [], 'test': [], 'tl': []}

        self.precision = {'train': [], 'val': [], 'test': [], 'tl': []}
        self.precision_weighted = {'train': [], 'val': [], 'test': [], 'tl': []}

        self.epochs = np.arange(0, args.num_epochs)

        self.best_meter = BestMeter()

        self.args = args

    def update(self, phase, loss, accuracy, f1_score,f1_score_weighted,
               confusion_matrix, recall,recall_weighted, precision,precision_weighted):
        # Update the metrics for every phase
        self.loss[phase].append(loss)
        self.accuracy[phase].append(accuracy)
        self.f1_score[phase].append(f1_score)
        self.f1_score_weighted[phase].append(f1_score_weighted)

        self.recall[phase].append(recall)
        self.recall_weighted[phase].append(recall_weighted)

        self.precision[phase].append(precision)
        self.precision_weighted[phase].append(precision_weighted)

        self.confusion_matrix[phase].append(confusion_matrix)

    def get(self):
        return self.loss, self.accuracy, self.f1_score, \
               self.f1_score_weighted, self.confusion_matrix, self.recall, self.precision, self.recall_weighted, self.precision_weighted, self.epochs


    def update_best_meter(self, best_meter):
        self.best_meter = best_meter


class BestMeter:
    def __init__(self):
        # Storing the best values
        self.loss = {'train': np.inf, 'val': np.inf, 'test': np.inf, 'tl': np.inf}
        self.accuracy = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        self.f1_score = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        self.f1_score_weighted = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        self.confusion_matrix = {'train': [], 'val': [], 'test': [], 'tl': []}

        self.recall = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        self.precision = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        
        self.recall_weighted = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        self.precision_weighted = {'train': 0.0, 'val': 0.0, 'test': 0.0, 'tl': 0.0}
        self.epoch = 0

        # For cross validation, we can track the test split predictions and
        # gt to compute the f1-score at the end
        self.preds = []
        self.gt = []

    def update(self, phase, loss, accuracy, f1_score, f1_score_weighted,
               confusion_matrix, recall,  recall_weighted, precision, precision_weighted, epoch):
        self.loss[phase] = loss
        self.accuracy[phase] = accuracy
        self.f1_score[phase] = f1_score
        self.f1_score_weighted[phase] = f1_score_weighted
        self.confusion_matrix[phase] = confusion_matrix

        self.recall[phase] =recall
        self.precision[phase] =precision
        
        self.recall_weighted[phase] =recall_weighted
        self.precision_weighted[phase] = precision_weighted


        self.epoch = epoch

    def get(self):
        return self.loss, self.accuracy, self.f1_score, \
               self.f1_score_weighted, self.confusion_matrix, self.recall, self.precision, self.recall_weighted, self.precision_weighted, self.epoch

    def display(self):
        print('The best epoch is {}'.format(self.epoch))
        for phase in ['train', 'val', 'test']:
            print('Phase: {}, loss: {}, accuracy: {}, f1_score: {}, f1_score '
                  'weighted: {}, recall: {}, precision: {}, recall_weighted: {}, precision_weighted: {}'
                  .format(phase, self.loss[phase], self.accuracy[phase],
                          self.f1_score[phase], self.f1_score_weighted[phase], self.recall[phase], self.recall_weighted[phase],
                          self.precision[phase], self.precision_weighted[phase] 
                          ))
