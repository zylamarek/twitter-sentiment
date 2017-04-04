"""
The EvaluationHelper class implements methods for RNN evaluation.

f_eval - function accepting (x_batch, y_batch, x_mask_batch) as arguments and returning (crossentropy, softmax_output).

All val_id, test_id and eval_ids are evaluated, in order of ascending ids. If you wish to evaluate the training
datasets - include them in the eval_ids.

batch_mul indicates how many batches should be evaluated at one time.

"""

from __future__ import print_function, division
import numpy as np

__all__ = [
    "EvaluationHelper"
]


class EvaluationHelper:
    def __init__(self, f_eval, data, val_id, test_id=None, eval_ids=None, print_progress=True, tol=1e-7, batch_mul=5):

        self.f_eval = f_eval
        self.data = data

        if not isinstance(val_id, (int, long)):
            raise ValueError('Validation id must be an integer')
        self.val_id = val_id
        if not isinstance(test_id, (int, long)) and test_id is not None:
            raise ValueError('Test id must be an integer or None')
        if test_id is None:
            test_id = val_id
        self.test_id = test_id
        if isinstance(eval_ids, (int, long)) or eval_ids is None:
            eval_ids = [eval_ids]
        self.eval_ids = sorted(set([_id for _id in [val_id] + [test_id] + eval_ids if _id is not None]))
        if not all([0 <= _id < self.data.n_datasets for _id in self.eval_ids]):
            raise ValueError('Incorrect ids')
        self.i_val = self.eval_ids.index(self.val_id)
        self.i_test = self.eval_ids.index(self.test_id)

        self.print_progress = print_progress
        self.tol = tol

        self.batch_mul = batch_mul

        self.reset()

    def reset(self):
        # Metrics
        self.ce = []  # crossentropy
        self.acc = []  # accuracy
        self.F = []  # F^PN_1 score
        self.R = []  # recall
        self.CM = []  # confusion matrix

        # Best validation values with corresponding test values, in form [epoch, valid, test]
        self.best_ce = [-1, 1.e9, 1.e9]
        self.best_acc = [-1, 0., 0.]
        self.best_F = [-1, 0., 0.]
        self.best_R = [-1, 0., 0.]

    @property
    def measurements(self):
        return {'ce': self.ce, 'acc': self.acc, 'F': self.F, 'R': self.R, 'CM': self.CM}

    def get_last_measurements(self):
        if len(self.ce) > 0:
            return None, None, None, None, None
        return self.ce[-1], self.acc[-1], self.F[-1], self.R[-1], self.CM[-1]

    def evaluate(self):
        self.ce.append([])
        self.acc.append([])
        self.F.append([])
        self.R.append([])
        self.CM.append([])

        for data_id in self.eval_ids:
            _ce, _acc, _F, _R, _CM = self.__evaluate_dataset(data_id)

            self.ce[-1].append(_ce)
            self.acc[-1].append(_acc)
            self.F[-1].append(_F)
            self.R[-1].append(_R)
            self.CM[-1].append(_CM)

        # Update best values
        n_measurement = len(self.ce) - 1
        new_ce = [n_measurement, self.ce[-1][self.i_val], self.ce[-1][self.i_test]]
        self.best_ce = self.best_ce if self.best_ce[1] <= new_ce[1] else new_ce
        new_acc = [n_measurement, self.acc[-1][self.i_val], self.acc[-1][self.i_test]]
        self.best_acc = self.best_acc if self.best_acc[1] >= new_acc[1] else new_acc
        new_F = [n_measurement, self.F[-1][self.i_val], self.F[-1][self.i_test]]
        self.best_F = self.best_F if self.best_F[1] >= new_F[1] else new_F
        new_R = [n_measurement, self.R[-1][self.i_val], self.R[-1][self.i_test]]
        self.best_R = self.best_R if self.best_R[1] >= new_R[1] else new_R

    def __evaluate_dataset(self, data_id):
        char_num, ce, acc = 0., 0., 0.
        counts = [0.] * self.data.n_labels * self.data.n_labels
        n = self.data.n_labels
        cm_order = [n * n - 1 - i // n - (i % n) * n for i in range(n * n)]

        if self.print_progress:
            print('evaluate ' + self.data.dataset_names[data_id] + ': ', end='')
        step = max(self.data.n_batches[data_id] // 10 + 1, 1)
        offset = step * 10 - self.data.n_batches[data_id] + 1
        if self.print_progress and self.data.n_batches[data_id] < 10:
            print('.' * (10 - self.data.n_batches[data_id]), end='')
        self.data.set_current_data(data_id)

        def big_batch_data():
            def reset_big_batch():
                x_big = np.zeros((self.batch_mul * self.data.batch_size, self.data.max_len, self.data.charset_size),
                                 dtype=np.uint32)
                x_mask_big = np.zeros((self.batch_mul * self.data.batch_size, self.data.max_len), dtype=np.uint32)
                y_big = np.zeros((self.batch_mul * self.data.batch_size), dtype=np.uint32)
                return x_big, x_mask_big, y_big

            x_batch_big, x_mask_batch_big, y_batch_big = reset_big_batch()
            for i_batch, (x_batch, x_mask_batch, y_batch) in enumerate(self.data):
                i_sub_batch = i_batch % self.batch_mul
                batch_slice = range(i_sub_batch * self.data.batch_size, (i_sub_batch + 1) * self.data.batch_size)

                x_batch_big[batch_slice] = x_batch
                x_mask_batch_big[batch_slice] = x_mask_batch
                y_batch_big[batch_slice] = y_batch

                if not (i_batch + 1) % self.batch_mul:
                    yield (x_batch_big, x_mask_batch_big, y_batch_big)
                    x_batch_big, x_mask_batch_big, y_batch_big = reset_big_batch()

            if self.data.n_batches[data_id] % self.batch_mul:
                yield (x_batch_big, x_mask_batch_big, y_batch_big)

        for i_batch, (x_batch, x_mask_batch, y_batch) in enumerate(big_batch_data()):
            # get output
            cost, out = self.f_eval(x_batch, y_batch, x_mask_batch)
            predicted_labels = np.argmax(out, axis=-1).flatten()
            mask_tweet = np.sign(np.sum(x_mask_batch, axis=-1))  # some batches might not be full

            # update crossentropy and accuracy vars
            char_num += np.sum(x_mask_batch)
            ce += cost
            acc += np.sum(np.equal(predicted_labels, y_batch) * mask_tweet)

            # create confusion matrix for current batch
            mask_cm = (np.ones_like(mask_tweet) - mask_tweet) * self.data.n_labels * self.data.n_labels
            labels, label_counts = np.unique(y_batch * self.data.n_labels + predicted_labels + mask_cm,
                                             return_counts=True)
            for i, c in enumerate(labels):
                if c < self.data.n_labels * self.data.n_labels and i < self.data.n_labels * self.data.n_labels:
                    counts[cm_order[c]] += label_counts[i]

            if self.print_progress and not (i_batch + offset) % step:
                print('.', end='')

        if self.print_progress:
            print('')

        # compute F^PN_1 and recall
        F_pn, recall = 0., 0.
        if self.data.n_labels == 3:
            PP, PU, PN, UP, UU, UN, NP, NU, NN = counts
            pi_p = PP / (PP + PU + PN + self.tol)
            rho_p = PP / (PP + UP + NP + self.tol)
            F_p = 2. * pi_p * rho_p / (rho_p + pi_p)
            pi_n = NN / (NN + NU + NP + self.tol)
            rho_n = NN / (NN + UN + PN + self.tol)
            F_n = 2. * pi_n * rho_n / (rho_n + pi_n)
            F_pn = (F_n + F_p) / 2.
            rho_u = UU / (NU + UU + PU + self.tol)
            recall = (rho_p + rho_u + rho_n) / 3.

        # compute crossentropy and accuracy
        ce = ce / char_num
        acc = acc * 100. / self.data.n_tweets[data_id]

        return ce, acc, F_pn, recall, counts

    def conf_matrix_to_str(self, conf_matrix):
        conf_str = ''
        for i_c, c in enumerate(conf_matrix):
            if (i_c + 1) % self.data.n_labels == 0:
                conf_str += '%.0f\n' % c
            else:
                conf_str += '%.0f\t' % c
        return conf_str.expandtabs(8)[:-1]
