"""
general rvm classify function
using for multi class classification
train n classifier if there are n classes

main function: rvm

@author: Feiyang Liu

"""
import datetime
import numpy as np
from numpy import random

def preprocess_data(images):
    new_data = np.copy(images)
    new_data /= 255
    new_data -= new_data.mean()
    return new_data


def log_msg(*msg):
    print(datetime.datetime.now(), *msg)

def lin_kernel(x,y):
    return 1 + x*y + x*y*min((x,y))-0.5*(x+y)*min((x,y))**2+(1/3)*min((x,y))**3


def kernel(xm, xn):
    k = (xm @ xn + 1)**3 / 1e3
    return k


def Cap_sigma(A, B, Phi):
    Sigma = (Phi.T @ B @ Phi) + A
    return np.linalg.inv(Sigma)


def w_MP(Sigma, Phi, B, t):
    return Sigma @ Phi.T @ B @ t


def cal_B(Y):
    Y[abs(Y) > 200] = 200
    sigma = 1 / (1 + np.exp(-Y))
    B = sigma * (1 - sigma)
    return np.diag(B)


def cal_Phi(data, sv):
    num_data = data.shape[0]
    num_sv = sv.shape[0]
    rep = np.empty((num_data, num_sv + 1))
    for i, x in enumerate(data):
        rep[i, 0] = 1
        for j, s in enumerate(sv):
            rep[i, j + 1] = kernel(x, s)
    return rep


def update_A(Sigma, w, A):
    n = A.shape[0]
    NOT_ZERO = 10 ** -50
    for i in range(n):
        A[i, i] = (1 - A[i, i] * Sigma[i, i]) / (w[i] ** 2 + NOT_ZERO)

    return A


def cal_correct(label, output):
    return np.count_nonzero((label > 0) == (output > 0))


def cal_out(test, train, w):
    test_num = test.shape[0]
    Y_out = np.empty(test_num)
    n = len(w)
    for i in range(test_num):
        phi = np.empty(n)
        phi[0] = 1
        for j in range(n - 1):
            phi[j + 1] = kernel(test[i], train[j])
        Y_out[i] = phi @ w

    return Y_out


def prune(w, Rec, phi, alpha):
    new_w = w[Rec]
    new_phi = phi[:, Rec]
    new_alpha = alpha[Rec]
    new_A = np.diag(new_alpha)

    return new_w, new_phi, new_A


def train_rvm(phi, t, print_every=100):
    n = len(t)
    A = np.diag(random.rand(n + 1))
    w = np.array(random.rand(n + 1))
    Y = phi @ w

    right_train = 0

    remaining_inx = np.indices(w.shape).reshape(-1, )

    # begin training
    iter = 0
    while (right_train < .97 or len(remaining_inx) > 100) and iter < 1000:
        print(iter)
        B = cal_B(Y)
        Sigma = Cap_sigma(A, B, phi)
        w = w_MP(Sigma, phi, B, t)
        A = update_A(Sigma, w, A)
        alpha = np.diag(A)
        R_vec = np.argwhere(abs(alpha) < 1e12).ravel()
        if R_vec[0] != 0:
            R_vec = [0] + R_vec
        remaining_inx = remaining_inx[R_vec]
        w, phi, A = prune(w, R_vec, phi, alpha)
        Y = phi @ w
        right_classify = cal_correct(t, Y)
        right_train = right_classify / n
        iter += 1
        if iter % print_every == 0 or len(w) > 1000:
            log_msg('At iteration {}, {} support vectors and {:.4f} training accuracy.'
                  .format(iter, len(w), right_classify / n))
    log_msg('Converged after {} iterations to {} support vectors and {:.4f} training accuracy.'
          .format(iter, len(w), right_classify / n))
    return w, remaining_inx[1:] - 1


def train_multiclass_rvm(train, labels):
    # decide the number of classes
    classes = list(set(labels))
    d = train.shape[1]
    pc = train.shape[0] // (len(classes)*2)

    rvms = {}

    for c in classes:
        log_msg('Training class {}...'.format(c))

        c_i = np.argwhere(labels == c).ravel()
        nc_i = np.argwhere(labels != c).ravel()
        n = len(c_i)
        t = 3

        train_c = np.empty((t*n, d))
        train_c[:n,:] = train[c_i,:]
        train_c[n:,:] = train[np.random.choice(nc_i, (t-1)*n, replace=False),:]

        label_c = np.empty(t*n,)
        label_c[:n] = 1
        label_c[n:] = -1

        w, sv_i = train_rvm(cal_Phi(train_c, train_c), label_c)
        rvms[c] = (w, train_c[sv_i])

    return rvms


def test_multiclass_rvm(x, y, rvms):
    log_msg('Testing...')
    ps = []
    for c, (w, sv) in rvms.items():
        phi = cal_Phi(x, sv)
        p = 1 / (1 + np.exp(-(phi @ w)))
        print(c, min(y), max(y), min(p), max(p))
        ps.append(p)
    correct = 0
    y_star = np.empty(y.shape, dtype=int)
    for i in range(len(y)):
        inferred = np.argmax([p[i] for p in ps])
        y_star[i] = inferred
        correct += 1 if inferred == y[i] else 0
    log_msg(correct, len(y))
    log_msg('\n{}'.format(confusion_matrix(classes=sorted(rvms.keys()), expected=y, actual=y_star)))


def confusion_matrix(classes, expected, actual):
    matrix = np.zeros((len(classes), len(classes)))
    for i in range(len(expected)):
        matrix[expected[i], actual[i]] += 1
    return matrix


def rvm(train_x, train_y, test_x, test_y):
    train_x = preprocess_data(train_x)
    test_x = preprocess_data(test_x)
    rvms = train_multiclass_rvm(train_x, train_y)
    test_multiclass_rvm(test_x, test_y, rvms)
