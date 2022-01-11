import sys

sys.path.append("..")
from common.np import *


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        # overflow�� ���� ���� �Է°� ��
        # �ִ밪�� ���ش�. >> �ط���-1, 3.5.2 ����
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # ���� �����Ͱ� ���� ������ ��� ���� ���̺� �ε����� ��ȯ
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    cross_entropy = np.log(y[np.arange(batch_size), t] + 1e-7)
    loss = -np.sum(cross_entropy) / batch_size

    return loss
