# CBOW���� �߷� ó�� ����
# chap03/cbow_predict.py
from common.layers import MatMul
import numpy as np
import sys
sys.path.append('..')

# ���� �ƶ� ������
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# ����ġ �ʱ�ȭ
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# ���� ����
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# ������
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)  # average
s = out_layer.forward(h)  # score

print(s)
