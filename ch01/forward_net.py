import numpy as np


# �ñ׸��̵�(Sigmoid) ���̾� ����
class Sigmoid:
    """Sigmoid Layer class

    Sigmoid layer���� �н��ϴ� params�� ���� �����Ƿ�
    �ν��Ͻ� ������ params�� �� ����Ʈ�� �ʱ�ȭ

    """

    def __init__(self):
        self.params = []

    def forward(self, x):
        """������(forward propagation) �޼���
        Args:
            x(ndarray): �Է����� ������ ��
        Returns:
            Sigmoid Ȱ��ȭ ��
        """
        return 1 / (1 + np.exp(-x))


# �����������(Affine) ����
class Affine:
    """FC layer"""

    def __init__(self, W, b):
        """
        Args:
            W(ndarray): ����ġ(weight)
            b(ndarray): ����(bias)
        """
        self.params = [W, b]

    def forward(self, x):
        """������(forward propagation) �޼���
        Args:
            x(ndarray): �Է����� ������ ��
        Returns:
            out(ndarray): Wx + b
        """
        W, b = self.params
        out = np.matmul(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # ����ġ�� ���� �ʱ�ȭ
        # input -> hidden
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        # hidden -> output
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # ���̾� ����
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        # ��� ����ġ�� ����Ʈ�� ������.
        self.parmas = [layer.params for layer in self.layers]
        # self.params = []
        # for layer in self.layers:
        #     self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)

    print(f"{s!r}")
