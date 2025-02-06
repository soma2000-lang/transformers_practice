from basic_operations import Tensor

class NumericalDiff(Tensor):

    def __init__(self, data, f):
        super().__init__(data)
        self.data = data
        self.f = f

    def backward(self, eps = 1e-6):
        x_plus = Tensor(self.data + eps)
        x_minus = Tensor(self.data - eps)
        y_plus = self.f(x_plus)
        y_minus = self.f(x_minus)
        grad = (y_plus - y_minus) / (2 * eps)
        return Tensor(grad)


if __name__ == '__main__':
    x = Tensor(3)

    def f(x):
        return (3 * x * x) + (2 * x)
    
    x = NumericalDiff(x, f)
    grad = x.backward()
    print(grad)