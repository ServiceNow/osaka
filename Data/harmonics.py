from torch.utils.data import Dataset
import numpy as np

class Harmonics(Dataset):
    def __init__(self, train=True, n_tasks=1000, sample_size=50):
        """
        Creates a dataset of sinusoids as described in https://arxiv.org/pdf/1712.05016.pdf
        """
        self.train = train
        self.x = self.sample_x(n_tasks * sample_size).reshape((n_tasks, sample_size))
        self.y = np.zeros((n_tasks, sample_size))
        if self.train:
            wmin = 2
            wmax = 4
        else:
            wmin = 5
            wmax = 7
        for n in range(n_tasks):
            self.y[n, :] = self.f_x(self.x[n, :], wmin, wmax)

        self.data = np.dstack([self.x, self.y])
        


    def sample_x(self, quantity):
        """
        Samples an array with the x axis
        """
        x = np.random.uniform(low=-4, high=4, size=quantity)
        x += np.random.randn(quantity)
        return x

    def f_x(self, x, wmin, wmax, sigma=0.01):
        """
        Computes y ~ N(f(x), sigma). wmin and wmax control the frequency of the wave
        """
        w = np.random.uniform(low=wmin, high=wmax, size=1)
        a1, a2 = np.random.randn(2)**2
        b1, b2 = np.random.uniform(low=0, high=2*np.pi, size=2) ** 2
        y = a1 * np.sin(w * x + b1) + a2 * np.sin(2 * w * x + b2)
        return y + np.random.randn(y.shape[0]) * sigma

if __name__ == "__main__":
    harmonics = Harmonics()
    import pylab

    x = np.argsort(harmonics.x[0])
    pylab.plot(harmonics.x[0, x], harmonics.y[0, x])
    pylab.savefig('tmp.png')



