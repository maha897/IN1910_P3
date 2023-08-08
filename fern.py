import numpy as np
import matplotlib.pyplot as plt

class AffineTransform:
    """Affine transformations are functions that are linear in both x and y coordinates.
    
    This class defines an instance of such a function."""

    def __init__(self, a: float=0, b: float=0, c: float=0, d: float=0, e: float=0, f: float=0) -> None:
        """Contructor for AffineTransform class. 

        The function will be on the form

            f(x, y) = [[a b][c d]][x y] + [e f]
        
        where A = abcd form a matrix and B = xy and C = ef are vectors.

        Args:
            a, b, c, d, e, f (float, Optional): 
                float numbers. Defaults to 0.
        """

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def __call__(self, x: float, y: float) -> np.ndarray:
        """Transforms point with f(x, y).

        Args: 
            x (float): 
                x coordinate
            y (float): 
                y coordinate

        Returns:
            np.ndarray: 
                the point resulting from the transform f(x, y).
        """

        A = np.array([[self.a, self.b], [self.c, self.d]])
        B = np.array([x, y])
        C = np.array([self.e, self.f])
        
        return A @ B + C

def random_f(fs: list, ps: list) -> "AffineTransform":
    """Draws a corner given probability for each corner. (Non-uniform drawing)

    Args:
        fs (list): 
            list of points that are instances of AffineTransform class.
        ps (list): 
            list of probabilities (elements between 0-1)

    Returns:
        AffineTransform: 
            a randomly chosen point.
    """

    r = np.random.random()
    p_cumulative = np.cumsum(ps)

    for j, p in enumerate(p_cumulative):
        if r < p:
            return fs[j]

def iterate_Fern(fs: list, ps: list, N: int) -> np.ndarray:
    """Iterates Fern and generates random points.

    Args:
        fs (list): 
            list of points that are instances of AffineTransform class.
        ps (list): 
            list of probabilities (elements between 0-1)
        N (int):
            number of iterations.

    Returns:
        np.ndarray: 
            the generated points.
    """

    X = np.zeros((N, 2))
    
    for i in range(N-1):
        f = random_f(fs, ps)
        X[i+1] = f(X[i][0], X[i][1])

    return X

def plot_Fern(X: np.ndarray) -> None:
    """Plots fractal image made from iterate_Fern().

    Args:
        X (np.ndarray): array of points consisting of x and y coordinates.
    """

    print("Saving plot as figures/barnsley_fern.png")
    plt.scatter(*zip(*X), c="forestgreen", marker=".", s=.2)
    plt.axis("equal")
    plt.savefig("figures/barnsley_fern.png", dpi=300)
    plt.close()

if __name__ == "__main__":

    # Exercise 3b
    f1 = AffineTransform(d=.16)
    f2 = AffineTransform(a=.85,b=.04,c=-.04,d=.85,f=1.6)
    f3 = AffineTransform(a=.2,b=-.26,c=.23,d=.22,f=1.6)
    f4 = AffineTransform(a=-.15,b=.28,c=.26,d=.24,f=.44)
    fs = [f1, f2, f3, f4]

    # Exercise 3c
    p1 = .01
    p2 = .85
    p3 = .07
    p4 = .07
    ps = [p1, p2, p3, p4]

    assert sum(ps) == 1

    # Exercise 3d
    X = iterate_Fern(fs, ps, 50000)

    # Exercise 3e
    plot_Fern(X)