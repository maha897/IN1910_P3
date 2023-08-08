from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from chaos_game import ChaosGame


class Variations:
    """A varation is a function that changes the shape and character of a solution in a recognizable way.
    This class creates transformations of figures by remapping their planes differently given a variation.
    
    This class allows for 9 different transformations:
        - linear
        - handkerchief
        - swirl
        - disc
        - ex
        - eyefish
        - diamond
        - hyperbolic
        - polar
    """

    def __init__(self, x: list, y: list, name: str) -> None:
        """Constructor of the variation class.

        Args:
            x (list): 
                list of the x values
            y (list): 
                list of the y values
            name (str): 
                name of the desired transformation
        """

        self.x = x
        self.y = y
        self.name = name
        self._func = getattr(Variations, self.name)

    def transform(self) -> list:
        """Calls the transformation function.

        Returns:
            list: 
                the x and y coordinates of the transformed plane.
        """

        return self._func(self.x, self.y)

    @staticmethod
    def linear(x: list, y: list) -> list:
        """Linear variation, V0. Conserves the plane."""

        return x, y

    @staticmethod
    def handkerchief(x: list, y: list) -> list:
        """"Handkerchief variation, V6."""

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        x = r*(np.sin(theta + r))
        y = r*(np.cos(theta - r))

        return x, y

    @staticmethod
    def swirl(x: list, y: list) -> list:
        """Swirl variation, V3."""

        r = np.sqrt(x**2 + y**2)
        x = x*np.sin(r**2) - y*np.cos(r**2)
        y = x*np.cos(r**2) + y*np.sin(r**2)

        return x, y

    @staticmethod
    def disc(x: list, y: list) -> list:
        """Disc variation, V8."""

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        x = (theta/np.pi)*np.sin(np.pi*r)
        y = (theta/np.pi)*np.cos(np.pi*r)

        return x, y

    @staticmethod 
    def ex(x: list, y: list) -> list:
        """Ex variation, V12."""

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        p0 = np.sin(theta + r)
        p1 = np.cos(theta - r)

        x = r*(p0**3 + p1**3)
        y = r*(p0**3 - p1**3)

        return x, y
    
    @staticmethod 
    def eyefish(x: list, y: list) -> list:
        """Eyefish variation, V27."""

        r = np.sqrt(x**2 + y**2)
        x = (2/(r + 1))*x
        y = (2/(r + 1))*y

        return x, y

    @staticmethod 
    def diamond(x: list, y: list) -> list:
        """Diamond variation, V11."""

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        x = np.sin(theta)*np.cos(r)
        y = np.cos(theta)*np.sin(r)

        return x, y

    @staticmethod 
    def hyperbolic(x: list, y: list) -> list:
        """Hyperbolic variation, V10."""

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        x = np.sin(theta)/r
        y = r*np.cos(theta)

        return x, y

    @staticmethod
    def polar(x: list, y: list) -> list:
        """Polar variation, V5."""

        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(x, y)

        x = theta/np.pi
        y = r - 1

        return x, y

    @classmethod
    def from_chaos_game(cls, chaos: "ChaosGame", name: str) -> "Variations":
        """Takes an instance of the ChaosGame and a tranformation and returns and instance 
        of the Variations class.

        Args:
            chaos (ChaosGame): 
                an instance of the ChaosGame class.
            name (str): 
                name of variation.

        Returns:
            Variations:
                an instance of Variations class.
        """

        x = chaos.points[:,0]
        y = chaos.points[:,1]
    
        return Variations(x, y, name)


def plot_variations() -> None:
    """Instanciates and plots every transformation of Variations class."""

    grid_values = np.linspace(-1, 1, 70)
    x, y = np.meshgrid(grid_values, grid_values)

    x_values = x.flatten()
    y_values = y.flatten()

    transformations = ["linear", "handkerchief", "swirl", "disc", "ex", "eyefish", "diamond", "hyperbolic", "polar"]
    variations = [Variations(x_values, y_values, version) for version in transformations]
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))

    for i, (ax, variation) in enumerate(zip(axs.flatten(), variations)):
        u, v = variation.transform()
        ax.plot(u, -v, markersize=.5, marker=".", linestyle="", color="black")
        ax.set_title(variation.name)
        ax.axis("off")

    print("Saving plot as figures/variations_4b.png")
    fig.savefig("figures/variations_4b.png")
    plt.show()

def plot_ngon_variations() -> None:
    """Instanciates with an instance of ChaosGame and plots every transformation of Variations class."""

    cg = ChaosGame(3)
    colors = cg.gradient_color

    transformations = ["linear", "handkerchief", "swirl", "disc", "ex", "eyefish", "diamond", "hyperbolic", "polar"]
    variations = [Variations.from_chaos_game(cg, name=version) for version in transformations]
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))

    for i, (ax, variation) in enumerate(zip(axs.flatten(), variations)):
        u, v = variation.transform()
        ax.scatter(u, -v, s=0.2, marker=".", c=colors)
        ax.set_title(variation.name)
        ax.axis("off")

    print("Saving plot as figures/variations_4c.png")
    fig.savefig("figures/variations_4c.png")
    plt.show()

def linear_combination_wrap(V1: "Variations", V2: "Variations") -> Callable:
    """Calculates linear combinations of two variations.

    Args:
        V1 (Variations): 
            an instance of the Variations class.
        V2 (Variations): 
            an instance of the Variations class.

    Returns:
        Callable: 
            function that takes a weight w as argument.
    """

    x1, y1 = V1.transform()
    x2, y2 = V2.transform()

    def f(w: float) -> np.ndarray:
        """Returns the linear combinations of two variations using weight.

        Args:
            w (float): 
                the weight

        Returns:
            np.darray:
                the linear combination.
        """
        
        x = w*x1 + (1 - w)*x2
        y = w*y1 + (1 - w)*y2

        return x, y
    
    return f

def plot_linear_combination() -> None:
    """Plots the linear combination as result of linear_combination_wrap()."""

    coeffs = np.linspace(0, 1, 9)
    ngon = ChaosGame(5, 3/8)
    n_color = ngon.gradient_color

    variation1 = Variations.from_chaos_game(ngon, "linear")
    variation2 = Variations.from_chaos_game(ngon, "disc")
    variation12 = linear_combination_wrap(variation1, variation2)    
        
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    for ax, w in zip(axs.flatten(), coeffs):
        u, v = variation12(w)
    
        ax.scatter(u, -v, s=0.2, marker=".", c=n_color)
        ax.set_title(f"weight = {w:.2f}")
        ax.axis("off")

    print("Saving plot as figures/linearcombination_4d.png")
    fig.savefig("figures/linearcombination_4d.png")
    plt.show()

if __name__ == "__main__":
    # Exercise 4b
    plot_variations()

    # Exercise 4c
    plot_ngon_variations()
    
    # Exercise 4d
    plot_linear_combination()