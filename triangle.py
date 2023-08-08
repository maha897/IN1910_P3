import numpy as np
import matplotlib.pyplot as plt

c0 = np.array([0,0])
c1 = np.array([1,0])
c2 = np.array([1/2, np.sqrt(3)/2])

corners = [c0, c1, c2]

def plot_corners(c: list[np.ndarray]) -> None:
    """Plots a set of corners.

    Args:
        c (list[np.asarray]): 
            a list of np.arrays with x and y values, representing the corners.
    """

    plt.scatter(*zip(*c))
    plt.grid(True)
    plt.show()
    plt.close()

def plot_triangle(c: list[np.ndarray]) -> None:
    """Draws the edges of triangle.

    Args:
        c (list[np.asarray]): 
            a list of np.arrays with x and y values, representing the corners.
    """

    x_vals = []
    y_vals = []

    for i in range(len(c)):
        x_vals.append(c[i][0])
        y_vals.append(c[i][1])

    plt.plot([x_vals[0], x_vals[1]], [y_vals[0], y_vals[1]])
    plt.plot([x_vals[1], x_vals[2]], [y_vals[1], y_vals[2]])
    plt.plot([x_vals[2], x_vals[0]], [y_vals[2], y_vals[0]])

def draw_point(c: list[np.ndarray]) -> np.ndarray:
    """Creates a random point inside area of a given triangle.

    Args:
        c (list[np.asarray]): 
            a list of np.arrays with x and y values, representing the corners.
    Returns:
        (np.ndarray): 
            the coordinates of the generated point.
    """

    w0, w1, w2 = np.random.random(size=3)
    sum_w = w0 + w1 + w2

    w0 = w0/sum_w
    w1 = w1/sum_w
    w2 = w2/sum_w

    weights = [w0, w1, w2]

    x = 0; y = 0
    for i in range(3):
        x += weights[i]*c[i][0]
        y += weights[i]*c[i][1]

    return np.array([x, y])

def plot_points(N: int, c: list[np.ndarray]) -> None:
    """ Plots N randomly generated points within the bounds of c.

    Args:
        N (int): 
            number of points to draw.
        c (list[np.ndarray]): 
            a list of np.arrays with x and y values, representing the corners.
    """

    plot_triangle(c)

    for _ in range(N):
        point = draw_point(c)
        plt.scatter(point[0], point[1])

    plt.show()
    plt.close()

def Sierpinski(c: list[np.ndarray], N: int) -> np.ndarray:
    """Generates N points within corners c using 
        X[i+1] = (X[i] + c[j])/2.

    Args:
        c (list[np.ndarray]): 
            list of corners.
        N (int): 
            number of points generated.

    Returns:
        np.ndarray: 
            the x and y values of N points.
    """

    points = np.zeros((N, 2))
    points[0] = draw_point(c)

    for i in range(1, N):
        points[i] = (points[i-1] + c[np.random.randint(3)])/2

    return points[5:]

def plot_Sierpinski(N: int) -> None:
    """Plots x and y values recieved by Sierpinski() function nicely. 

    Args:
        N (int): 
            number of points.
    """

    fig, ax = plt.subplots()
    ax.scatter(*zip(*Sierpinski(corners, N)), s=0.1, marker=".")
    ax.axis("equal")
    ax.axis("off")
    ax.set_title("Sierpinski Triangle")

    plt.show()

def colored_Sierpinski(c: list[np.ndarray], N: int) -> np.ndarray:
    """Generates N points within corners c using 
        X[i+1] = (X[i] + c[j])/2 and generating color while iterating.

    Args:
        c (list[np.ndarray]): 
            list of corners.
        N (int): 
            number of points generated.

    Returns:
        np.ndarray: 
            the x and y values of N points, and assigned color to given point.
    """

    points = np.zeros((N, 2))
    color = np.zeros(N)
    points[0] = draw_point(c)

    for i in range(1, N):
        j = np.random.randint(3)

        points[i] = (points[i-1] + c[j])/2

        if j == 0:
            color[i] = 0
        if j == 1:
            color[i] = 1
        if j == 2:
            color[i] = 2

    return points[5:], color[5:]

def plot_colored_Sierpinski(N: int) -> None:
    """Plots Sierpinski trianlge with colors.

    Args:
        N (int): 
            number of points.
    """

    X, colors = colored_Sierpinski(corners, N)

    red  =  X[colors == 0]
    green = X[colors == 1]
    blue  = X[colors == 2]

    fig, ax = plt.subplots()
    ax.scatter(*zip(*red), s=0.1, marker=".", color="red")
    ax.scatter(*zip(*green), s=0.1, marker=".", color="green")
    ax.scatter(*zip(*blue), s=0.1, marker=".", color="blue")
    ax.set_title("Colored Sierpinski Triangle")
    ax.axis("equal")
    ax.axis("off")

    plt.show()

def alternative_color_func(N: int) -> np.ndarray:
    """Alternative color function. Functionally same colored_Sierpinski.

    Args:
        N (int):
            number of points

    Returns:
        np.ndarray: 
            RGB values
    """

    X = np.zeros((N, 2))
    C = np.zeros((N, 3))
    
    r0 = np.array([1,0,0])
    r1 = np.array([0,1,0])
    r2 = np.array([0,0,1])
    r = [r0, r1, r2]

    for i in range(N-1):
        j = np.random.randint(3)
        C[i+1] = (C[i] + r[j])/2
        X[i+1] = (X[i] + corners[j])/2

    return X[5:], C[5:]

def alternative_plot_color_Sierpinski(N: int) -> None:
    """Plots triangle with colors generates by alternative_color_func.

    Args:
        N (int): 
            number of points
    """

    X, colors = alternative_color_func(N)
    plt.scatter(*zip(*X), c=colors, s=0.2)
    plt.title("Alternative color Sierpinski Triangle")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Exercise 1a
    plot_corners(corners)
    
    # Exercise 1b
    plot_points(1000, corners)

    # Exercise 1d
    plot_Sierpinski(10005)

    # Exercise 1e
    plot_colored_Sierpinski(10000)

    # Exercise 1f 
    alternative_plot_color_Sierpinski(10000)
    
