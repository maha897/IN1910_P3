import pytest
import os
from chaos_game import ChaosGame, make_figure
import matplotlib.pyplot as plt

cg = ChaosGame(5)

@pytest.mark.parametrize(
    "n, r, exception", [
        (" ", .5, TypeError),
        (4, " ", TypeError),
        (0, 1/3, ValueError),
        (3, -1, ValueError),
        (2, .5, ValueError),
        (5, 1, ValueError),
    ],
)

def test_errors_constructor(n, r, exception):
    with pytest.raises(exception):
        ChaosGame(n, r)


def test_error_savepng():
    with pytest.raises(ValueError):
        cg.savepng("noe.txt")


@pytest.mark.parametrize(
    "c, outfile", [
        (False, "test"), 
        (True, "test.png")
    ]
)

def test_savepng_and_color(c, outfile):
    cg.savepng(outfile, c)
    dest = os.getcwd()+"/"+outfile

    if not outfile.endswith(".png"):
        dest += ".png"
    
    assert os.path.isfile(dest)

    if os.path.exists(dest):
       os.remove(dest)


def test_corners():
    for i in range(3, 10):
        assert len(ChaosGame(i).corners) == i


def test_iterate():
    cg.iterate(300)
    len(cg.indices) == 300
    len(cg.gradient_color) == 300


def test_make_figure():
    make_figure(88, 7, 1/2)
    dest = os.getcwd()+"/figures/chaos88.png"

    assert os.path.isfile(dest)

    if os.path.exists(dest):
       os.remove(dest)
