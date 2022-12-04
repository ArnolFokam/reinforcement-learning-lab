from matplotlib import pyplot as plt

from rl_lab.plotters import set_plot_info

fig = plt.figure()


def test_set_plot_info():
    axes = fig.add_axes([0.5, 1, 0.5, 1])
    title = "test"
    x_label = "test_x_label"
    y_label = "test_y_label"
    axes = set_plot_info(axes, title, x_label, y_label)

    assert axes.get_xlabel() == x_label
    assert axes.get_ylabel() == y_label
    assert axes.get_title() == title
