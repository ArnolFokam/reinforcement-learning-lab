def set_plot_properties(plot, title, x_label, y_label):
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plot.legend(bbox_to_anchor=(1.3, 0.5))

def plot_lines(plot, data, x_values, y_key, title, x_label, y_label):
    for name in data:
        plot.plot(x_values, 
                data[name][y_key],
                color=data[name]["color"],
                label=name)
    
    set_plot_properties(plot, title, x_label, y_label)
    


def plot_bars(plot, data, bins, y, title, x_label, y_label, offset=0.0):
    width = (1.0 / len(data)) - 0.01
    
    for name in data:
        plot.bar(bins + offset*width, 
                data[name][y], 
                color=data[name]["color"],
                label=name,
                width=width)
        offset += 1

    set_plot_properties(plot, title, x_label, y_label)