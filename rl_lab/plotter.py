from matplotlib.axes import Axes


def set_plot_properties(plot: Axes, title: str, x_label: str, y_label: str):
    """
    Seet properties of the plot

    Parameters
    ----------
        
    plot : Axes 
        plot for the visualization
        
    title : str
        title of the plot
        
    x_label : str
        x label of the plot
        
    y_label : str
        y label of the plot
    """
    plot.set_title(title)
    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)
    plot.legend(bbox_to_anchor=(1.3, 0.5))
    
    return plot

def plot_lines(plot, data, x_values, y_key, title, x_label, y_label):
    """
    Plot multiple lines on the same plot

    Parameters
    ----------
        
    plot : Axes 
        plot for the visualization
        
    data: 
        data sctructure structure containing all the ploting information.
        data = {
            "plot1": [0.2, 0.5, 0.8, 0.9]
            "color": "red"
        }
        
    x_values : List[Union[int, float]]
        list of values to appear on the common x-axis. The number of values must be the same as that of 'data[name][y_key]'
        
    y_key : Axes 
        key to get the values of the y-axis in the dictionary data
        
    title : str
        title of the plot
        
    x_label : str
        x label of the plot
        
    y_label : str
        y label of the plot
    """
    for name in data:
        plot.plot(x_values, 
                data[name][y_key],
                color=data[name]["color"],
                label=name)
    
    set_plot_properties(plot, title, x_label, y_label)
    
    return plot
    


def plot_bars(plot, data, bins, y, title, x_label, y_label, offset=0.0):
    """
    Plot multiple sets of bars on the same plot

    Parameters
    ----------
        
    plot : Axes 
        plot for the visualization
        
    data: 
        data sctructure structure containing all the ploting information.
        data = {
            "plot1": [0.2, 0.5, 0.8, 0.9]
            "color": "red"
        }
        
    bins : List[Union[int, float]]
        number of bins to use
        
    y_key : Axes 
        key to get the values of the y-axis in the dictionary data
        
    title : str
        title of the plot
        
    x_label : str
        x label of the plot
        
    y_label : str
        y label of the plot
    """
    width = (1.0 / len(data)) - 0.01
    
    for name in data:
        plot.bar(bins + offset*width, 
                data[name][y], 
                color=data[name]["color"],
                label=name,
                width=width)
        offset += 1

    set_plot_properties(plot, title, x_label, y_label)
    
    return plot