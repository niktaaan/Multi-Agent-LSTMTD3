"""
Description:
    I decided to write a class to do the following things.

    * Display plots/graphs (Matplotlib, Seaborn)
    * Export plots/graphs to folders/files (Matplotlib, Seaborn)
    * Calculate running averages
    * Calculate running standard deviations
    * Export data as .csv to folders/files (Pandas)

Author:
    Jordan Cramer
Date:
    2023-09-05
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataManager:
    """
    A reusable class with functions useful for the following things.

    * Display plots/graphs (Matplotlib, Seaborn)
    * Export plots/graphs to folders/files (Matplotlib, Seaborn)
    * Calculate running averages
    * Calculate running standard deviations
    * Export data as .csv to folders/files (Pandas)
    """
    def __init__(self):
        pass

    @staticmethod
    def display_plot(
            scores: np.array,
            sliding_window_size: int = 50,
            title: str = '',
            y_label: str = 'Score',
            x_label: str = 'Episode'
    ):
        """ A function that will display a plot of the data. """
        # create the plot
        DataManager._create_plot_single_series_moving_average_std(scores, sliding_window_size, title, y_label, x_label)

        # display the plot
        plt.show()

        # clear/close the plot
        plt.clf()
        plt.close()  # closing the figures (after receiving a warning about too many pyplot figures being open at once)

    @staticmethod
    def export_plot(
            directory: str,  # example: './folder_name/rl_agent_type/trial_number/'
            filename: str,  # example: 'my_plot.png'
            scores: np.array,
            sliding_window_size: int = 50,
            title: str = '',
            y_label: str = 'Score',
            x_label: str = 'Episode'
    ):
        """ A function that will export a plot of the data to a .png file. """
        # create the plot
        DataManager._create_plot_single_series_moving_average_std(scores, sliding_window_size, title, y_label, x_label)

        # save the plot (must create the directory first)
        DataManager._create_directory(directory)
        plt.savefig(os.path.join(directory, filename))

        # clear/close the plot
        plt.clf()
        plt.close()  # closing the figures (after receiving a warning about too many pyplot figures being open at once)

    @staticmethod
    def export_plot_multiple_series(
            directory: str,  # example: './folder_name/rl_agent_type/trial_number/'
            filename: str,  # example: 'my_plot.png'
            title: str,
            x_label: str,
            y_label: str,
            legend_labels: list[str],
            x_series: list[np.ndarray],
            y_series: list[np.ndarray],
            show_grid_lines: bool = True,
            calculate_moving_average: bool = False,
            moving_average_window_size: int = 25
    ):
        """
        Create a plot with multiple data series and save it to a file.

        Args:
            directory (str): The directory to save the plot file to. example: './folder_name/rl_agent_type/trial_number/'

            filename (str): The file name to give the plot. example: 'my_plot.png'

            title (str): The title for the plot.

            x_label (str): The x-axis label for the plot.

            y_label (str): The y-axis label for the plot.

            legend_labels (list[str]): The labels for each data series that will be shown in the legend.

            x_series (list[np.ndarray]): A list of one or more x-value data series.

            y_series (list[np.ndarray]): A list of one or more y-value data series.

            show_grid_lines (bool): If this is set to True, then grid lines will be shown on the plot.

            calculate_moving_average (bool): If this is set to True, then moving averages will be calculated for the data series before plotting.

            moving_average_window_size (int): The moving average window size used for calculating moving averages for the y-values.
        """
        # create the plot
        DataManager._create_plot_multiple_series(
            title=title,
            x_label=x_label,
            y_label=y_label,
            legend_labels=legend_labels,
            x_series=x_series,
            y_series=y_series,
            show_grid_lines=show_grid_lines,
            calculate_moving_average=calculate_moving_average,
            moving_average_window_size=moving_average_window_size
        )

        # save the plot (must create the directory first)
        DataManager._create_directory(directory)
        plt.savefig(os.path.join(directory, filename))

        # clear/close the plot
        plt.clf()
        plt.close()  # closing the figures (after receiving a warning about too many pyplot figures being open at once)

    @staticmethod
    def export_csv(
            directory: str,  # example: './folder_name/rl_agent_type/trial_number/'
            filename: str,  # example: 'my_plot.csv'
            scores: np.array,
            sliding_window_size: int = 50
    ):
        """
        A function that will export data as a .csv file.

        Example:
            filename example: 'my_data.csv'
        """
        # specify the data (with table headers) to export to the .csv file
        data = DataManager._create_data_dict(scores, sliding_window_size)

        # create the pandas data frame
        data_frame = pd.DataFrame(data)

        # create the directory if it doesn't already exist
        DataManager._create_directory(directory)

        # export the score history to a .csv file
        file_path = os.path.join(directory, filename)
        data_frame.to_csv(file_path, index=False)

    @staticmethod
    def export_csv_from_dictionary(
            directory: str,  # example: './folder_name/rl_agent_type/trial_number/'
            filename: str,  # example: 'my_plot.csv'
            dictionary: dict
    ):
        """
        Export data from a dictionary to a .csv file.
        The dictionary should have a particular form shown in the example.

        Example:
            The dictionary containing the data needs to have a particular form to export to a .csv file.
            The keys of the dictionary will be the column headers,
            and the data in the columns should be 1D numpy.ndarray of the same lengths.

            >>> import numpy
            >>> dummy_data_1 = np.zeros(shape=[10], dtype=int)
            >>> dummy_data_2 = np.zeros(shape=[500], dtype=float)
            >>>
            >>> same_array_length = 10
            >>>
            >>> data = {
            >>>     "Episode": dummy_data_1[0:same_array_length],
            >>>     "Score": dummy_data_2[0:same_array_length],
            >>>     "Running Average": dummy_data_2[0:same_array_length],
            >>>     "Running std": dummy_data_2[0:same_array_length]
            >>> }

        Args:
            directory (str): The directory to save the .csv file to. example: './folder_name/rl_agent_type/trial_number/'

            filename (str): The file name to give the file. example: 'my_data.csv'

            dictionary (dict): The dictionary with the data to export to a .csv file.
        """
        # create the pandas data frame from the dictionary
        data_frame = pd.DataFrame(dictionary)

        # create the directory if it doesn't already exist
        # (can't save the .csv to a folder that doesn't exist)
        DataManager._create_directory(directory)

        # export the data to a .csv file
        filepath = os.path.join(directory, filename)
        data_frame.to_csv(filepath, index=False)

    @staticmethod
    def _calculate_running_averages(
            scores: np.array,
            sliding_window_size: int
    ) -> np.array:
        """ A function to calculate running average values. """
        # get the length of the numpy array
        length = scores.shape[0]
        running_averages = np.zeros(length, dtype=float)
        for i in range(length):
            running_averages[i] = scores[max(0, i + 1 - sliding_window_size):i + 1].mean()
        return running_averages

    @staticmethod
    def _calculate_running_standard_deviations(
            scores: np.array,
            sliding_window_size: int
    ) -> np.array:
        """ A function to calculate running standard deviation values. """
        # get the length of the numpy array
        length = scores.shape[0]
        running_stds = np.zeros(length, dtype=float)
        for i in range(length):
            running_stds[i] = scores[max(0, i + 1 - sliding_window_size):i + 1].std()
        return running_stds

    @staticmethod
    def _create_data_dict(scores: np.array, sliding_window_size: int = 50):
        """
        Creates a dictionary of data that can be used easily with Pandas data frames.

        Example:
            data = {
                "Episode": episode_number[0:i+1],
                "Score": score_history[0:i+1],
                "Running Average": score_history_running_average[0:i+1],
                "Running std": score_history_running_std[0:i+1]
            }
        """
        # get the length of the numpy array
        length = scores.shape[0]
        # useful data format for exporting to .csv files with headers
        data = {
            "Episode": np.array([i for i in range(length)], dtype=int),
            "Score": scores,
            "Running Average": DataManager._calculate_running_averages(scores, sliding_window_size),
            "Running std": DataManager._calculate_running_standard_deviations(scores, sliding_window_size)
        }
        return data

    @staticmethod
    def _create_plot_multiple_series(
            title: str,
            x_label: str,
            y_label: str,
            legend_labels: list[str],
            x_series: list[np.ndarray],
            y_series: list[np.ndarray],
            show_grid_lines: bool = True,
            calculate_moving_average: bool = False,
            moving_average_window_size: int = 25
    ):
        """
        Create a plot with multiple data series.

        Args:
            title (str): The title for the plot.

            x_label (str): The x-axis label for the plot.

            y_label (str): The y-axis label for the plot.

            legend_labels (list[str]): The labels for each data series that will be shown in the legend.

            x_series (list[np.ndarray]): A list of one or more x-value data series.

            y_series (list[np.ndarray]): A list of one or more y-value data series.

            show_grid_lines (bool): If this is set to True, then grid lines will be shown on the plot.

            calculate_moving_average (bool): If this is set to True, then moving averages will be calculated for the data series before plotting.

            moving_average_window_size (int): The moving average window size used for calculating moving averages for the y-values.
        """
        # create the plot figure with a particular size
        width = 10
        height = 6
        plt.figure(figsize=(width, height))

        # set the theme to be the standard Seaborn theme to make the plot style look good
        sns.set_theme()

        # set plot information
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # plot all the data series with their labels
        # colors are automatic unless specified explicitly in plt.plot(color='') code
        for x, y, label in zip(x_series, y_series, legend_labels):
            # calculate the moving averages for y-values if desired
            if calculate_moving_average is True:
                y = DataManager._calculate_running_averages(y, moving_average_window_size)
            # plot the data series
            plt.plot(x, y, label=label, marker='o')

        # set plot information
        plt.title(title)  # set the title of the plot
        plt.xlabel(x_label)  # set the label for the x-axis
        plt.ylabel(y_label)  # set the label for the y-axis
        plt.legend()  # show the data series legend on the plot

        # show grid lines on the plot
        if show_grid_lines is True:
            plt.grid(True)

        # adjust layout for better spacing
        plt.tight_layout()

    @staticmethod
    def _create_plot_single_series_moving_average_std(
            scores: np.array,
            sliding_window_size: int = 25,
            title: str = '',
            y_label: str = 'Score',
            x_label: str = 'Episode'
    ):
        """
        Creates a nice looking plot with Matplotlib and Seaborn.

        There will be two plots side by side horizontally.

        1. Running Average Score and Running Standard Deviation vs Episode Number
        2. Score and Running Average Score vs Episode Number
        """
        # Given the scores, calculate the needed y-series data
        data = DataManager._create_data_dict(scores, sliding_window_size)
        x = data['Episode']
        y1 = data['Running Average']
        std = data['Running std']
        y2 = data['Score']

        # Set the standard Seaborn theme to make the plot styling look good
        sns.set_theme()

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 9))

        # Plot the first line plot with error bands
        ax1.plot(x, y1, color='maroon', label=f'Running Average Score - Window Size [{sliding_window_size}]')
        ax1.fill_between(x, y1 - std, y1 + std, color='orangered', alpha=0.3)
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label)
        appended_title = title + '\n' + 'Running Average Score and Standard Deviation'
        ax1.set_title(appended_title)
        ax1.legend()

        # Plot the second line plot
        ax2.plot(x, y2, color='olive', label='Score')
        ax2.plot(x, y1, color='darkorchid', label=f'Running Average Score - Window Size [{sliding_window_size}]')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label)
        appended_title = title + '\n' + 'Score and Running Average Score'
        ax2.set_title(appended_title)
        ax2.legend()

        # Adjust the spacing between subplots
        plt.tight_layout()

    @staticmethod
    def _create_directory(directory: str):
        """ Creates a directory if it doesn't already exist. """
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError:
                print(f'Failed to create the directory {directory}')

    @staticmethod
    def write_string_to_file(directory: str, filename: str, string: str):
        """
        Write the string to a text file specified by filepath.

        Example:
            write_string_to_file("./directory/", "my_file.txt", string)
        """
        DataManager._create_directory(directory)
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'w') as file:
                file.write(string)
        except IOError:
            print(f'Error writing to {file_path}')


if __name__ == '__main__':
    """ This section is for writing simple and quick tests for the class. """
