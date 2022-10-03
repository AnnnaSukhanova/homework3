import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px


class Visualization:

    def __create_df(self, data, columns):
        return pd.DataFrame(data=data, columns=columns)

    def models_error_scatter_plot(self, error_valid, error_test, names, lambda_lst, title, show=False, save=False, name=None,
                                  path2save=None):
        """

        :param error_valid: numpy.array - errors on validation set
        :param error_test: numpy.array - errors on test set
        :param names: numpy.array of strings - how y(x,w) looks
        :param lambda_lst: numpy.array - lambda value
        :param title: title of plot
        :param show: (bool) optional if True show figure in browser
        :param save: (bool) optional if True save figure in html format
        :param name: (str) optional name of html file
        :param path2save: (str) optional path to directory, where html file is going to be saved
        example
            /dir/dir/
        """

        args_sort = np.argsort(error_valid)[::-1]
        df = self.__create_df(np.stack((error_valid[args_sort],
                                error_test[args_sort],
                                names[args_sort],lambda_lst[args_sort]), axis=1),
                              ['error_valid', 'error_test', 'function','lambda'])
        fig = px.scatter(df, y="error_valid", x="function",hover_data=['error_test','lambda'])

        fig.update_layout(title=title)
        if show:
            fig.show()
        if save:
            assert name is not None, "name shouldn't be None if  save is True"
            if path2save:
                fig.write_html(f"{path2save}/{name}.html")
            else:
                fig.write_html(f"{name}.html")
