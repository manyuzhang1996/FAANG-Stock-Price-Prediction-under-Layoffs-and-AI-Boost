# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 19:37:24 2023

@author: yunhui
"""

from data_visualization import data_visualization
from training import run_models_and_plot
from dataset import get_stock_data

def main():
    df = get_stock_data()

    print("Data Visualization: \n")
    data_visualization()

    print("Training:\n")
    amzn_results = run_models_and_plot('AMZN')
    meta_results = run_models_and_plot('META')
    aapl_results = run_models_and_plot('AAPL')
    nflx_results = run_models_and_plot('NFLX')
    goog_results = run_models_and_plot('GOOG')

    print(amzn_results)
    print(meta_results)
    print(aapl_results)
    print(nflx_results)
    print(goog_results)

if __name__ == "__main__":
    main()

