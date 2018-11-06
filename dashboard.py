#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:00:38 2018
@author: pabloruizruiz
"""

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

port='8050'
#import argparse
#parser = argparse.ArgumentParser(description='Select the port')
#parser.add_argument('-p', '--port',  type=str, default='8050',
#                    help='Select the port to run the script')
#args = parser.parse_args()
#port = str(args.port)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)







if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port) ## Docker config