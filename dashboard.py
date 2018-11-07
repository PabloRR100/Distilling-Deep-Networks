#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:00:38 2018
@author: pabloruizruiz
"""

import pickle
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


import torch
from networks import TorchNet
net = torch.load('torchnet.pkl')
## TODO Import model by drag and drop a .pkl

Winp_stats = net.weight_stats['Winp']
Whid_stats = net.weight_stats['Whid']
Wout_stats = net.weight_stats['Wout']
Winp_scale = net.weight_stats['rWinp']
Whid_scale = net.weight_stats['rWhid']
Wout_scale = net.weight_stats['rWout']

xaxis = range(len(Winp_stats['mean']))



# Dashboard Layout
# ----------------

app.layout = html.Div([
    
    # Top Bar 
    # -------
    
    html.Div([
            
        # Squiggle Logo
        html.Div([
            html.A([
                html.Img(src='https://static.scholar.harvard.edu/files/styles/os_files_xxlarge/public/stratos/files/short_logo_seas_shield_dark.png?m=1499735141&itok=-7Y9nAsB', 
                     style={'max-width':'60%', 'max-height':'60%'})       
            ], href='https://github.com/Lab41/squiggle')
        ], className = 'three columns', style={'text-align':'left', 'padding-left':'40px',}),

        # Welcome to Squiggle
        html.Div([
            html.H1('Network Analyzer', style={'fontColor':'white'})
        ], className = 'six columns', style={'margin': '0 auto'}),

        # Harvard Logo
        html.Div([
            html.A([
                html.Img(src='https://upload.wikimedia.org/wikipedia/en/thumb/2/29/Harvard_shield_wreath.svg/1200px-Harvard_shield_wreath.svg.png', 
                     style={'max-width':'20%', 'max-height':'20%'})       
            ], href='https://www.harvard.edu/')
        ], className = 'three columns', style={'text-align':'right', 'max-width':'100%', 'max-height':'100%'}),        
        
    ], className = 'row', style={'backgroundColor':'rgba(165,28,48, 0.7)',
                                 'text-align':'center', 'padding':'20px'}),
    
            
    
    # Drag the file
    # -------------
    
    html.Div([
        dcc.Upload(
            id='upload-data',
            children = html.Div([
                    'Drag and Drop or ',
                    html.A('Select a File')
            ], style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center'
            }),
            multiple=True
        )
    ], style = {'margin-top':'15px'}),
        
        
    # Plot the sequences
    # ------------------
    html.Div([

        # First box of first row

        html.Div([

            html.Div([
                html.H4('DNA Sequences')
            ], id = 'title1', style = {'text-align':'center'}),

            # Table of first box
            html.Div([
                    
#                dcc.Dropdown(id='mode_picker', options = modes, value = modes[0]['value']),
#                html.Div([
#                    dcc.Graph(id='graph'),
#                ], style={'visibility':'hidden'})
            ]),
            
            # Div container for uploaded data
            html.Div(id='output-data-upload')
        ])
    ])
])




if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port) ## Docker config