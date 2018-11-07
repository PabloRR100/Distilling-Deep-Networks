#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:00:38 2018
@author: pabloruizruiz
"""

'''

TODO 
====

 - [] Include triaining parameters choice for the displayed summary

'''


import numpy as np
import pandas as pd

import dash
import plotly.figure_factory as ff
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
net = torch.load('torchnet.pkl')
from utils import count_parameters
## TODO Import model by drag and drop a .pkl

Winp_stats = net.weight_stats['Winp']
Whid_stats = net.weight_stats['Whid']
Wout_stats = net.weight_stats['Wout']
Winp_scale = net.weight_stats['rWinp']
Whid_scale = net.weight_stats['rWhid']
Wout_scale = net.weight_stats['rWout']


## TODO Incrust these onto the model being trained so they are kept
LR = 0.1
WD = 0
MOMENTUM = 0
EPOCHS = 10
BATCHSIZE = 64
OPTIM = 'SGD'


ls = net.n_lay
xaxis = list( range(len(Winp_stats['mean'])))

#fig, axs = plt.subplots(nrows=2+ls, ncols=2)
#sns.lineplot(xaxis, Winp_stats['mean'], label='WInp mean', ax=axs[0,0])
#sns.lineplot(xaxis, Winp_stats['var'], label='WInp var', ax=axs[0,1])
#for l in range(ls):
#    sns.lineplot(xaxis, Whid_stats[l]['mean'], label='W{} mean'.format(l+1), ax=axs[l+1,0])
#    sns.lineplot(xaxis, Whid_stats[l]['var'], label='W{} var'.format(l+1), ax=axs[l+1,1])    
#sns.lineplot(xaxis, Wout_stats['mean'], label='WOut mean', ax=axs[ls+1,0])
#sns.lineplot(xaxis, Wout_stats['var'], label='WOut var', ax=axs[ls+1,1])
#plt.plot()


# Network Paramaters
model_table = pd.DataFrame(columns=['# Layers', 'Layer width', '# Parameters'])
model_value = [[net.n_lay, net.fcHid[0].in_features, count_parameters(net)]]

for i, r in enumerate(model_value):
    model_table.loc[i] = r
model_table = model_table.T.reset_index()
model_table.columns = ['Parameter', 'Value']


# Training Paramaters
trian_table = pd.DataFrame(columns=['Bach size', 'Optimizer', 'Weight Decay', 'Learning Rate', 'Momentum', 'Epochs'])
train_value = [[BATCHSIZE, OPTIM, WD, LR, MOMENTUM, EPOCHS]]

for i, r in enumerate(train_value):
    trian_table.loc[i] = r
trian_table = trian_table.T.reset_index()
trian_table.columns = ['Parameter', 'Value']


def draw_Table(data, title):
    c = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
    table = ff.create_table(data, colorscale=c, index=False)
    table.layout.title = title
    table.layout.height = 240
    table.layout.margin = {'b': 30, 'l': 20, 'r': 20, 't': 50}
    table.layout.paper_bgcolor='rgba(0,0,0,0)'
    table.layout.plot_bgcolor='rgba(0,0,0,0)'

    for i in range(len(table.layout.annotations)):
        table.layout.annotations[i].align = 'center'
    return table


from plotly import tools
import plotly.graph_objs as go

def plotweights():

    # Gather data
    c = ['rgba(55, 128, 191, 0.7)', 'rgba(219, 64, 82, 0.7)']
    trace11 = go.Scatter(x=xaxis, y=Winp_stats['mean'], marker=dict(color=c[0]))
    trace12 = go.Scatter(x=xaxis, y=Winp_stats['var'], marker=dict(color=c[1]))
    
    trace21 = go.Scatter(x=xaxis, y=Wout_stats['mean'], marker=dict(color=c[0]))
    trace22 = go.Scatter(x=xaxis, y=Wout_stats['var'], marker=dict(color=c[1]))
    
    # Make grid
    traceshid = list()
    for l in range(ls):
        trace1 = go.Scatter(x=xaxis, y=Whid_stats[l]['mean'], marker=dict(color=c[0]))
        trace2 = go.Scatter(x=xaxis, y=Whid_stats[l]['var'], marker=dict(color=c[1]))
        traceshid.append((trace1, trace2))
    
    titles = ['Mean W Inp', 'Var W Inp']
    for l in range(ls):        
        titles += ['Mean W Hid {}'.format(l)] 
        titles += ['Var W Hid {}'.format(l)]
    titles += ['Mean W Out', 'Var W Out']
    fig = tools.make_subplots(rows=2+ls, cols=2, subplot_titles=(titles))
    
    # Plot data by subplot
    fig.append_trace(trace11, 1, 1)
    fig.append_trace(trace12, 1, 2)
    for l in range(ls):
        fig.append_trace(traceshid[l][0], l+2, 1)
        fig.append_trace(traceshid[l][1], l+2, 2)     
    fig.append_trace(trace21, ls+2, 1)
    fig.append_trace(trace22, ls+2, 2)
    fig['layout'].update(
            height=200*(ls)+200, 
            xaxis = dict(showline=False),
            yaxis = dict(), 
            showlegend = False,
            title='Evolution of the parameters')
    return fig


#xaxis=dict(
#        autorange=True,
#        showgrid=False,
#        zeroline=False,
#        showline=False,
#        ticks='',
#        showticklabels=False
#    ),
#    yaxis=dict(
#        autorange=True,
#        showgrid=False,
#        zeroline=False,
#        showline=False,
#        ticks='',
#        showticklabels=False
#    )


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
        
    ], className = 'row', style={'backgroundColor':'rgba(165,28,48, 0.6)',
                                 'text-align':'center', 'padding':'20px'}),
    
            
    
    # Drag the file
    # -------------
    
    html.Div([
        dcc.Upload(
            id='upload-data',
            children = html.Div([
                    'Drag and Drop your Pickled Model or',
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
    ], style={'margin-top':'20px', 'margin-bottom':'20px', 'padding':'20px'}),
            
            
    html.Div([
    ], className = 'row', style = {'height':'20px'}),
       
            
        
    # Summary of Parameters Chosen
    # ----------------------------
    html.Div([
        
        # Model Parameters
        html.Div([
            dcc.Graph(id='table_net', figure=draw_Table(model_table, 'Model Parameters'))
        ], className = 'six columns', style = {'text-align':'center'}),
       
        # Training Parameters
        html.Div([
            dcc.Graph(id='table_train', figure=draw_Table(trian_table, 'Training Parameters'))
        ], className = 'six columns', style = {'text-align':'center'}),
            
    ], className='row', style={}),
        
        
    # Plot the paameter statistics
    # ----------------------------
    html.Div([

        # First box of first row

        html.Div([

            html.Div([
                html.H4('')
            ], id = 'title1', style = {'text-align':'center'}),

            # Table of first box
            html.Div([
                    
#                dcc.Dropdown(id='mode_picker', options = modes, value = modes[0]['value']),
                html.Div([
                    dcc.Graph(id='graph', figure=plotweights()),
                ])
            ]),
            
            # Div container for uploaded data
            html.Div(id='output-data-upload')
        ])
            
    ], style={'padding-right':'30px', 'padding-left':'30px'})
    
], style={'padding-right':'5px', 'padding-left':'5px'})




if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=port) ## Docker config