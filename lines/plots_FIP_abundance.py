# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:21:21 2022

@author: tania
"""
import pandas as pd
import numpy as np
from pathlib import Path
import math as m
from os.path import dirname, join as pjoin
from scipy.io import readsav
import seaborn as sns
import matplotlib.pyplot as plt


basename = Path("C:\\Users\\tania\\Documents")
lines = pd.read_csv(basename / "relevant_lines_studidcsv.csv", sep=",", header=0)

plt.figsize=[15,15]
for i in range(len(lines)):
    plt.plot(lines.at[i,'logT'], lines.at[i,'FIP'], marker='x', label = lines.at[i,'Window name'])
    plt.text(lines.at[i,'logT'], lines.at[i,'FIP'],lines.at[i,'ion'], fontsize=8)
plt.title('log(T) vs FIPs'), plt.xlabel('log(T)'), plt.ylabel('FIP'), plt.grid(True)
plt.show()
for i in range(len(lines)):
    plt.plot(lines.at[i,'logT'], lines.at[i,'Ab ratio'], marker='+', label = lines.at[i,'Window name'])
    plt.text(lines.at[i,'logT'], lines.at[i,'Ab ratio'],lines.at[i,'ion'], fontsize=8)
plt.title('log(T) vs abundance ratio'), plt.xlabel('log(T)'), plt.ylabel('abundance ratio (coronal/photo)'), plt.grid(True)
plt.show()

#%% by study_id

lines_by_sid = pd.read_csv(basename / "relevant_lines_studidcsv.csv", sep=",", header=0)
lines_by_sid = lines_by_sid[['Window index', 'Window name', 'Winow wvl min', 'Window wvl max', 'ion',
       'wvl', 'logT', 'intensity', 'coronal ab', 'photo ab', 'Ab ratio', 'FIP',
       '0', '2', '4', '5', '6', '8', '9', '10', '11', '12', '15', '16', '17',
       '18']]


lines0 = lines_by_sid.loc[(lines_by_sid['0'] == 'x')].reset_index()
lines2 = lines_by_sid.loc[(lines_by_sid['2'] == 'x')].reset_index()
lines4 = lines_by_sid.loc[(lines_by_sid['4'] == 'x')].reset_index()
lines5 = lines_by_sid.loc[(lines_by_sid['5'] == 'x')].reset_index()
lines6 = lines_by_sid.loc[(lines_by_sid['6'] == 'x')].reset_index()
lines8 = lines_by_sid.loc[(lines_by_sid['8'] == 'x')].reset_index()
lines9 = lines_by_sid.loc[(lines_by_sid['9'] == 'x')].reset_index()
lines10 = lines_by_sid.loc[(lines_by_sid['10'] == 'x')].reset_index()
lines11 = lines_by_sid.loc[(lines_by_sid['11'] == 'x')].reset_index()
lines12 = lines_by_sid.loc[(lines_by_sid['12'] == 'x')].reset_index()
lines15 = lines_by_sid.loc[(lines_by_sid['15'] == 'x')].reset_index()
lines16 = lines_by_sid.loc[(lines_by_sid['16'] == 'x')].reset_index()
lines17 = lines_by_sid.loc[(lines_by_sid['17'] == 'x')].reset_index()
lines18 = lines_by_sid.loc[(lines_by_sid['18'] == 'x')].reset_index()

list_df_lines = [lines0,lines2,lines4,lines5,lines6,lines8,lines9,lines10 ,lines11, lines12,lines15,
                 lines16 ,lines17,lines18]
list_df_lines_str = ['lines0','lines2','lines4','lines5','lines6','lines8','lines9','lines10','lines11' ,'lines12',
                     'lines15','lines16' ,'lines17','lines18']
k=0
for l in list_df_lines :
    fig=plt.figure(figsize = [15,8])
    for i in range(len(l)):
        plt.subplot(1,2,1)
        plt.plot(l.at[i,'logT'], l.at[i,'FIP'], marker='+')
        plt.text(l.at[i,'logT'], l.at[i,'FIP'],
                 l.at[i,'ion'] + '\n' + str(l.at[i,'wvl']),
                 fontsize=12)
    plt.title('log(T) vs FIPs  ' + list_df_lines_str[k]), plt.xlabel('log(T)')
    plt.xlim(right=6)
    plt.ylabel('FIP'), plt.grid(True)
    for i in range(len(l)):
        plt.subplot(1,2,2)
        plt.plot(l.at[i,'logT'], l.at[i,'Ab ratio'], marker='*')
        plt.text(l.at[i,'logT'], l.at[i,'Ab ratio'],
                 l.at[i,'ion'] + '\n' + str(l.at[i,'wvl']),
                 fontsize=12)
    plt.xlim(right=6)
    plt.title('log(T) vs abundance ratio  ' + list_df_lines_str[k]), plt.xlabel('log(T)') 
    plt.ylabel('ab ratio (coronal/photo)'), plt.grid(True)
    fig.tight_layout()
    plt.show()
    k+=1

    
#%% mosaic lines 
keys_idx = [20,24,45,40]
mosaic_lines = lines.loc[keys_idx].reset_index()
for i in range(len(mosaic_lines)):
    plt.plot(mosaic_lines.at[i,'logT'], mosaic_lines.at[i,'FIP'], marker='x',
             label = mosaic_lines.at[i,'Window name'])
    plt.text(mosaic_lines.at[i,'logT'], mosaic_lines.at[i,'FIP'],mosaic_lines.at[i,'ion'], fontsize=10)
plt.title('log(T) vs FIPs'), plt.xlabel('log(T)'), plt.ylabel('FIP'), plt.grid(True)
plt.show()
for i in range(len(mosaic_lines)):
    plt.plot(mosaic_lines.at[i,'logT'], mosaic_lines.at[i,'Ab ratio'], marker='+',
             label = mosaic_lines.at[i,'Window name'])
    plt.text(mosaic_lines.at[i,'logT'], mosaic_lines.at[i,'Ab ratio'],mosaic_lines.at[i,'ion'], fontsize=10)
plt.title('log(T) vs abundance ratio'), plt.xlabel('log(T)'), plt.ylabel('abundance ratio (coronal/photo)') 
plt.grid(True)
plt.show()

#%%

# with plt.xkcd():
#     fig1 = plt.figure()
#     for i in range(len(lines)):
#         plt.plot(lines.at[i,'logT'], lines.at[i,'FIP'], marker='o', label = lines.at[i,'Window name'])
#         plt.text(lines.at[i,'logT'], lines.at[i,'FIP'],lines.at[i,'ion'], fontsize=8)
#     plt.title('log(T) vs FIPs'), plt.xlabel('log(T)'), plt.ylabel('FIP'), plt.grid(True)
#     plt.show()
    