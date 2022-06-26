import numpy as np
import yaml
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os
import csv
import pandas as pd

'''
    Output result in single csv file and plot
'''

				
file_path = '/data/itri_output/tracking_output/output/track_lifespan_no_occlusion_w_merge/result'
figure_path = '/data/itri_output/tracking_output/output/track_lifespan_no_occlusion_w_merge/figure'

def output_result(scenes, MOTA, MOTP, TP, FP, FN, Pre, Rec, F1, Frag, IDSW, IDF1, MT, ML, Over_Seg, LOST_GT, NUM_GT, NUM_TRAs):
    output_file = os.path.join(file_path, 'overall_metric.csv')
    titles = ['scenes','mota', 'motp [m]', 'recall', 'precision', 'F1-socre', 'IDSW', 'FP', 
                            'FN', 'over-seg', 'Frag', 'IDF1', 'lost_trajectory', 'MT', 'ML', 'gt_num', 'object_num']
    with open(output_file, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(titles)
        for idx, f in enumerate(scenes):
            one = [scenes[idx], MOTA[idx], MOTP[idx], Rec[idx],
                    Pre[idx], F1[idx], IDSW[idx], FP[idx], FN[idx], Over_Seg[idx], Frag[idx], IDF1[idx], LOST_GT[idx], MT[idx], ML[idx], NUM_GT[idx], NUM_TRAs[idx]]
            
            assert len(titles) == len(one)
            writer.writerow(one)

def double_y(MOTA, IDF1, IDSW, Frag):
    # double y axis
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    line1, = ax.plot(scenes, [mota*100 for mota in MOTA], color="r", marker="x", label='MOTA')
    line2, = ax.plot(scenes, [idf1*100 for idf1 in IDF1], color="b", marker="x", label='IDF1')
    # set x-axis label
    ax.set_xlabel("Life of tracks [sec]", fontsize=18)
    # set y-axis label
    ax.set_ylabel("% (x)", color="red", fontsize=20)

    x = np.arange(len(scenes)) + 1 - 0.1
    xpos = np.arange(0.3, 1.1, 0.1, dtype=float)
    labels = np.array([str(s) for s in scenes])
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    line3, = ax2.plot(scenes, IDSW, color="c", marker="o", label ='IDSW')
    line4, = ax2.plot(scenes, Frag, color="y", marker="o", label ='FRAG')
    # line3 = ax2.bar(x+0.05, IDSW, width=0.1, color='lightblue', label ='IDSW', align='center')
    # line4 = ax2.bar(x-0.05, Frag, width=0.1, color="yellowgreen", label ='FRAG', align='center')
    ax2.set_ylabel("Number (o)", color="blue", fontsize=20)
    
    ax.legend(handles =[line1, line2, line3, line4], loc ='lower left')
    # work
    plt.xlim(float(scenes[0])-0.05, float(scenes[-1])+0.05)
    # # plt.xticks(x + 0.05 / 2, labels)
    # plt.xticks(x, labels)
    # plt.xticks(xpos, labels)

    plt.grid(True)
    plt.show()

def double_y_pd(scenes, MOTA, IDF1, IDSW, Frag):
    df = pd.DataFrame({"range": scenes,"IDSW":np.array(IDSW), "Frag":np.array(Frag), "IDF1": np.array([idf1*100 for idf1 in IDF1]), "MOTA": np.array([mota*100 for mota in MOTA])})

    width = 0.25
    margin = 0.5
    ticks_size = 16
    labels_size = 18
    fig = plt.figure()
    ax = df['IDF1'].plot(marker='o', c='b', linewidth=3, label='IDF1', zorder=0)
    ax = df['MOTA'].plot(marker='o', c='r', linewidth=3, label='MOTA', zorder=1)
    ax.set_ylabel("%", color="black", fontsize=labels_size)
    ax.set_xticklabels(df['range'], fontsize=ticks_size)
    # ticks value size
    ax.set_yticklabels(ax.get_yticks(), rotation=0, fontsize=ticks_size)
    # set value prcision(no decimal point)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    df['IDSW'].plot(kind="bar", alpha=0.6, color='lightblue', position=0, width=width,label='IDSW')
    df['Frag'].plot(kind="bar", alpha=0.6, color='yellowgreen', position=1, width=width, label='Frag')
    ax2.set_ylabel("Number", color="black", fontsize=labels_size)
    ax2.set_xticklabels(df['range'], fontsize=ticks_size)
    ax2.set_yticklabels(ax2.get_yticks(), rotation=0, fontsize=ticks_size)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.set_xlabel("Life of Tracks [sec]", fontsize=18)
    ax.set_ylim(0,1.2*max(df["IDF1"].max(), df["MOTA"].max()))
    ax2.set_ylim(0,1.2*max(df["IDSW"].max(), df["Frag"].max()))
    ax2.set_xlim(-margin, len(scenes)-1+margin)
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.title("MOTA.IDF1.IDSW.FRAG - Lifetime", fontsize=22) 
    plt.show()
    # fig.savefig(os.path.join(figure_path, 'MOTA,IDF1,IDSW,FRAG_Lifetime_2.jpg'))

def plot_num_bar_plot(scenes, FP, FN, IDSW):
    labels = np.array([str(s) for s in scenes])
    width = 0.3
    margin = 1
    ticks_size = 16
    labels_size = 18
    x = np.arange(len(scenes))
    fig = plt.figure()
    plt.bar(x+width, IDSW, width=width, color='yellow', label ='IDSW', align='center')
    plt.bar(x, FP, width=width, color='lightblue', label ='FP', align='center')
    plt.bar(x-width, FN, width=width, color='yellowgreen', label ='FN', align='center')
    plt.legend(loc='upper left')
    plt.xlabel("Life of Tracks [sec]", fontsize=labels_size)
    plt.ylabel("Number", color="black", fontsize=labels_size)
    plt.title("FP.FN.IDSW - Lifetime", fontsize=22)
    plt.xlim(-margin, len(scenes)-1+margin)
    plt.xticks(x, labels, fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.show()
    fig.savefig(os.path.join(figure_path, 'FP,FN,IDSW_Lifetime_1.jpg'))

if __name__ == '__main__':
    files = sorted(os.listdir(file_path))
    scenes = [float(f.split('_')[0]) for f in files]

    MOTA = []
    MOTP = []
    TP = []
    FP = []
    FN = []
    Pre = []
    Rec = []
    F1 = []
    Frag = []
    IDSW = []
    IDF1 = []
    MT = []
    ML = []
    Over_Seg = []
    LOST_GT = []
    NUM_GT = []
    NUM_TRAs = []

    for f in files:
        metrics = {}
        with open(os.path.join(file_path, f, 'metrics.json'), "r") as m:
            metrics = json.load(m)

        MOTA.append(float(metrics['mota']))
        MOTP.append(metrics['motp'])
        TP.append(metrics['TP'])
        FP.append(metrics['FP'])
        FN.append(metrics['FN'])
        Pre.append(metrics['precision'])
        Rec.append(metrics['recall'])
        F1.append(metrics['F1-socre'])
        Frag.append(int(metrics['Frag']))
        IDSW.append(int(metrics['IDSW']))
        IDF1.append(float(metrics['IDF1']))
        MT.append(metrics['MT'])
        ML.append(metrics['ML'])
        Over_Seg.append(metrics['over-seg'])
        LOST_GT.append(metrics['lost_trajectory'])
        NUM_GT.append(metrics['gt_num'])
        NUM_TRAs.append(metrics['trajectory_num'])

    # output
    # output_result(scenes, MOTA, MOTP, TP, FP, FN, Pre, Rec, F1, Frag, IDSW, IDF1, MT, ML, Over_Seg, LOST_GT, NUM_GT, NUM_TRAs)

    # not clear for line plot
    # double_y(MOTA, IDF1, IDSW, Frag)

    # bar and line plot
    double_y_pd(scenes, MOTA, IDF1, IDSW, Frag)

    # plot FP FN IDSW NUM BAR PLOT
    plot_num_bar_plot(scenes, FP, FN, IDSW)


