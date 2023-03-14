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

def compare_mota_idf1(scenes, MOTA, IDF1, MOTA_2, IDF1_2):
    '''
        plot 2 metrics in single plot
    '''
    ticks_size = 16
    labels_size = 18
    margin = 0.5

    MOTA = [mota*100 for mota in MOTA]
    IDF1 = [idf1*100 for idf1 in IDF1]
    MOTA_2 = [mota_2*100 for mota_2 in MOTA_2]
    IDF1_2 = [idf1_2*100 for idf1_2 in IDF1_2]

    fig, ax = plt.subplots()
    plt.plot(scenes, IDF1, label = "IDF1 rm FP", marker='o', c='b', linewidth=2, markersize=8)
    plt.plot(scenes, MOTA, label = "MOTA rm FP", marker='*', c='b', linewidth=2, markersize=8)
    plt.plot(scenes, IDF1_2, label = "IDF1", marker='o', c='r', linewidth=2, markersize=8)
    plt.plot(scenes, MOTA_2, label = "MOTA", marker='*', c='r', linewidth=2, markersize=8)
    y_min = min(min(IDF1), min(IDF1_2), min(MOTA), min(MOTA_2))-3
    y_max = max(max(IDF1), max(IDF1_2), max(MOTA), max(MOTA_2))+3
    plt.xlim(scenes[0]-margin, scenes[-1]+margin)
    plt.ylim(y_min, y_max)
    plt.xlabel("Base A", fontsize=labels_size)
    plt.ylabel("%", color="black", fontsize=labels_size)
    plt.xticks(scenes, fontsize=ticks_size)
    plt.yticks(np.arange(y_min, y_max, 2), fontsize=ticks_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    plt.legend()
    plt.grid(True)
    plt.show()
    # fig.savefig(os.path.join(figure_path, 'different_MOTAnIDF1_Base.jpg'))


def plot_mota_idf1_scatter(scenes, MOTA, IDF1, scenes_2, MOTA_2, IDF1_2, plot_compared_baseline=False, base_metric=0.3):
    '''
        plot idf1-mota plot of different method
        1 is the OH based
        2 is the fixed-duration based

        @ param: plot_compared_baseline
            We can only plot the metric above baseline to show improvement
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    margin = 2
    ticks_size = 16
    labels_size = 18
    ticks_indent = 1

    MOTA = [mota*100 for mota in MOTA]
    IDF1 = [idf1*100 for idf1 in IDF1]
    MOTA_2 = [mota_2*100 for mota_2 in MOTA_2]
    IDF1_2 = [idf1_2*100 for idf1_2 in IDF1_2]

    s = [(300+80*n) for n in range(len(scenes))]
    s2 = [(300+80*n) for n in range(len(scenes_2))]
    
    if plot_compared_baseline:
        # find baseline and plot marker, exclude baseline metric in scene1 and scene2
        base_index = -1
        try:
            base_index = scenes_2.index(base_metric)
        except ValueError as e:
            print('No secenes include baseline metric {}'.format(base_metric))
        
        if base_index != -1:
            scatter = ax.scatter(MOTA_2[base_index], IDF1_2[base_index], s=s2[base_index], alpha=0.7, linewidth=2, c='#AAFF00', edgecolors='green' , marker='^', label='baseline')
        
        MOTA_2 = MOTA_2[base_index+1:]
        IDF1_2 = IDF1_2[base_index+1:]
        scenes_2 = scenes_2[base_index+1:]

        base_index = -1
        try:
            base_index = scenes.index(1)
        except ValueError as e:
            print('No secenes include baseline metric {}'.format(base_metric))

        if base_index != -1:
            MOTA = MOTA[base_index+1:]
            IDF1 = IDF1[base_index+1:]
            scenes = scenes[base_index+1:]

    scatter1 = ax.scatter(MOTA, IDF1, s=s, alpha=0.7, linewidth=2, c='#FFFAAA', edgecolors='red' , marker='*', label='w/ OH: base A')
    scatter2 =  ax.scatter(MOTA_2, IDF1_2, s=s2, alpha=0.7, linewidth=2, c='#AAAFFF', edgecolors='blue', marker='p', label='w/o OH: fixed time(sec)')

    x_min = round(min(min(MOTA), min(MOTA_2))-margin)
    x_max = math.ceil(max(max(MOTA), max(MOTA_2))+margin)
    y_min = round(min(min(IDF1), min(IDF1_2))-margin)
    y_max = math.ceil(max(max(IDF1), max(IDF1_2))+margin)

    ax.grid(True)
    # set equal aspect ratio in both x and y axis 
    # ax.axis('equal')
    # scale plot size to autofit the graph
    if plot_compared_baseline:
        ax.axis('scaled')
    ax.set_xlabel('MOTA (%)', fontsize=labels_size)
    ax.set_ylabel('IDF1 (%)', fontsize=labels_size)
    plt.xticks(np.arange(x_min, x_max, ticks_indent), fontsize=ticks_size)
    plt.yticks(np.arange(y_min, y_max, ticks_indent), fontsize=ticks_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # only one marker is shown in lagend
    lgnd = plt.legend(scatterpoints=1)
    # change the individual marker size manually in legends
    lgnd.legendHandles[0]._sizes = [300]
    lgnd.legendHandles[1]._sizes = [300]
    if plot_compared_baseline:
        lgnd.legendHandles[2]._sizes = [300]

    for i, label in enumerate(scenes):
        plt.annotate(label, (MOTA[i], IDF1[i]+0.7), fontsize=18, color='r', ha='center', va='center')

    for i, label in enumerate(scenes_2):
        plt.annotate(label, (MOTA_2[i], IDF1_2[i]+0.7), fontsize=18, color='b', ha='center', va='center')
    plt.show()
    if plot_compared_baseline:
        fig.savefig(os.path.join(figure_path, 'equal_baseline_track_termination_mota_idf1_s4.0_no_ticks.jpg'))
    else:
        fig.savefig(os.path.join(figure_path, 'track_termination_mota_idf1_no_ticks_s4.0_ticks.jpg'))


def plot_mota_idf1(scenes, motas, idf1s, bases, outpath):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    scenes = [('lifetime: '+str(s)+' (sec)') for s in scenes]

    margin = 1
    ticks_size = 16
    labels_size = 18
    ticks_indent = 1

    for idx, mota in enumerate(motas) :
        motas[idx] = [m*100 for m in motas[idx]]
    for idx, idf1 in enumerate(idf1s):
        idf1s[idx] = [f*100 for f in idf1s[idx]]

    label_size = []
    for mota in motas:
        l_s = [(300+200*n) for n in range(len(mota))]
        label_size.append(l_s)
    
    print(motas, idf1s)

    color_codes = ['#FFFAAA', '#AAAFFF']
    markers = ['*', 'p']
    edge_colors = ['red', 'blue']

    for idx, mota in enumerate(motas):
        sca =  ax.scatter(mota, idf1s[idx], s=label_size[idx], alpha=0.7, linewidth=2, c=color_codes[idx], edgecolors=edge_colors[idx] , marker=markers[idx], label=scenes[idx])

    x_min = round(np.min(np.array(motas))-margin)
    x_max = round(np.max(np.array(motas))+margin)
    y_min = round(np.min(np.array(idf1s))-margin)
    y_max = round(np.max(np.array(idf1s))+margin)

    print('x_min: {}, x_max: {}'.format(x_min, x_max))
    print('y_min: {}, y_max: {}'.format(y_min, y_max))


    ax.grid(True)
    # set equal aspect ratio in both x and y axis 
    # ax.axis('equal')
    # scale plot size to autofit the graph
    ax.axis('scaled')
    ax.set_xlabel('MOTA (%)', fontsize=labels_size)
    ax.set_ylabel('IDF1 (%)', fontsize=labels_size)
    plt.xticks(np.arange(x_min, x_max, ticks_indent), fontsize=ticks_size)
    plt.yticks(np.arange(y_min, y_max, ticks_indent), fontsize=ticks_size)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # only one marker is shown in lagend
    lgnd = plt.legend(scatterpoints=1)
    # change the individual marker size manually in legends
    for lengend_han in lgnd.legendHandles:
        lengend_han._sizes = [300]

    print(bases)
    for i, mota in enumerate(motas):
        for j, b in enumerate(sorted(bases)):
            plt.annotate(b, (mota[j], idf1s[i][j]+0.3), fontsize=18, color=edge_colors[i], ha='center', va='center')

    plt.show()
    fig.savefig(os.path.join(outpath, 'idf1_mota_with_different_track_lifetime_and_base.jpg'))

def plot_y_MOTA_IDF1(scenes, MOTA, IDF1):
    print(scenes)
    print(MOTA)
    IDF1_p = [idf1*100 for idf1 in IDF1]
    MOTA_p = [mota*100 for mota in MOTA]
    ticks_size = 25 
    labels_size = 30

    fig, ax = plt.subplots()
    y_1_min = min(min(IDF1_p), min(MOTA_p))-2
    y_1_max = max(max(IDF1_p), max(MOTA_p))+2
    # plot lines
    ax.plot(scenes, MOTA_p, label = 'MOTA', c='r', linewidth=4, marker='o', markersize=7)
    ax.plot(scenes, IDF1_p, label = 'IDF1', c='b', linewidth=4, marker='o', markersize=7)
    # fixed
    # plt.xlabel('A_max (sec)', color='black', fontsize=labels_size)
    
    # occlusion
    # plt.xlabel('Base A', color='black', fontsize=labels_size)
    # occlusion linear
    plt.xlabel('Linear A', color='black', fontsize=labels_size)


    plt.ylabel('%', color='black', fontsize=labels_size)
    # plt.yticks(np.arange(y_1_min, y_1_max, 2), fontsize=ticks_size)
    # plt.ylim(y_1_min, y_1_max)

    # occlusion map y range same as track scale
    plt.yticks(np.arange(35, 59, 2), fontsize=ticks_size)
    plt.ylim(35, 59)

    plt.xticks(scenes, fontsize=ticks_size)
    start, end = ax.get_xlim()
    # fixed, show ticks freq to 0.5 second
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 0.5))
    # plt.xlim(0, end+0.1)
    
    # exponen occlusion incident 1  
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 1))
    # plt.xlim(0.5, end+0.5)

    # linear occlusion incident 0.5  
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 0.5))
    plt.xlim(-0.1, end+0.1)

    # set y ticks precision
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax.grid(axis='y', linewidth=1, linestyle='-', c='gray', alpha=0.7)
    plt.legend(loc='lower right', fontsize=ticks_size)
    plt.show()
    fig.savefig(os.path.join(figure_path, 'MOTA,IDF1_Duration_2.jpg'))

def plot_y_MOTA_IDF1_bolder(scenes, MOTA, IDF1):
    print(scenes)
    print(MOTA)
    IDF1_p = [idf1*100 for idf1 in IDF1]
    MOTA_p = [mota*100 for mota in MOTA]
    ticks_size = 65
    labels_size = 60

    fig, ax = plt.subplots()
    y_1_min = min(min(IDF1_p), min(MOTA_p))-2
    y_1_max = max(max(IDF1_p), max(MOTA_p))+2

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3.5)  # change width
        ax.spines[axis].set_color('black')    # change color

    # plot lines
    ax.plot(scenes, MOTA_p, label = 'MOTA', c='r', linewidth=9, marker='o', markersize=15)
    ax.plot(scenes, IDF1_p, label = 'IDF1', c='b', linewidth=9, marker='o', markersize=15)

    # fixed
    # plt.xlabel('A_max (sec)', color='black', fontsize=labels_size)
    # occlusion
    # plt.xlabel('Base A', color='black', fontsize=labels_size)
    # occlusion linear
    # plt.xlabel('Linear A', color='black', fontsize=labels_size)


    plt.ylabel('%', color='black', fontsize=labels_size)
    # plt.yticks(np.arange(y_1_min, y_1_max, 2), fontsize=ticks_size)
    # plt.ylim(y_1_min, y_1_max)

    # occlusion map y range same as track scale
    plt.yticks(np.arange(35, 59, 2), fontsize=ticks_size-10)
    plt.ylim(35, 59)

    plt.xticks(scenes, fontsize=ticks_size)
    start, end = ax.get_xlim()
    # fixed, show ticks freq to 0.5 second
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 0.5))
    # plt.xlim(0, end+0.1)
    # exponen occlusion incident 1  
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 2))
    # plt.xlim(0.5, end+0.5)
    # linear occlusion incident 0.5  
    ax.xaxis.set_ticks(np.arange(start, end+0.1, 0.5))
    plt.xlim(-0.1, end+0.1)

    # set y ticks precision
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_axisbelow(True)
    # ax.grid(axis='y', linewidth=1, linestyle='-', c='gray', alpha=0.7)
    ax.grid(linewidth=3, linestyle='-', c='gray', alpha=0.7)
    plt.legend(loc='lower right', fontsize=55)
    plt.show()
    fig.savefig(os.path.join(figure_path, 'MOTA,IDF1_Duration_border.jpg'))


def plot_y_MOTA_IDF1_boken_y(scenes, MOTA, IDF1):
    print(scenes)
    print(MOTA)
    IDF1_p = [idf1*100 for idf1 in IDF1]
    MOTA_p = [mota*100 for mota in MOTA]
    ticks_size = 25
    labels_size = 30

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0.05)

    y_1_min = min(min(IDF1_p), min(MOTA_p))-2
    y_1_max = max(max(IDF1_p), max(MOTA_p))+2
    # plot lines
    ax1.plot(scenes, MOTA_p, label = 'MOTA', c='r', linewidth=4, marker='o', markersize=7)
    ax1.plot(scenes, IDF1_p, label = 'IDF1', c='b', linewidth=4, marker='o', markersize=7)

    ax2.plot(scenes, MOTA_p, label = 'MOTA', c='r', linewidth=4, marker='o', markersize=7)
    ax2.plot(scenes, IDF1_p, label = 'IDF1', c='b', linewidth=4, marker='o', markersize=7)

    # include to lower 35-44
    ylim  = [y_1_min, y_1_max]
    ylim2 = [0, 5]
    ylimratio = (ylim[1]-ylim[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
    ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])

    ax1.set_ylim(ylim) #main data, top
    ax2.set_ylim(ylim2)   #no data

    # hide the spines between ax and ax2
    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()


    # plt.yticks(np.arange(35, y_1_max, 2), fontsize=ticks_size)
    plt.xticks(scenes, fontsize=ticks_size)
    start, end = ax2.get_xlim()

    xlim = [0.5, end+0.5]
    # plt.xlim(0.5, end+0.5)
    ax2.set_xlabel('Base A', color='black', fontsize=labels_size)
    ax2.set_ylabel('%', color='black', fontsize=labels_size)
    ax2.yaxis.set_label_coords(5, 2, transform=fig.transFigure)


    # Now, let's turn towards the cut-out slanted lines.
    # We create line objects in axes coordinates, in which (0,0), (0,1),
    # (1,0), and (1,1) are the four corners of the axes.
    # The slanted lines themselves are markers at those locations, such that the
    # lines keep their angle and position, independent of the axes size or scale
    # Finally, we need to disable clipping.

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    '''
    kwargs = dict(color='k', clip_on=False)
    xlim = ax1.get_xlim()
    dx = .02*(xlim[1]-xlim[0])
    dy = .01*(ylim[1]-ylim[0])/ylimratio
    ax1.plot((xlim[0]-dx,xlim[0]+dx), (ylim[0]-dy,ylim[0]+dy), **kwargs)
    ax1.plot((xlim[1]-dx,xlim[1]+dx), (ylim[0]-dy,ylim[0]+dy), **kwargs)
    dy = .01*(ylim2[1]-ylim2[0])/ylim2ratio
    ax2.plot((xlim[0]-dx,xlim[0]+dx), (ylim2[1]-dy,ylim2[1]+dy), **kwargs)
    ax2.plot((xlim[1]-dx,xlim[1]+dx), (ylim2[1]-dy,ylim2[1]+dy), **kwargs)
    '''
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    ax1.xaxis.set_ticks(np.arange(start, end+0.1, 1))
    ax2.xaxis.set_ticks(np.arange(start, end+0.1, 1))


    '''
    # fixed
    # plt.xlabel('A_max (sec)', color='black', fontsize=labels_size)
    # occlusion
    plt.xlabel('Base A', color='black', fontsize=labels_size)
    plt.ylabel('%', color='black', fontsize=labels_size)
    plt.yticks(np.arange(35, y_1_max, 2), fontsize=ticks_size)
    # plt.yticks(np.arange(y_1_min, y_1_max, 2), fontsize=ticks_size)
    # plt.ylim(y_1_min, y_1_max)

    plt.xticks(scenes, fontsize=ticks_size)
    start, end = ax1.get_xlim()
    # show ticks freq to 0.5 second
    # ax.xaxis.set_ticks(np.arange(start, end+0.1, 0.5))
    # incident 1 for occlusion
    ax1.xaxis.set_ticks(np.arange(start, end+0.1, 1))
    # plt.xlim(0.1, end+0.1)
    # occlusion
    plt.xlim(0.5, end+0.5)


    # set y ticks precision
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    '''
    # plt.grid(True)
    ax1.grid(axis='y', linewidth=1, linestyle='-', c='gray', alpha=0.7)
    ax2.grid(axis='y', linewidth=1, linestyle='-', c='gray', alpha=0.7)
    plt.legend(loc='lower right', fontsize=ticks_size)
    plt.show()
    fig.savefig(os.path.join(figure_path, 'MOTA,IDF1_Duration_broken.jpg'))


    # ax1.set_ylim(y_1_min, y_1_max) #main data, top
    # ax2.set_ylim(35, y_1_max)
    # ylim  = [y_1_min, y_1_max]
    # ylim2 = [0.0, 0.32]
    # ylimratio = (ylim[1]-ylim[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
    # ylim2ratio = (ylim2[1]-ylim2[0])/(ylim2[1]-ylim2[0]+ylim[1]-ylim[0])
    # gs = gridspec.GridSpec(2, 1, height_ratios=[ylimratio, ylim2ratio])
    # fig = plt.figure()
    # ax = fig.add_subplot(gs[0])
    # ax2 = fig.add_subplot(gs[1])
    # ax.plot(pts)
    # ax2.plot(pts)
    # ax.set_ylim(ylim)
    # ax2.set_ylim(ylim2)
    # plt.subplots_adjust(hspace=0.03)

    # ax.spines['bottom'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax.xaxis.tick_top()
    # ax.tick_params(labeltop='off')
    # ax2.xaxis.tick_bottom()

    # ax2.set_xlabel('xlabel')
    # ax2.set_ylabel('ylabel')
    # ax2.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)



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


