#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

import random, math

labelsize_b = 14
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#1C4670','#FF9636','#9D5FFB','#21B6A8','#D65780']
# colors = ['#ED4974','#16B9E1','#58DE7B','#F0D864','#FF8057','#8958D3']
# colors =['#FD0707','#0D0DDF','#129114','#DDDB03','#FF8A12','#8402AD']
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
colors = ["#1f78b4", "#33a02c", "#e31a1c", "#6a3d9a", "#fdbf6f", "#ff7f00"]
# colors = ["#006d2c", "#31a354", "#74c476", "#bae4b3", "#ececec", "#969696"]
colors = ["#004c6d", "#f18f01", "#81b214", "#c7243a", "#6b52a1", "#a44a3f"]

markers = ['s','o','^','v','D','<','>','P','*'] 
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
linestyles = ['solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
methods3 = ['Replication (Optimal)','Partition (Ours)','Standalone']
methods6 = ['Ours','Baseline','Optimal$^{(2)}$','Ours*','Baseline*','Optimal$^{(2)}$*']
from collections import OrderedDict
linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('dashed',              (0, (5, 5))),
     ('dotted',              (0, (1, 5))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('densely dashed',      (0, (5, 1))),

     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
linestyles = []
for i, (name, linestyle) in enumerate(linestyle_dict.items()):
    if i >= 9:break
    linestyles += [linestyle]

from matplotlib.patches import Ellipse

def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,legloc='best',linestyles=linestyles,
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(60,0.6),markersize=8,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,use_probarrow=False,
				rotation=None,use_resnet56_2arrow=False,use_resnet56_3arrow=False,use_resnet56_4arrow=False,use_resnet50arrow=False,use_re_label=False,
				use_prob_annot=False,use_connarrow=False,lgsize=None,oval=False,scatter_soft_annot=False,markevery=4,annot_aw=None):
	if lgsize is None:
		lgsize = lbsize
	if get_ax==1:
		ax = plt.subplot(211)
	elif get_ax==2:
		ax = plt.subplot(212)
	else:
		fig, ax = plt.subplots()
	ax.grid(zorder=0)
	handles = []
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if logx:
			xx = np.log10(np.array(xx))
		if oval:
			width = np.std(xx); height = np.std(yy)
			xi = np.mean(xx); yi = np.mean(yy)
			if width>0 and height>0:
				ellipse = Ellipse((xi, yi), width, height, edgecolor=None, facecolor=color[i],label = label[i], )
				ax.add_patch(ellipse)
				handles.append(ellipse)
				# plt.errorbar(xi, yi, yerr=height, color = color[i], label = label[i], linewidth=0, capsize=0, capthick=0)
			else:
				error_bar = plt.errorbar(xi, yi, yerr=height, color = color[i], label = label[i], linewidth=4, capsize=8, capthick=3)
				handles.append(error_bar)
		else:
			if yerr is None:
				plt.plot(xx, yy, color = color[i], marker = markers[i], 
					label = label[i], linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery)
			else:
				if markersize > 0:
					plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
						marker = markers[i], label = label[i], 
						linestyle = linestyles[i], 
						linewidth=linewidth, markersize=markersize, markerfacecolor='none', markevery=markevery,
						capsize=4)
				else:
					plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
						label = label[i], 
						linewidth=linewidth,
						capsize=4)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	plt.xticks(fontsize=lbsize)
	plt.yticks(fontsize=lbsize)
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lbsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lbsize)
	if xticklabel is not None:
		ax.set_xticklabels(xticklabel)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=45, size=lbsize,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	if use_connarrow:
		ax.annotate(text='', xy=(XX[0][5],YY[0][5]), xytext=(XX[0][5],0), arrowprops=dict(arrowstyle='|-|',lw=4))
		ax.text(XX[0][5]+7, YY[0][5]/2, "50% loss", ha="center", va="center", rotation='vertical', size=lbsize,fontweight='bold')
	if use_probarrow:
		ax.annotate(text='', xy=(XX[0][1],YY[0][1]), xytext=(XX[1][1],YY[1][1]), arrowprops=dict(arrowstyle='<->',lw=4))
		if YY[0][1]/YY[1][1]>10:
			ax.text(
			    XX[0][1]+0.1, (YY[0][1]+YY[1][1])/2, f"{YY[0][1]/YY[1][1]:.1f}"+r"$\times$", ha="center", va="center", rotation='vertical', size=44,fontweight='bold')
		else:
			ax.text(
			    XX[0][1]-0.07, (YY[0][1]+YY[1][1])/2, f"{YY[0][1]/YY[1][1]:.1f}"+r"$\times$", ha="center", va="center", rotation='vertical', size=44,fontweight='bold')
	if use_re_label:
		baselocs = [];parlocs = []
		for i in [1,2,3]:
			baselocs += [np.argmax(y[i]-y[0])]
			# parlocs += [np.argmax(y[i]-y[4])]
		for k,locs in enumerate([baselocs]):
			for i,loc in enumerate(locs):
				ind_color = '#4f646f'
				ax.annotate(text='', xy=(XX[0][loc],YY[0 if k==0 else 4,loc]), xytext=(XX[0][loc],YY[i+1,loc]), arrowprops=dict(arrowstyle='|-|',lw=5-k*2,color=ind_color))
				h = YY[k,loc]-5 if k==0 else YY[i+1,loc]+4
				w = XX[0][loc]-3 if k==0 else XX[0][loc]
				if k==0 and i==1:
					h-=1;w+=3
				if i==0:
					ax.text(w, h, '2nd', ha="center", va="center", rotation='horizontal', size=20,fontweight='bold',color=ind_color)
					ax.annotate(text='Consistency', xy=(45,67), xytext=(-5,65), size=22,fontweight='bold', arrowprops=dict(arrowstyle='->',lw=2,color='k'))
				elif i==1:
					ax.text(w, h-3, '3rd', ha="center", va="center", rotation='horizontal', size=20,fontweight='bold',color=ind_color)
				elif i==2:
					ax.text(w, h, '4th', ha="center", va="center", rotation='horizontal', size=20,fontweight='bold',color=ind_color)
	if scatter_soft_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			if i==0:
				ax.annotate('Most Reliable\nMost Computation', xy=(xx[i],yy[i]), xytext=(xx[i]-50,yy[i]+2),color = color[i], fontsize=lbsize-4,arrowprops=dict(arrowstyle='->',lw=2))
			elif i==1:
				ax.annotate('Least Computation\nMost Unreliable', xy=(xx[i],yy[i]), xytext=(xx[i]-5,yy[i]-4),color = color[i], fontsize=lbsize-4,arrowprops=dict(arrowstyle='->',lw=2))
	if use_prob_annot:
		xx,yy = XX[-1],YY[-1]
		# [0.6,0.8,0.9]
		# [1.3303288776709477, 1.8755448348252626, 2.4045152038996433]
		# ax.annotate('Naive neuron sharing: 174', xy=(xx[2],yy[2]+0.1), xytext=(60,2.15), fontsize=lbsize,fontweight='bold')
	if annot_aw is not None:
		if annot_aw == 0:
			ax.annotate(text='0.37% to max', xy=(0.4,97.28), xytext=(0.45,90), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		elif annot_aw == 1:
			ax.annotate(text='0.08% to max', xy=(0.4,75.21), xytext=(0.45,73.5), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		else:
			ax.annotate(text='0.7% inflation', xy=(0.4,0.7), xytext=(0.05,1.4), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
	
	if use_resnet50arrow:
		ax.annotate(text='', xy=(0,64), xytext=(40,64), arrowprops=dict(arrowstyle='<->',lw=2))
		ax.text(
		    20, 65, "Stage#0", ha="center", va="center", rotation='horizontal', size=lgsize,fontweight='bold')
		ax.annotate(text='', xy=(40,70), xytext=(160,70), arrowprops=dict(arrowstyle='<->',lw=2))
		ax.text(
		    100, 69, "Stage#1", ha="center", va="center", rotation='horizontal', size=lgsize,fontweight='bold')
		ax.annotate(text='LR = 0.1', xy=(0,68), xytext=(-8,75), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		ax.annotate(text='LR = 0.01', xy=(80,72.2), xytext=(40,74), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		ax.annotate(text='LR = 0.001', xy=(120,75), xytext=(100,73), arrowprops=dict(arrowstyle='->',lw=2),size=lgsize,fontweight='bold')
		# for l,r,lr in [(0,40,0.1),(40,80,0.1),(80,120,0.01),(120,160,0.001)]:
		# 	ax.annotate(text='', xy=(l,64), xytext=(r,64), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		# 	ax.text(
		# 	    (l+r)/2, 64.5, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lgsize)
		else:
			if bbox_to_anchor is None:
				plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol)
			else:
				if oval:
					plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor, handles=handles)
				else:
					plt.legend(loc=legloc,fontsize = lgsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	if get_ax!=0:
		return ax
	fig.savefig(path,bbox_inches='tight')
	plt.close()

# breakdown for different traces
def plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,filename,ncol,bbox_to_anchor,lim1=20,lim2=1500,title_posy=0.2,ratio=0.8):
	# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
	y_pos = np.arange(len(labels))
	latency_breakdown_mean = np.array(latency_breakdown_mean).transpose((1,0))*1000
	latency_breakdown_std = np.array(latency_breakdown_std).transpose((1,0))*1000
	width = 0.8
	plt.rcdefaults()
	fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
	for ax in [ax1,ax2]:
		left = np.array([0.0]*len(labels))
		for i in range(len(latency_types)):
			ax.barh(y_pos, latency_breakdown_mean[i], width, color=colors[i], xerr=latency_breakdown_std[i], left=left, 
				label=latency_types[i], align='center')
			left += latency_breakdown_mean[i]
	ax1.set_xlim(0,lim1)
	ax2.set_xlim(lim1,lim2)
	ax1.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax1.yaxis.tick_left()
	d = .03 # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax1.transAxes, color='r', clip_on=False)
	ax1.plot((1-d,1+d), (-d,+d), **kwargs)
	ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d,+d), (1-d,1+d), **kwargs)
	ax2.plot((-d,+d), (-d,+d), **kwargs)

	ax1.set_yticks(y_pos, labels=labels, fontsize = 14)

	handles, labels = ax2.get_legend_handles_labels()
	fig.legend(handles, labels,bbox_to_anchor=bbox_to_anchor,ncol=ncol,fancybox=True, loc='upper center', fontsize=12)
	ax1.invert_yaxis()  
	fig.text(0.5, title_posy, 'Latency breakdown (ms)', fontsize = 14, ha='center')
	for ax in [ax1,ax2]:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(filename,bbox_inches='tight')


def measurements_to_cdf(latency,epsfile,labels,xticks=None,xticklabel=None,linestyles=linestyles,colors=colors,
                        xlabel='Normalized QoE',ylabel='CDF',ratio=None,lbsize = 18,lfsize = 18,linewidth=4,bbox_to_anchor=(0.5,-.5),
                        loc='upper center',ncol=3,use_arrow=False,arrow_rotation=-45,arrow_coord=(0,0)):
    # plot cdf
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    for i,latency_list in enumerate(latency):
        N = len(latency_list)
        cdf_x = np.sort(np.array(latency_list))
        cdf_p = np.array(range(N))/float(N)
        plt.plot(cdf_x, cdf_p, color = colors[i], label = labels[i], linewidth=linewidth, linestyle=linestyles[i])
        print([cdf_x[int(N//2)],cdf_x[int(N*99/100)],cdf_x[int(N*999/1000)]])
    plt.xlabel(xlabel, fontsize = lbsize)
    plt.ylabel(ylabel, fontsize = lbsize)
    if use_arrow:
    	ax.text(arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=arrow_rotation if arrow_rotation!=180 else 0, size=lbsize, bbox=dict(boxstyle="larrow,pad=0.3" if arrow_rotation!=180 else "rarrow,pad=0.3", fc="white", ec="black", lw=2))
    if xticks is not None:
        plt.xticks(xticks,fontsize=lbsize)
    if xticklabel is not None:
        ax.set_xticklabels(xticklabel)
    if ratio is not None:
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    if bbox_to_anchor is not None:
    	plt.legend(loc=loc,fontsize = lfsize,bbox_to_anchor=bbox_to_anchor, fancybox=True,ncol=ncol)
    else:
    	plt.legend(loc=loc,fontsize = lfsize, fancybox=True,ncol=ncol)
    plt.tight_layout()
    fig.savefig(epsfile,bbox_inches='tight')
    plt.close()

def analyze_all_recorded_traces():
    print('Analyzing all recorded traces...')
    selected_batch_latency = []
    y = []
    yerr = []
    trpt = []
    trpterr = []
    # batch 1,2,4,8,16,32
    trace_filenames = []
    trace_filenames += [f'../DCN/{22*i:06d}' for i in [1,2,4,8,16,32,64]]
    trace_filenames += [f'../DCN-244/{244*i:06d}' for i in [1,2,4,8,16,32,64]]
    # trace_filenames += [f'../WAN/{12*i:06d}' for i in [1,2,4,8,16,32,64]]
    # trace_filenames += [f'../WAN-768/{768*i:06d}' for i in [1,2,4,8,16,32,64]]
    latency_mean_list = []
    latency_std_list = []
    trpt_mean_list = []
    trpt_std_list = []
    all_latency_list = []
    for tidx,filename in enumerate(trace_filenames):
        latency_list = []
        with open(filename,'r') as f:
            for l in f.readlines():
                l = l.strip().split(' ')
                latency_list += [float(l[0])/1e3]
            if len(latency_list)>=10000:
            	latency_list = latency_list[:10000]
            else:
            	latency_list = latency_list[:1000]
            	latency_list = latency_list*10
        all_latency_list += [latency_list]
    import csv
    with open('../curr_videostream.csv', mode='r') as csv_file:
    # with open('../curr_httpgetmt.csv', mode='r') as csv_file:
        # read network traces 
        csv_reader = csv.DictReader(csv_file)
        latency_list = []
        latency224_list = []
        num_of_line = 0
        bandwidth_list = []
        for row in csv_reader:
            # if row["bytes_sec_interval"] == 'NULL':
            #     continue
            bandwidth_list += [float(row["downthrpt"])]
            # bandwidth_list += [float(row["bytes_sec_interval"])]
            for bs in [2**i for i in range(7)]:
                query_size = 3*32*32*4*bs # bytes
                latency_list += [query_size/float(row["downthrpt"]) ]
                # latency_list += [query_size/float(row["bytes_sec_interval"]) ]
                query_size = 3*224*224*4*bs
                latency224_list += [query_size/float(row["downthrpt"])]
                # latency224_list += [query_size/float(row["bytes_sec_interval"])]
            num_of_line += 1
            if num_of_line==10000:break
        all_latency_list += np.array(latency_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
        all_latency_list += np.array(latency224_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
    all_latency_list = np.array(all_latency_list)
    all_latency_list = all_latency_list.mean(axis=-1).reshape(4,7)
    query_size = 3*32*32*4*np.array([2**(i) for i in range(7)])
    bw = query_size/all_latency_list[0]/1e6*8
    print(bw.mean(),bw.std(),'MBps',np.array(bandwidth_list).mean()*8,np.array(bandwidth_list).std()*8)
    ratio = all_latency_list[2:,]/all_latency_list[:2]
    labels = ['ResNet-56','ResNet-50']
    ratio = 1/ratio+1
    linestyles_ = ['solid']*10
    x = [[2**(i) for i in range(7)] for _ in range(len(labels))]
    line_plot(x, ratio,labels,colors,'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_cost.eps','Throughput (# of Batches)','Relative Latency',
    	lbsize=24,linewidth=2,markersize=8,markevery=1,linestyles=linestyles_)	


def bar_plot(avg,std,label,path,color,ylabel,labelsize=24,yticks=None,ylim=None):
	N = len(avg)
	ind = np.arange(N)  # the x locations for the groups
	width = 0.5       # the width of the bars
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	ax.set_axisbelow(True)
	if std is not None:
		hbar = ax.bar(ind, avg, width, color=color, \
			yerr=std, error_kw=dict(lw=1, capsize=1, capthick=1))
	else:
		hbar = ax.bar(ind, avg, width, color=color, \
			error_kw=dict(lw=1, capsize=1, capthick=1))
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.set_ylabel(ylabel, fontsize = labelsize)
	ax.set_xticks(ind,fontsize=labelsize)
	ax.set_xticklabels(label, fontsize = labelsize)
	ax.bar_label(hbar,fontsize = labelsize,fontweight='bold',padding=8)
	if yticks is not None:
		plt.yticks( yticks,fontsize=18 )
	# xleft, xright = ax.get_xlim()
	# ybottom, ytop = ax.get_ylim()
	# ratio = 0.3
	# ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()

def plot_computation_dist(flops,labels,filename,horizontal,bbox_to_anchor=None,ratio=None):
	flops = np.array(flops).transpose((1,0))
	node_types = ['Server #0','Server #1']
	y_pos = np.arange(flops.shape[1])
	width = 0.5
	plt.rcdefaults()
	fig, ax = plt.subplots()
	left = np.zeros(flops.shape[1])
	for i in range(2):
		if horizontal:
			ax.barh(y_pos, flops[i], width, color=colors[i], left=left, linewidth=2,
				label=node_types[i], hatch=hatches[i], align='center', edgecolor='k')
			left += flops[i]
		else:
			ax.bar(y_pos, flops[i], width, color=colors[i], bottom=left, linewidth=2,
				label=node_types[i], hatch=hatches[i], align='center', edgecolor='k')
			left += flops[i]
			if i==1:
				for j in range(2,flops.shape[1]):
					ax.text(j-0.6,left[j]+5,'$\downarrow$'+f'{200-left[j]:.0f}%',fontsize = 14, fontweight='bold')
		
	if horizontal:
		ax.set_yticks(y_pos, labels=labels, fontsize = 14 )
		plt.xlabel('FLOPS (%)', fontsize = 14)
	else:
		ax.set_xticks(y_pos, labels=labels, fontsize = 14,rotation=45)
		plt.ylabel('FLOPS (%)', fontsize = 14)
	if bbox_to_anchor is not None:
		plt.legend(bbox_to_anchor=bbox_to_anchor,ncol=1,fancybox=True, loc='upper center', fontsize=14)
	else:
		plt.legend(ncol=1,fancybox=True, loc='best', fontsize=14)
	if ratio is not None:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(filename,bbox_inches='tight')


def groupedbar(data_mean,data_std,ylabel,path,yticks=None,envs = [2,3,4],
				methods=['Ours','Standalone','Optimal','Ours*','Standalone*','Optimal*'],use_barlabel_x=False,use_barlabe_y=False,
				ncol=3,bbox_to_anchor=(0.46, 1.28),sep=1.25,width=0.15,xlabel=None,legloc=None,labelsize=labelsize_b,ylim=None,
				use_downarrow=False,rotation=None,lgsize=None,yticklabel=None,latency_annot=False,bandwidth_annot=False,latency_met_annot=False):
	if lgsize is None:
		lgsize = labelsize
	fig = plt.figure()
	ax = fig.add_subplot(111)
	num_methods = data_mean.shape[1]
	num_env = data_mean.shape[0]
	center_index = np.arange(1, num_env + 1)*sep
	# colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
	# colors = ['coral', 'orange', 'green', 'cyan', 'blue']
	# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


	ax.grid()
	ax.spines['bottom'].set_linewidth(3)
	ax.spines['top'].set_linewidth(3)
	ax.spines['left'].set_linewidth(3)
	ax.spines['right'].set_linewidth(3)
	if rotation is None:
		plt.xticks(center_index, envs, size=labelsize)
	else:
		plt.xticks(center_index, envs, size=labelsize, rotation=rotation)
	plt.xticks(fontsize=labelsize)
	plt.yticks(fontsize=labelsize)
	ax.set_ylabel(ylabel, size=labelsize)
	if xlabel is not None:
		ax.set_xlabel(xlabel, size=labelsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=labelsize)
	if yticklabel is not None:
		ax.set_yticklabels(yticklabel)
	if ylim is not None:
		ax.set_ylim(ylim)
	for i in range(num_methods):
		x_index = center_index + (i - (num_methods - 1) / 2) * width
		hbar=plt.bar(x_index, data_mean[:, i], width=width, linewidth=2,
		        color=colors[i], label=methods[i], hatch=hatches[i], edgecolor='k')
		if data_std is not None:
		    plt.errorbar(x=x_index, y=data_mean[:, i],
		                 yerr=data_std[:, i], fmt='k.', elinewidth=3,capsize=4)
		if use_barlabel_x:
			if i in [2,3]:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+3,f'{data_mean[k,i]:.4f}',fontsize = labelsize, rotation='vertical',fontweight='bold')
		if use_barlabe_y and i==2:
			for k,xdx in enumerate(x_index):
				ax.text(xdx-0.08,data_mean[k,i]+1,f'{data_mean[k,i]:.4f}',fontsize = labelsize, rotation='vertical',fontweight='bold')
		if use_downarrow:
			if i==1:
				for j in range(2,data_mean.shape[0]):
					ax.annotate(text='', xy=(x_index[j],data_mean[j,i]), xytext=(x_index[j],200), arrowprops=dict(arrowstyle='<->',lw=4))
					ax.text(x_index[j]-0.04, 160, '$\downarrow$'+f'{200-data_mean[j,i]:.0f}%', ha="center", va="center", rotation='vertical', size=labelsize ,fontweight='bold')
					# ax.text(center_index[j]-0.02,data_mean[j,i]+5,'$\downarrow$'+f'{200-data_mean[j,i]:.0f}%',fontsize = 16, fontweight='bold')
			else:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+5,f'{data_mean[k,i]:.2f}',fontsize = labelsize,fontweight='bold')

		if latency_annot:
			if i==1:
				for k,xdx in enumerate(x_index):
					mult = data_mean[k,i]/data_mean[k,0]
					ax.text(xdx-0.3,data_mean[k,i]+2,f'{mult:.1f}\u00D7',fontsize = labelsize)
		if bandwidth_annot:
			if i==1:
				for k,xdx in enumerate(x_index):
					mult = int(10**data_mean[k,i]/10**data_mean[k,0])
					ax.text(xdx-0.4,data_mean[k,i]+0.1,f'{mult}\u00D7',fontsize = labelsize)
		if latency_met_annot:
			if i>=1:
				for k,xdx in enumerate(x_index):
					mult = (-data_mean[k,i] + data_mean[k,0])/data_mean[k,0]*100
					ax.text(xdx-0.07,data_mean[k,i],'$\downarrow$'+f'{mult:.1f}%',fontsize = labelsize,rotation='vertical')
	if ncol>0:
		if legloc is None:
			plt.legend(bbox_to_anchor=bbox_to_anchor, fancybox=True,
			           loc='upper center', ncol=ncol, fontsize=lgsize)
		else:
			plt.legend(fancybox=True,
			           loc=legloc, ncol=ncol, fontsize=lgsize)
	fig.savefig(path, bbox_inches='tight')
	plt.close()

def plot_acc_n_failure_response():
	# the chance for a result to be full or degraded is 50/50
	acc_method_model = [[85.8,85.2,84.1,84.2],[83.5,83.2,82.6,82.8],[76.13,75.3,72.2,72.4],[72.192,71.1,68.0,68.1]]
	acc_method_model = np.array(acc_method_model)
	# Extract columns 2 and 3
	column2 = acc_method_model[:, 1]
	column3 = acc_method_model[:, 2]

	# Calculate the average of columns 2 and 3
	average_column = (column2 * 0.971 + column3 * (1-0.971))

	# Insert the average column at index 3
	acc_method_model = np.insert(acc_method_model, 3, average_column, axis=1)
	acc_method_model = acc_method_model[:,[0,4,3,1,2]]
	print(acc_method_model)

	envs = ['ConvNeXt','Swin','ResNet','MobileNet']
	methods = ['Original','Proactive','REACTIQ-Runtime','REACTIQ-Full','REACTIQ-Degraded']
	groupedbar(acc_method_model,None,'Top-1 Acc. (%)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/acc_method_model.eps',methods=methods,labelsize=20,xlabel='Models',
		envs=envs,ncol=1,width=1./6,sep=1,legloc='upper right',lgsize=14,ylim=(65,90))

	x = [];y = []
	default = 0.1
	for i in range(1):
		xx = np.linspace(0,11)*0.01
		yy = acc_method_model[i][0] * (1 - xx) + default * xx
		x += [xx]; y += [yy]
		yy = acc_method_model[i][0] * (1 - xx**2) + default * xx**2
		x += [xx]; y += [yy]
		yy = acc_method_model[i][3] * (1-(2+i) * xx**(1+i) * (1-xx)-xx**(2+i)) + acc_method_model[i][2] * (2+i) * xx**(1+i) * (1-xx) + default * xx**(2+i)
		x += [xx]; y += [yy]
		yy = acc_method_model[i][4] * (1 - xx**2) + default * xx**2
		x += [xx]; y += [yy]
	methods = ['Original-N1','Original-N2\n(200% Comp.)','Proactive-N2','REACTIQ-N2']
	linestyles_ = ['solid']*10#,'dashed','dotted','dashdot']
	y = np.array(y)
	y = y[[0,1,3,2],:]
	line_plot(x,y,methods,colors,
			'/home/bo/Dropbox/Research/NSDI24fFaultless/images/failure_response.eps',
			'Failure Rate','Top-1 Acc. (%)',lbsize=24,linewidth=2,markersize=8,linestyles=linestyles_,
			lgsize=18,legloc='lower left')

	acc_repl_model = [[85.8,85.2,84.1,84.2],[85.8,84.8,83.0,83.2],[85.8,84.3,82.0,82.4]]
	acc_repl_model = np.array(acc_repl_model)
	# Extract columns 2 and 3
	column2 = acc_repl_model[:, 1]
	column3 = acc_repl_model[:, 2]

	# Calculate the average of columns 2 and 3
	average_column = (column2 * 0.971 + column3 * (1-0.971))

	# Insert the average column at index 3
	acc_repl_model = np.insert(acc_repl_model, 3, average_column, axis=1)
	acc_repl_model = acc_repl_model[:,[0,4,3,1,2]]
	envs = ['2','3','4']
	methods = ['Original','Proactive','REACTIQ-Runtime','REACTIQ-Full','REACTIQ-Degraded']
	groupedbar(acc_repl_model,None,'Top-1 Acc. (%)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/acc_repl_model.eps',methods=methods,labelsize=20,
		envs=envs,ncol=2,width=1./6,sep=1,legloc='upper left',lgsize=13.5,ylim=(81,88),xlabel='# of Replicas')
	print(acc_repl_model)

	x = [];y = []
	default = 0.0
	for i in range(3):
		xx = np.linspace(0,10)*0.01
		yy = acc_repl_model[i][3] * (1-(2+i) * xx**(1+i) * (1-xx)-xx**(2+i)) + acc_repl_model[i][4] * (2+i) * xx**(1+i) * (1-xx) + default * xx**(2+i)
		x += [xx]; y += [yy]
		yy = acc_repl_model[i][1] * (1 - xx**(2+i)) + default * xx**(2+i)
		x += [xx]; y += [yy]
	methods = ['REACTIQ-N2','REACTIQ-N3','REACTIQ-N4','Proactive-N2','Proactive-N3','Proactive-N4']
	linestyles_ = ['solid']*6#,'dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
	y = np.array(y)
	y = y[[0,2,4,1,3,5]]
	line_plot(x,y,methods,colors,
			'/home/bo/Dropbox/Research/NSDI24fFaultless/images/failure_repl_response.eps',
			'Failure Rate','Top-1 Acc. (%)',lbsize=24,linewidth=2,markersize=8,linestyles=linestyles_,
			lgsize=18,legloc='lower left',ncol=2,ylim=(80,86))

def list_to_tail(latency_list,tail_options=[0.5,0.99,0.999]):
	L = len(latency_list)
	cdf_x = np.sort(np.array(latency_list))
	return [cdf_x[int(p*L)] for p in tail_options]

def plot_reactive_varywait(keyword='reactive'):
	# latency/computation/accuracy of reactive approaches when the threshold changes
	latency_list = []
	for idx,filename in enumerate(['original']):
		with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/imagenet/{filename}.log','r') as f:
			for l in f.readlines():
				latency_list += [float(l)]
	L = len(latency_list)
	latency_min,latency_max = min(latency_list),max(latency_list)

	# DCN
	dcn_latency_list = []
	for idx,filename in enumerate(['imagenet']):
		with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/{filename}/dcn.log','r') as f:
			for l in f.readlines():
				dcn_latency_list += [float(l)]
	dcn_latency_min,dcn_latency_max = min(dcn_latency_list),max(dcn_latency_list)
	cdf_x = np.sort(np.array(dcn_latency_list))
	dcn_L = len(cdf_x)
	dcn_latency_min = min(cdf_x)
	dcn_latency_90th = cdf_x[int(dcn_L*0.9)]
	dcn_latency_99th = cdf_x[int(dcn_L*0.99)]
	dcn_latency_999th = cdf_x[int(dcn_L*0.999)]
	dcn_latency_max = max(cdf_x)
	# print([dcn_latency_min,dcn_latency_90th,dcn_latency_99th,dcn_latency_999th,dcn_latency_max])

	# infer
	inf_mean = 0.019284586669921874; inf_std = 0.001057063805851016
	# inf_mean = 0.0048398783412604285; inf_std = 0.00021182092754556427
	inf_latency = np.random.normal(inf_mean, inf_std, dcn_L).tolist()
	inf_max = max(inf_latency)

	# resnet-50
	base_comp = [0.4883683402282493,0.4931303416137934, 0.28606033891261035]
	base_acc = [75.3,72.2,72.4]
	base_latency = []
	for idx,filename in enumerate(['rep2','react2']):
		with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/imagenet/{filename}.log','r') as f:
			latency_tmp = []
			for l in f.readlines():
				latency_tmp += [float(l)]
			base_latency += [np.mean(latency_tmp)]

	labels = ['Reactive','REACTIQ-R'] if keyword == 'reactive' else ['REACTIQ']
	M = 10 #if keyword == 'reactive' else 100
	all_acc_list = [[base_acc[2] for _ in range(M)],[]] if keyword == 'reactive' else [[]]
	all_latency_list = [[],[]] if keyword == 'reactive' else [[]]
	all_comp_list = [[],[]] if keyword == 'reactive' else [[]]
	# threshold to launch second
	if keyword == 'reactive':
		thresh_list = [i/(M-1)*(latency_max) for i in range(M)]
	else:
		thresh_list = [i/(M-1)*(dcn_latency_max) for i in range(M)]
	shuffled_list = latency_list.copy()
	random.shuffle(shuffled_list)
	shuffled_inf = inf_latency.copy()
	random.shuffle(shuffled_inf)

	if keyword != 'reactive':
		usage1 = [x+dcn_latency_99th>=y+z for x, y, z in zip(inf_latency, shuffled_inf, dcn_latency_list)]
		usage2 = [y+dcn_latency_99th>=x+z for x, y, z in zip(inf_latency, shuffled_inf, dcn_latency_list)]
		full_rate = sum(usage1 + usage2)/len(usage1+usage2)
		comp = full_rate * base_comp[0] * 2 + (1 - full_rate) * 2 * (base_comp[0]) * (1 - base_comp[2]/2)
		acc = full_rate * (base_acc[0]) + (1 - full_rate) * base_acc[1]
		print(full_rate, dcn_latency_99th/dcn_latency_max)
		print(comp,base_comp[0]*2-comp)
		print(acc,base_acc[0]-acc)
	for t in thresh_list:
		if keyword == 'reactive':

			min_values = [min(x, y + t) for x, y in zip(latency_list, shuffled_list)]
			all_latency_list[0] += [np.mean(min_values)]
			all_latency_list[1] += [np.mean(min_values)]

			# the first server must be used
			# the second reactive server is used when the first server latency > threshold
			usage = [x>=t for x, y in zip(latency_list, shuffled_list)]
			full_rate = sum(usage)/len(usage)
			print(full_rate)
			all_comp_list[0] += [full_rate*base_comp[1]+base_comp[1]]

			# run at low power mode when not reactive
			all_comp_list[1] += [full_rate * base_comp[0] * 2 + (1 - full_rate) * (base_comp[0]) * (1 - base_comp[2]/2)]

			all_acc_list[1] += [full_rate * (base_acc[0] + base_acc[1])/2 + (1 - full_rate) * base_acc[1]]
		else:
			min_values = [t/min(x,y)*100 for x, y in zip(latency_list, shuffled_list)]
			all_latency_list[0] += [np.mean(min_values)]

			usage1 = [x+t>=y+z for x, y, z in zip(inf_latency, shuffled_inf, dcn_latency_list)]
			usage2 = [y+t>=x+z for x, y, z in zip(inf_latency, shuffled_inf, dcn_latency_list)]
			full_rate = sum(usage1 + usage2)/len(usage1+usage2)

			all_comp_list[0] += [full_rate * base_comp[0] * 2 + (1 - full_rate) * 2 * (base_comp[0]) * (1 - base_comp[2]/2)]

			all_acc_list[0] += [full_rate * (base_acc[0]) + (1 - full_rate) * base_acc[1]]


	all_latency_list = np.array(all_latency_list)
	if keyword == 'reactive':
		for i in range(2):
			all_latency_list[i,:] += base_latency[i] - all_latency_list[i,0]
		all_latency_list = (all_latency_list) / (latency_max) * 100

	all_comp_list = np.array(all_comp_list) * 100

	x = [[i/(M-1) for i in range(M)] for _ in labels]
	xlabel = 'Reaction Threshold (Norm.)' if keyword == 'reactive' else 'Sync Threshold (Norm.)'
	ncol = None if keyword == 'reactive' else 0
	markersize = 8 if keyword == 'reactive' else 0
	xticks = None if keyword == 'reactive' else np.array([0,dcn_latency_99th,dcn_latency_max])/dcn_latency_max
	xticklabel = None if keyword == 'reactive' else ['0',"99th (Default)",'1']
	linestyles_ = ['solid']*10#,'dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
	lgsize = 26 if keyword == 'reactive' else 32
	line_plot(x,all_comp_list,labels,colors,
			f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/comp_{keyword}.eps',
			xlabel,'Computation(%)',lbsize=32,linewidth=2,markersize=markersize,linestyles=linestyles_,
			lgsize=lgsize,legloc='best',markevery=1,ncol=ncol,xticks=xticks,xticklabel=xticklabel,annot_aw=0 if keyword !='reactive' else None)
	line_plot(x,all_acc_list,labels,colors,
			f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/acc_{keyword}.eps',
			xlabel,'Top-1 Acc. (%)',lbsize=32,linewidth=2,markersize=markersize,linestyles=linestyles_,
			lgsize=lgsize,legloc='best',markevery=1,ncol=ncol,xticks=xticks,xticklabel=xticklabel,annot_aw=1 if keyword !='reactive' else None)
	line_plot(x,all_latency_list,labels,colors,
			f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_{keyword}.eps',
			xlabel,'Latency Inf. (%)',lbsize=32,linewidth=2,markersize=markersize,linestyles=linestyles_,
			lgsize=lgsize,legloc='best',markevery=1,ncol=ncol,xticks=xticks,xticklabel=xticklabel,annot_aw=2 if keyword !='reactive' else None)

def plot_breakdown():
	# batch breakdown
	latency_breakdown_mean = [[0.00455909734249115, 0.0005560131160076708, 0.00023642764806747438, 0.04561761306425502],
	[0.004631135659217835, 0.0008592742218635976, 0.00022894992828369142, 0.06668495337616376],
	[0.004496246471405029, 0.0013584858769550919, 0.0002239444637298584, 0.11089228095896249],
	[0.004519511394500733, 0.0025342730802483857, 0.00022607625961303712, 0.19675006694938904],
	[0.00478874797821045, 0.005448275371268392, 0.0002342068099975586, 0.3728581001051526],
	[0.0048398783412604285, 0.014246439476907482, 0.00023509953349543075, 0.7486890470647757],
	[0.004399494304778469, 0.015999999999999365, 0.0002341903698672155, 1.4240682725217528]]

	latency_breakdown_std = [[0.00019468925378149094, 5.584275363686788e-05, 1.9531520795695604e-05, 0.017782071823511884],
	[0.00017537580325603641, 7.260013062751902e-05, 1.736606173976656e-05, 0.025110829662576592],
	[0.00019017245067606885, 9.149818119041286e-05, 1.685588933829107e-05, 0.043189631076095754],
	[0.00021328910781494919, 0.00013665798550337163, 1.4873707000809235e-05, 0.07779634547285684],
	[0.00019141467146913094, 0.0004524667563076057, 1.677122266685227e-05, 0.1569391868139879],
	[0.00021182092754556427, 0.0011024832600207727, 4.049920186137311e-05, 0.34634288409670605],
	[0.00015224290905889435, 6.349087922075114e-16, 1.9471022131885656e-05, 0.5982422255192984]]
	latency_types = ['MAP','SYN','RED','F-M']	
	labels = ['1','2','4','8','16','32','64']
	plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_breakdown_vs_batch.eps',2,(0.77,0.64),title_posy=0.29,lim1=25,ratio=.48,lim2=2500)
	# cifar breakdown
	latency_types = ['MAP','SYN','RED','INF','F-M']	
	labels = ['Original', 'Proactive','REACTIQ',]
	latency_breakdown_mean = [[0,0,0,0.00483251379701657, 1.0169146132483973],
	[0,0,0,0.00483251379701657, 0.7486877294765364],
	[0.0048398783412604285, 0.014246439476907482, 0.00023509953349543075,0, 0.7486890470647757],]

	latency_breakdown_std = [[0,0,0,0.0002204459821230588, 0.5684324923094014],
	[0,0,0,0.0002204459821230588, 0.34634275598435527],
	[0.00021182092754556427, 0.0011024832600207727, 4.049920186137311e-05,0, 0.34634288409670605],]

	plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_breakdown_cifar.eps',5,(0.5,0.66),title_posy=0.35,ratio=0.25,lim2=2000)

	# imagenet latency breakdown
	latency_breakdown_mean = [
	[0,0,0,0.018242218704223632, 15.481805917434501],
	[0,0,0,0.018242218704223632, 11.0142774438325],
	[0.019284586669921874, 0.06311613967338686, 0.014764462451934815,0, 11.01427887952024],]

	latency_breakdown_std = [
	[0,0,0,0.007524489114244404, 8.696532160000752],
	[0,0,0,0.007524489114244404, 4.517442844121096],
	[0.001057063805851016, 0.0015932168873451931, 0.007375720249399674,0, 4.517442117207892],
	]
	plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_breakdown_imagenet.eps',5,(0.5,0.66),title_posy=0.35,lim1=100,lim2=25000,ratio=0.25)

def plot_flops_vs_acc():
	x = [
		[4.5,8.7,15.4,34.4,60.9],
		[4.5,8.7,15.4],
		[1.82,3.68,4.12,7.85,11.58],
		[0.01292,0.03721,0.05929,0.09714,0.20908,0.30079],
		]
	y = [
		[82.9,84.6,85.8,86.6,87.0],
		[81.2,83.2,83.5],
		[69.758,73.314,76.130,77.374,78.312],
		[34.896,52.352,60.092,64.592,69.952,72.192],
		]

	x = [np.log10(np.array(l)*1e9).tolist() for l in x]
	methods = ['ConvNeXt','Swin-Transformer','ResNet','MobileNet-v2']
	linestyles = ['solid']*10#,'dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
	line_plot(x,y,methods,colors,
			'/home/bo/Dropbox/Research/NSDI24fFaultless/images/flops_vs_acc.eps',
			'MFLOPS','Top-1 Acc. (%)',lbsize=24,linewidth=2,markersize=8,linestyles=linestyles,xticks=range(7,12),ylim=(30,90),
			lgsize=20,xticklabel=[f'$10^{i}$' for i in range(1,6)],markevery=1)


def plot_motivation():
	# motivation: DCN latency enable this
	# if we run two servers, at least one will have the other's information as the DC network is stable
	# but the early one might be sent to the front-end
	# need to make sure the fast server receives the slow server in 50% cases.
	# constant threshold
	# constant delay

	latency_list = []
	for idx,filename in enumerate(['original']):
		with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/cifar10/{filename}.log','r') as f:
			for l in f.readlines():
				latency_list += [float(l)]
	latency_min,latency_max = min(latency_list),max(latency_list)
	latency_list = (np.array(latency_list))
	L = len(latency_list)

	y = []
	# N=2
	p_list = []
	for p in [0.05,0.01,0.001]:
		p = 1-p**(1/2)
		p_list += [p**2]
	y += [p_list]
	# N=3
	p_list = []
	for p in [0.05,0.01,0.001]:
		p = 1-p**(1/3)
		p_list += [p**3 + 3 * p**2 * (1-p)]
	y += [p_list]
	# N=4
	p_list = []
	for p in [0.05,0.01,0.001]:
		p = 1-p**(1/4)
		p_list += [p**4 + 4 * p**3 * (1-p) + 6 * p**2 * (1-p)**2]
	y += [p_list]

	envs = ['95th','99th','99.9th']
	methods_tmp = ['N=2','N=3', 'N=4']
	y = np.array(y).T*100
	groupedbar(y,None,'Redundancy (%)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/prob_vs_repl.eps',methods=methods_tmp,labelsize=24,xlabel='Failure Threshold',
		envs=envs,ncol=1,width=.25,sep=1,legloc='lower right',lgsize=20,)

	N = 100
	x = [];y = [[],[],[],]
	p90,p95,p99 = 0,0,0
	for i in range(N):
		latency = i/(N-1)*(latency_max-latency_min) + latency_min
		x += [latency]
		p = sum(latency_list<latency)/L
		y[0] += [p*p]
		y[1] += [2*p*(1-p)]
		y[2] += [(1-p)**2]

	x = [x for _ in range(len(y))]

	linestyles_ = ['solid']*10#,'dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
	methods = ['Redundant', 'Exact', 'Failed',]
	# colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
	line_plot(x,y,methods,colors,
			'/home/bo/Dropbox/Research/NSDI24fFaultless/images/prob_vs_latency.eps',
			'Failure Threshold (s)','Likelihood (%)',lbsize=24,linewidth=2,markersize=8,linestyles=linestyles_,
			lgsize=20,legloc='best',use_prob_annot=True,
			xticks=[latency_min,1.324710109274886, 1.8153979640093003, 2.5920196835625706,latency_max],
			xticklabel=[f'{latency_min:.2f}\nmin','1.32\n95th','1.82\n99th','2.59\n99.9th',f'{latency_max:.2f}\nmax']
			)


def plot_metrics():
	# latency_list = [[],[]]
	# for idx,filename in enumerate(['cifar10','imagenet']):
	# 	with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/{filename}/dcn.log','r') as f:
	# 		for l in f.readlines():
	# 			latency_list[idx] += [float(l)]
	# labels = ['CIFAR-10', 'ImageNet']
	# measurements_to_cdf(latency_list,f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/dcn_latency_cdf.eps',labels,linestyles=linestyles,
	# 		colors=colors,bbox_to_anchor=(.7,0.4),lfsize=20,ncol=1,lbsize=24,xlabel=f'Latency (s)')
	latency = [[0.014551684027537704, 0.01568313699681312, 0.026199982967227697],
	[0.06311217998154461, 0.06720246304757893, 0.0727699960116297]
				]
	envs = ['CIFAR-10', 'ImageNet']
	methods_tmp = ['Medium','99th','99.9th']
	y = np.array(latency)
	groupedbar(y,None,'Latency (s)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_metrics_dcn.eps',methods=methods_tmp,labelsize=24,xlabel='Dataset',
		envs=envs,ncol=1,width=.25,sep=1,legloc=None,bbox_to_anchor=(0.25,1.02),lgsize=20,)

	# latency_list = [[],[],[]]
	# for idx,filename in enumerate(['original','react2','rep2']):
	# 	with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/imagenet/{filename}.log','r') as f:
	# 		for l in f.readlines():
	# 			latency_list[idx] += [float(l)]
	# labels = ['Original', 'REACT-N2','Approx-N2']
	# measurements_to_cdf(latency_list,f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_cdf2.eps',labels,linestyles=linestyles,
	# 		colors=colors,bbox_to_anchor=(.7,0.4),lfsize=20,ncol=1,lbsize=24,xlabel=f'Latency (s)')
	latency = [[13.118708950557236, 47.00770999836955, 54.96142094344102],
	[9.801519507608141, 26.822823907091642, 36.71819809887111],
	[9.723657777764773, 26.744062898290704, 36.63905317049958]
				]
	envs = ['Medium','99th','99.9th']
	methods_tmp = ['Original','Proactive', 'REACTIQ']
	y = np.array(latency).T
	y = y[:,[0,2,1]]
	groupedbar(y,None,'Latency (s)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_metrics_imagenet.eps',methods=methods_tmp,labelsize=24,xlabel='Metrics',
		envs=envs,ncol=1,width=.25,sep=1,legloc=None,bbox_to_anchor=(0.7,0.4),lgsize=20,latency_met_annot=True)

	# latency_list = [[],[],[],[]]
	# for idx,filename in enumerate(['original','react2','react3','react4']):
	# 	with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/cifar10/{filename}.log','r') as f:
	# 		for l in f.readlines():
	# 			latency_list[idx] += [float(l)]
	# labels = ['Original', 'REACT-N2', 'REACT-N3','REACT-N4']
	# measurements_to_cdf(latency_list,f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_repl_cdf.eps',labels,linestyles=linestyles,
	# 		colors=colors,bbox_to_anchor=(.7,0.5),lfsize=20,ncol=1,lbsize=24,xlabel=f'Latency (s)')

	latency = [[0.8651444067997442, 3.1388551201301063, 3.467007832221128],
	[0.669176916184692, 1.9810741319450342, 2.752692678973059],
	[0.5923948994946825, 1.3548866270896853, 1.9175132929601952],
	[0.5504748315755827, 1.1861622719566873, 1.5737354110882709]
				]
	envs = ['Medium','99th','99.9th']
	methods_tmp = ['Original', 'REACTIQ-N2', 'REACTIQ-N3','REACTIQ-N4']
	y = np.array(latency).T
	groupedbar(y,None,'Latency (s)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_repl_metrics.eps',methods=methods_tmp,labelsize=24,xlabel='Metrics',
		envs=envs,ncol=1,width=.2,sep=1,legloc=None,bbox_to_anchor=(0.2,1.02),lgsize=14,latency_met_annot=True,ylim=(0,4.5))

	# latency_list = [[],[],[]]
	# for idx,filename in enumerate(['original','react2','rep2']):
	# 	with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/logs/cifar10/{filename}.log','r') as f:
	# 		for l in f.readlines():
	# 			latency_list[idx] += [float(l)]
	# labels = ['Original', 'REACT-N2','Approx-N2']
	# measurements_to_cdf(latency_list,f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_cdf.eps',labels,linestyles=linestyles,
	# 		colors=colors,bbox_to_anchor=(.7,0.4),lfsize=20,ncol=1,lbsize=24,xlabel=f'Latency (s)')
	latency = [[0.8651444067997442, 3.1388551201301063, 3.467007832221128],
	[0.669176916184692, 1.9810741319450342, 2.752692678973059],
	[0.6544319294650239, 1.9654508961028212, 2.737339210027809]
				]
	envs = ['Medium','99th','99.9th']
	methods_tmp = ['Original','Proactive', 'REACTIQ']
	y = np.array(latency).T
	y = y[:,[0,2,1]]
	groupedbar(y,None,'Latency (s)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_metrics_cifar.eps',methods=methods_tmp,labelsize=24,xlabel='Metrics',
		envs=envs,ncol=1,width=.25,sep=1,legloc=None,bbox_to_anchor=(0.7,0.4),lgsize=20,latency_met_annot=True,ylim=(0,4.5))

def plot_comp_vs_model():
	latency = [[1,0.4877350051600989,0.4842981426839224, 0.0981940241180258],
				[1,0.4772859694677207,0.4816118024089462, 0.00898199113806668],
				[1,0.4883683402282493,0.4931303416137934, 0.28606033891261035],
				[1,0.49477564147949954,0.49501848582754915, 0.08724856076480889]]
	y = np.array(latency)
	y[:,1] *= 2
	y[:,2] *= 2
	y[:,3] = (1-y[:,3]) * y[:,2]
	y *= 100

	# Extract columns 2 and 3
	column2 = y[:, 2]
	column3 = y[:, 3]

	# Calculate the average of columns 2 and 3
	average_column = (column2 * 0.971 + column3 * (1-0.971))

	# Insert the average column at index 3
	y = np.insert(y, 4, average_column, axis=1)

	y[:,[2,3,4]] = y[:,[4,2,3,]]
	print(y)

	envs = ['ConvNeXt','Swin','ResNet','MobileNet']
	methods_tmp = ['Original','Proactive','REACTIQ-Runtime','REACTIQ-Full','REACTIQ-Degraded']
	groupedbar(y,None,'Relative Computation (%)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/comp_vs_model.eps',methods=methods_tmp,labelsize=20,xlabel='Models',
		envs=envs,ncol=1,width=.16,sep=1,legloc=None,bbox_to_anchor=(0.35,0.6),lgsize=18)


def plot_cost():
	# convnext
	# 25 0.4877350051600989
	# 28 0.4842981426839224 0.0981940241180258

	# swin
	# 10 0.4772859694677207
	# 10 0.4816118024089462 0.00898199113806668

	# mobilenetv2
	# 20 0.4883683402282493
	# 27 0.4931303416137934 0.28606033891261035

	# resnet-50, 64
	# 19 0.49477564147949954
	# 21 0.49501848582754915 0.08724856076480889

	# Before save (MB): 80.19921875 After save (MB): 1.45947265625 Before to after ratio: 54.950819672131146 Before to input ratio: 139.66666666666666 After to input ratio: 2.5416666666666665
	# Before save (MB): 2.03125 After save (MB): 0.0625 Before to after ratio: 32.5 Before to input ratio: 173.33333333333334 After to input ratio: 5.333333333333333 27

	# required communication / input communication
	# print('Required to input ratio:',2.03125/(32*32*3*4/1024/1024),40.099609375/(224*224*3*4/1024/1024))
	ratio_method_model = [[174.33,6.33],[140.67,3.54]]#96%,97
	envs = ['ResNet-56','ResNet-50']
	methods = ['w/o Selective Neural Sharing','w/ Selective Neural Sharing']
	y = np.array(ratio_method_model)
	print(y)
	# y = np.log10(y)
	groupedbar(y,None,'Relative BW Usage', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/bw_cost.eps',methods=methods,labelsize=24,latency_met_annot=True,
		envs=envs,ncol=1,width=.3,sep=1,legloc=None,lgsize=20,bbox_to_anchor=(0.5,0.7),
		# yticks=[np.log10(3),1,2,np.log10(200)],yticklabel=[3,10,100,200]
		)


def plot_challenge():
	# comp vs. acc, network improve with comp
	# plot computation allocation

	model_speed = [[0.001550,0.001603,0.001611,0.001658,0.001571],
	[0.002421,0.002373,0.002384,0.002364,0.002371],
	[0.003368,0.003466,0.003473,0.003332,0.003474],
	[0.004526,0.004356,0.004388,0.004400,0.004459],
	[0.008456,0.008505,0.008504,0.008475,0.008479]]
	model_speed = np.array(model_speed).mean(axis=1)

	rounds = [19,31,43,55,109]
	latency_list = []
	with open(f'../DCN/{22:06d}','r') as f:
		for l in f.readlines():
			l = l.strip().split(' ')
			latency_list += [float(l[0])/1000]
	latency = np.array(rounds) * np.array(latency_list).mean()


	envs = ['20','32','44','56','110']
	methods_tmp = ['Comp.','Sync.']
	y = np.stack((model_speed,latency)).T*1000

	groupedbar(y,None,'Latency (ms)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/latency_vs_model.eps',methods=methods_tmp,labelsize=24,
		envs=envs,ncol=1,width=.3,sep=1,legloc=None,bbox_to_anchor=(0.25,1),latency_annot=True,ylim=(0,70),xlabel='ResNet Config')

	relative_bw = [61,99,136,173,341]
	base_bw = 32*32*3*4 / 1024#12kB
	y = [[12,12,12,12,12],[61*12,99*12,136*12,173*12,341*12]]
	methods_tmp = ['Query','Neuron']
	y = np.array(y).T;y = np.log10(y)
	groupedbar(y,None,'Bandwidth Usage (KB)', 
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/bandwidth_vs_model.eps',methods=methods_tmp,labelsize=24,
		envs=envs,ncol=1,width=.3,sep=1,legloc=None,bbox_to_anchor=(0.7,.7),bandwidth_annot=True,
		yticks=[1,2,3,4],yticklabel=[10,'$10^2$','$10^3$','$10^4$'],ylim=(0,4),xlabel='ResNet Config')

	print(model_speed,latency,latency/model_speed,np.array(latency_list).mean())


def plot_OI_failure_response():
	acc_repl_model = [[85.8,85.2,84.1,84.2],[85.8,84.8,83.0,83.2],[85.8,84.3,82.0,82.4]]

	x = [];y = []
	default = 0.0
	for i in range(3):
		xx = np.linspace(0,100)*0.01
		if i==0:
			p1 = 4 * xx**3 * (1-xx) + 2 * xx**2 * (1-xx)**2
		elif i==1:
			p1 = 4 * xx**3 * (1-xx) + 1 * xx**2 * (1-xx)**2
		else:
			p1 = 4 * xx**3 * (1-xx)
		yy = acc_repl_model[i][1] * (1-p1-xx**4) + acc_repl_model[i][2] * p1 + default * xx**4
		x += [xx]; y += [yy]
	methods = ['REACTIQ-N2','REACTIQ-N3','REACTIQ-N4']
	linestyles_ = ['solid']*10#,'dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
	line_plot(x,y,methods,colors,
			'/home/bo/Dropbox/Research/NSDI24fFaultless/images/failure_N_response.eps',
			'Failure Rate','Top-1 Acc. (%)',lbsize=24,linewidth=2,markersize=8,linestyles=linestyles_,
			lgsize=16,legloc='lower left')

def plot_learning_curve():
	for name in ['resnet50']:
		numsn = 4
		colors_tmp = colors
		all_acc = [[] for _ in range(numsn)]
		all_epoch = [[] for _ in range(numsn)]
		lines = 1
		with open(f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/{name}.log','r') as f:
			for line in f.readlines():
				line = line.split(' ')
				epoch = eval(line[0])
				line = line[1].strip().split(',')[2:]
				accuracy_list = [eval(n.split('(')[0]) for n in line if len(n)>=6]
				for i,n in enumerate(accuracy_list):
					all_acc[i] += [n]
					all_epoch[i] += [lines]
				lines += 1
		for line in all_acc:
			print(max(line))
		all_acc[0] = (np.array(all_acc[0]) + np.array(all_acc[1]))/2
		del all_acc[1]
		del all_epoch[1]
		max_list = [75.3,72.2]
		max1 = max(all_acc[0])
		all_acc[0] = (np.array(all_acc[0])/max1*max_list[0]).tolist()
		max1 = max(max(all_acc[1]),max(all_acc[2]))
		all_acc[1] = (np.array(all_acc[1])/max1*max_list[1]).tolist()
		all_acc[2] = (np.array(all_acc[2])/max1*max_list[1]).tolist()
		methods_tmp = [f'SN#1Rank#1', 'SN#2Rank#2', 'SN#3Rank#2']
		xticks = [0,40,80,120,160]
		ncol = 1
		line_plot(all_epoch,all_acc,methods_tmp,colors_tmp,
				f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/convnext_learning_curve.eps',
				'Epoch','Test Top-1 Acc. (%)',markersize=0,linewidth=4,xticks=xticks,legloc='lower right',lbsize=24,ncol=ncol,use_resnet50arrow=True,lgsize=18)

def simulate():
    print('Analyzing all recorded traces...')
    selected_batch_latency = []
    y = []
    yerr = []
    trpt = []
    trpterr = []
    trace_filenames = []
    trace_filenames += [f'../DCN/{22*i:06d}' for i in [1,2,4,8,16,32,64]]
    trace_filenames += [f'../DCN-244/{244*i:06d}' for i in [1,2,4,8,16,32,64]]
    latency_mean_list = []
    latency_std_list = []
    trpt_mean_list = []
    trpt_std_list = []
    all_latency_list = []
    for tidx,filename in enumerate(trace_filenames):
        latency_list = []
        with open(filename,'r') as f:
            for l in f.readlines():
                l = l.strip().split(' ')
                latency_list += [float(l[0])/1e3]
            if len(latency_list)>=10000:
            	latency_list = latency_list[:10000]
            else:
            	latency_list = latency_list[:1000]
            	latency_list = latency_list*10
        all_latency_list += [latency_list]
    import csv
    with open('../curr_videostream.csv', mode='r') as csv_file:
    # with open('../curr_httpgetmt.csv', mode='r') as csv_file:
        # read network traces 
        csv_reader = csv.DictReader(csv_file)
        latency_list = []
        latency224_list = []
        num_of_line = 0
        bandwidth_list = []
        for row in csv_reader:
            if row["bytes_sec_interval"] == 'NULL':
                continue
            # bandwidth_list += [float(row["downthrpt"])]
            bandwidth_list += [float(row["bytes_sec_interval"])]
            for bs in [2**i for i in range(7)]:
                query_size = 3*32*32*4*bs # bytes
                latency_list += [query_size/float(row["bytes_sec_interval"]) ]
                query_size = 3*224*224*4*bs
                latency224_list += [query_size/float(row["bytes_sec_interval"])]
            num_of_line += 1
            if num_of_line==10000:break
        all_latency_list += np.array(latency_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
        all_latency_list += np.array(latency224_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
    all_latency_list = np.array(all_latency_list)
    all_latency_list = all_latency_list.mean(axis=-1).reshape(4,7)
    query_size = 3*32*32*4*np.array([2**(i) for i in range(7)])
    bw = query_size/all_latency_list[0]/1e6*8
    print(bw.mean(),bw.std(),'MBps',np.array(bandwidth_list).mean()*8,np.array(bandwidth_list).std()*8)

num_samples = 50000
comp_time = [0.004653,0.005051,0.005240,0.005663,0.005940,0.006688,0.006945]
comp_time0 = [0.010791,0.016995,0.017772,0.019092,0.021050,0.025342,0.033885]
analyze_all_recorded_traces()
exit(0)
plot_reactive_varywait(keyword='adaptive_wait')
plot_acc_n_failure_response()

exit(0)
plot_OI_failure_response()
plot_reactive_varywait()
plot_breakdown()
plot_cost()
exit(0)
plot_metrics()
plot_comp_vs_model()
plot_challenge()
plot_learning_curve()
plot_motivation()
plot_flops_vs_acc()


x = [[1-1.0*i/32 for i in range(32)] for _ in range(3)]
y = [[15438473216.0, 14498716864.0, 13588464000.0, 12707714624.0, 11856468736.0, 11034726336.0, 10242487424.0, 9479752000.0, 8746520064.0, 8042791616.0, 7368566656.0, 6723845184.0, 6108627200.0, 5522912704.0, 4966701696.0, 4439994176.0, 3942790144.0, 3475089600.0, 3036892544.0, 2628198976.0, 2249008896.0, 1899322304.0, 1579139200.0, 1288459584.0, 1027283456.0, 795610816.0, 593441664.0, 420776000.0, 277613824.0, 163955136.0, 79799936.0, 25148224.0],
[4094803968.0, 3843109680.0, 3599398080.0, 3363669168.0, 3135922944.0, 2916159408.0, 2704378560.0, 2500580400.0, 2304764928.0, 2116932144.0, 1937082048.0, 1765214640.0, 1601329920.0, 1445427888.0, 1297508544.0, 1157571888.0, 1025617920.0, 901646640.0, 785658048.0, 677652144.0, 577628928.0, 485588400.0, 401530560.0, 325455408.0, 257362944.0, 197253168.0, 145126080.0, 100981680.0, 64819968.0, 36640944.0, 16444608.0, 4230960.0],
[300774272.0, 282935512.0, 265641240.0, 248891456.0, 232686160.0, 217025352.0, 201909032.0, 187337200.0, 173309856.0, 159827000.0, 146888632.0, 134494752.0, 122645360.0, 111340456.0, 100580040.0, 90364112.0, 80692672.0, 71565720.0, 62983256.0, 54945280.0, 47451792.0, 40502792.0, 34098280.0, 28238256.0, 22922720.0, 18151672.0, 13925112.0, 10243040.0, 7105456.0, 4512360.0, 2463752.0, 959632.0]]
selected = 10
y = np.array(y)
print(y[:,selected]/y[:,0])
# y = y/(y[:,0].repeat(32).reshape(3,32))
methods = ['Swin-B','ResNet50','MobileNet-v2']
line_plot(x,y,methods,colors,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/flops_vs_ratio.eps',
		'Model Width','FLOPS',lbsize=24,linewidth=4,markersize=8,linestyles=linestyles,
		lgsize=20)	
exit(0)

# ======================================new/old========================================================

flops = [[200,100,132,129],
[200,100,132,115],
[200,100,112,109],
[200,100,112,106],
[200,100,112,100]]
x = np.array(flops).T
small=0
soft = [[small,0.18165135782747602,0.013478434504792358,0.00457667731629392],
[small,0.18320886581469645, 0.00808706070287546,0.0020966453674121643],
[small,0.1850938498402556, 0.01126198083067087,0.005990415335463184],
[small,0.18546325878594247,0.017771565495207642,0.00467252396166129],
[small,0.1863797923322683, 0.010163738019169366,0.006689297124600646]
]
y = np.array(soft).T*100
methods_tmp = ['MultiShot','OneShot','EfficientMulti','Ours']
# optimize for consistency
line_plot(x,y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/scatter_soft.eps',
		'FLOPS (%)','Consistency (%)',lbsize=24,linewidth=0,markersize=4,linestyles=linestyles,xticks=[100,150,200],
		# yticks=[-5,-4,-3,-2,-1], yticklabel=[1e-5,1e-4,1e-3,1e-2,1e-1],
		lgsize=20,bbox_to_anchor=(1.02,1.02),ncol=1,use_arrow=False,arrow_coord=(65,0.8),ratio=.78,oval=True,scatter_soft_annot=True)	
exit(0)
envs = ['R20','R32','R44','R56','R110']
methods_tmp = ['TR','NR','RR','Ours']
flops = [[200,100,132,129],
[200,100,132,115],
[200,100,112,109],
[200,100,112,106],
[200,100,112,100]]
y = np.array(flops)
groupedbar(y,None,'FLOPS (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/flops_res.eps',methods=methods_tmp,labelsize=24,
	envs=envs,ncol=1,width=.2,sep=1,legloc='lower right',lgsize=18)

envs = ['R20','R32','R44','R56','R110']
methods_tmp = ['TR','NR','RR','Ours']
soft = [[0.18165135782747602,0.013478434504792358,0.00457667731629392],
[0.18320886581469645, 0.00808706070287546,0.0020966453674121643],
[0.1850938498402556, 0.01126198083067087,0.005990415335463184],
[0.18546325878594247,0.017771565495207642,0.00467252396166129],
[0.1863797923322683, 0.010163738019169366,0.006689297124600646]
]

y = np.array(soft)*100
for i in [0,1]:
	print((y[:,2]-y[:,i])/y[:,i])
groupedbar(y,None,'Consistency (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/soft_re_res.eps',methods=methods_tmp,labelsize=24,yticks=range(0,20,5),
	envs=envs,ncol=1,width=.25,sep=1,legloc=None,use_barlabe_y=True,bbox_to_anchor=(0.2,0.9))

hard = [[0.20585662939297125,0.01487619808306706,0.0019730336223946314],
[0.2075758785942492,0.009384984025559095,0.004093450479233263],
[0.21724041533546323, 0.010882587859424864,0.009185303514377074],
[0.20588857827476037,0.018370607028753816,0.004293130990415284],
[0.21376597444089454, 0.011781150159744347,0.008286741214057591]]

y = np.array(hard)*100
for i in [0,1]:
	print((y[:,2]-y[:,i])/y[:,i])
groupedbar(y,None,'Consistency (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/hard_re_res.eps',methods=methods_tmp,labelsize=24,
	envs=envs,ncol=1,width=.25,sep=1,legloc=None,use_barlabe_y=True,bbox_to_anchor=(0.2,0.9))

exit(0)

# baseline
# flops_base = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
flops_base = [0.9355,0.8224,0.7171,0.6196,0.5298,0.4479,0.3737,0.3073]
re_base = [[0.006060303514377014, -0.001283660429027841],
[0.007938298722044735,2.3771489426446934e-05],
[0.0085513178913738,0.0013074319184542707],
[0.011028354632587872,0.002904876007911146],
[0.010889576677316302,0.004559371671991471],
[0.010039936102236427,0.006133044272021897],
[0.013776956869009583,0.010430929560322535],
[0.012270367412140598,0.01332154267457781]
]
re_base = np.array(re_base)

# no bridge
flops_nobridge = [0.25088513674100354,0.3172, 0.3914543375525,0.4734,0.5632,0.6606950325238663,0.7660,0.8791134250074207]
flops_nobridge = flops_nobridge[::-1]
re_nobridge = [[0.008202875399361022,-0.0011315228966986173],
[0.010236621405750799,0.0016782671535067594],
[0.014620607028754024,0.007278830062376389],
[0.014349041533546325,0.007112429636391295],
[0.015884584664536745,0.008343792788680969],
[0.018492412140575072,0.013397611440742443],
[0.018699081469648566,0.01871291647649475],
[0.01905650958466453,0.020410200821542695]
]
re_nobridge = np.array(re_nobridge)

# no onn
# flops_noonn = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
flops_noonn = [0.9355,0.8224,0.7171,0.6196,0.5298,0.4479,0.3737,0.3073]
re_noonn = [[0.011729233226837053, 0.13601665905979005],
[0.010072883386581464, 0.13977730868705307],
[0.01117511980830671, 0.14046858359957404],
[0.01695686900958466, 0.13745816217860943],
[0.016213059105431298, 0.13781378366042904],
[0.02036641373801915, 0.13612125361326638],
[0.02206869009584664, 0.13753232922561995],
[0.02264676517571885,0.13729366347177846]
]
re_noonn = np.array(re_noonn)

# no collaboration
flops_nolab = [0.8791134250074207,0.7660117691820428,0.6606950325238663,0.5631632150328911,0.4734163167091172,0.3914543375525446,0.3172772775631734,0.25088513674100354]
re_nolab = [[0.00775159744408944, 0.13789555758405597],
[0.011381789137380194, 0.14174273543283128],
[0.010660942492012767, 0.1395329377757493],
[0.011745207667731619, 0.13819412749125207],
[0.021690295527156546, 0.13601856077894414],
[0.02897763578274759, 0.14184352654799937],
[0.022319289137380186, 0.13540525635174197],
[0.02690095846645364, 0.1367792484405903]]
re_nolab = np.array(re_nolab)


x = np.array([flops_base,flops_nobridge,flops_noonn,flops_nolab])*200
# Model Transform; Collaborative Training
methods_tmp = ['Ours','w/o MT','w/o FTT','w/o FTT+MT']
y1 = np.concatenate((re_base[:,0].reshape((1,8)),
					re_nobridge[:,0].reshape((1,8)),
					re_noonn[:,0].reshape((1,8)),
					re_nolab[:,0].reshape((1,8))))*100
line_plot(x,y1,methods_tmp,colors,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/ablation_soft.eps',
		'FLOPS (%)','Consistency (%)',lbsize=24,linewidth=0,markersize=16,linestyles=linestyles,xticks=[50,100,150],yticks=[1,2,3],
		lgsize=20,bbox_to_anchor=(.4,.49),ncol=1,use_arrow=True,arrow_coord=(65,0.8),ratio=.78)	
exit(0)

# different convs
re_vs_conv = [[0.00467252396166129,0.004293130990415284],
[0.004492811501597527,0.007488019169329063],
[0.005451277955271783,0.004093450479233263],
  [0.004592651757188482,0.003993610223642197],
[0.0034944089456869776, 0.0036940894568689986],
[0.0030950479233227135,0.006489616613418514]
]
flops_vs_conv = [0.5298202593545005,0.558022230677192,0.5862242019998837,0.6003,0.642628144645267,0.6990320872906503]
bridge_size = [64,96,128,144,192,256]
re_vs_conv = np.array(re_vs_conv)
y = re_vs_conv*100
envs = [f'{bridge_size[i]}\n{round(flops_vs_conv[i]*200)}' for i in range(6)]
groupedbar(y,None,'Consistency (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/re_vs_conv.eps',methods=['Jitter','Fail-stop'],envs=envs,labelsize=24,yticks=[0,.2,.4,.6],
	ncol=1,sep=1,bbox_to_anchor=(0.63, 1.03),width=0.4,xlabel='Bridge Size and FLOPS (%)',legloc=None)

# different partition ratios
diff_ratio_data = [[-0.0023961661341853624,-0.0020966453674121643],
[-0.00029952076677319805,-0.00029952076677319805],
[0.0014776357827476216,0.0009984025559105492],
[0.0035543130990413063,0.003993610223642197],
 [0.004592651757188482,0.003993610223642197],
[0.00439297124600635,0.006689297124600646],
[0.008386581469648546,0.011481629392971149],
[0.00658945686900958,0.013578274760383424]
]
diff_ratio_data = np.array(diff_ratio_data)*100
y = diff_ratio_data
# flops_base = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
flops_base = [0.9355,0.8224,0.7171,0.6196,0.5298,0.4479,0.3737,0.3073]
envs = [f'{round(100-6.25*i)}\n{round(flops_base[i-1]*200)}' for i in range(1,9)]
groupedbar(y,None,'Consistency (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/re_vs_ratio.eps',methods=['Jitter','Fail-stop'],envs=envs,labelsize=24,yticks=[0,.5,1],
	ncol=1,sep=1,width=0.4,xlabel='Partition Ratio (%) and FLOPS (%)',legloc='upper left')
exit(0)

x = [[5*i for i in range(21)]for _ in range(4)]
y = [[0.8957627795527155, 0.8517651757188498, 0.8090595047923321, 0.7735822683706071, 0.7221285942492013, 0.691517571884984, 0.6490095846645367, 0.6029273162939297, 0.5602416134185303, 0.5173462460063898, 0.48724640575079875, 0.4363218849840256, 0.38952476038338657, 0.3393550319488818, 0.31298322683706076, 0.2747703674121406, 0.22207468051118212, 0.18240415335463261, 0.14094049520766774, 0.09999999999999999], [0.9389436900958467, 0.9327096645367412, 0.9219668530351438, 0.9096026357827476, 0.883426517571885, 0.8675299520766775, 0.8387220447284346, 0.7953873801916932, 0.7715415335463258, 0.7225039936102237, 0.6893610223642173, 0.6380251597444089, 0.5819189297124601, 0.5162260383386582, 0.47091653354632584, 0.41250599041533553, 0.3273941693290735, 0.265491214057508, 0.18740814696485625, 0.09999999999999999], [0.9408266773162939, 0.9405591054313097, 0.9381709265175718, 0.9347224440894569, 0.9279133386581468, 0.9181908945686901, 0.9031988817891374, 0.8845347444089458, 0.8644009584664536, 0.8393170926517571, 0.801920926517572, 0.749129392971246, 0.7044189297124601, 0.6362100638977635, 0.5899420926517571, 0.5227376198083067, 0.41543730031948883, 0.3366453674121407, 0.22356230031948882, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.940023961661342, 0.9392511980830671, 0.9363278753993611, 0.9344948083067092, 0.9302615814696485, 0.9201138178913739, 0.9052875399361024, 0.8918390575079872, 0.8627715654952077, 0.8249940095846645, 0.7863258785942492, 0.7312519968051119, 0.6771405750798722, 0.6024860223642172, 0.4888238817891374, 0.39297723642172533, 0.2562579872204473, 0.09999999999999999], [0.898817891373802, 0.8929632587859425, 0.8826697284345049, 0.8703753993610224, 0.8460463258785943, 0.8304492811501596, 0.8028394568690096, 0.7618710063897763, 0.7391533546325878, 0.6926218051118211, 0.6609764376996805, 0.6121765175718851, 0.5579772364217253, 0.49667731629392964, 0.45305511182108626, 0.3973103035143771, 0.3167911341853036, 0.25841253993610225, 0.1836341853035144, 0.09999999999999999]]
yerr = [[0.006997876375020307, 0.016734437743554532, 0.012127500296505958, 0.017237607923604327, 0.01652817916156893, 0.017862496663828824, 0.019094398530911494, 0.01293758123754703, 0.02790298260476726, 0.029083235857071756, 0.019613131810753536, 0.024548452749052943, 0.012422231004442159, 0.015561299532737535, 0.020248306445012344, 0.017593245190660217, 0.013815886487961736, 0.010064554627632874, 0.00901415000465792, 0.0], [0.0020199744070912577, 0.0032324253766238334, 0.00653287651965923, 0.009172708278275014, 0.011921123855137186, 0.01059779721918944, 0.017090001119459443, 0.012361923551600719, 0.02400840721149313, 0.026234013096169042, 0.0228978001598712, 0.03175155848795646, 0.02244152268715682, 0.025468525848158535, 0.029358407348361502, 0.02099587933965674, 0.024345903249069753, 0.017092721271991466, 0.013484202410392266, 0.0], [0.0008027156549520575, 0.0010712184807357074, 0.0022035031668343617, 0.005239027964211368, 0.00393279828078102, 0.005460873192321837, 0.010205587905032077, 0.008860327405948483, 0.020381674123960886, 0.015519384629006138, 0.01613874370173752, 0.025357654777092082, 0.016296640957360668, 0.020385145055574323, 0.026988731096102322, 0.026800322050731698, 0.024095142805632887, 0.017292520880111212, 0.01813718868803247, 0.0], [0.0, 0.0, 0.0024308623529587818, 0.0024002403129800703, 0.0020382685888009405, 0.004076723499975016, 0.004875208983659424, 0.003370617083741069, 0.005564876243903203, 0.0051922542858615466, 0.00958072502904603, 0.019763452711440668, 0.016496599841994884, 0.019692192854194834, 0.02522283850573193, 0.022579075887578987, 0.024949860614209174, 0.012598416304351604, 0.020184203882597448, 0.0], [0.001927342112372416, 0.003179089331609038, 0.006199251512525477, 0.008842385139736059, 0.01141126165629694, 0.01041494363648053, 0.01664410498867549, 0.011930115136505527, 0.023236953811564296, 0.025148276409804056, 0.021797757967920994, 0.03150124050809064, 0.022916120965365556, 0.024505531034889692, 0.028665699102147366, 0.0206405564153535, 0.022501503135747496, 0.016330672689323523, 0.012477112118501896, 0.0]]
y = [[0.9412]+l for l in y]
yerr = [[0]+l for l in yerr]
y = np.array(y)[[0,1,2,3]]*100;yerr = np.array(yerr)[[0,1,2,3]]*100
line_plot(x, y,['NR','TR2','TR3','TR4'],colors,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/mot0.eps',
		'Failure Rate (%)','Accuracy (%)',lbsize=20,use_re_label=True,markersize=8,ylim=(0,100),lgsize=19)
exit(0)
n_soft = 11;n_hard = 11;
methods_tmp = ['TR','NR','RR','Ours']
x = [[-1+0.1*i for i in range(4,n_soft)] for _ in range(4)]
y = [[73.8868, 73.8868, 73.8868, 73.8868, 73.8868, 73.8868, 73.83120360000001, 73.63901616, 72.71547744, 69.89506776, 63.217518080000005, 51.496512640000006, 36.774109679999995, 15.64173784, 3.37097128, 0.59556048, 0.2275852, 0.09799368, 0.01379936, 0.004999999999999998, 0.004999999999999998], [73.914, 73.914, 73.914, 73.914, 73.914, 73.24084496, 71.47976215999999, 69.45809848, 64.24065096000001, 56.77935784, 45.968288959999995, 33.777114479999995, 21.788126, 8.800402719999997, 1.9752658399999994, 0.30298015999999994, 0.09679383999999999, 0.058396399999999994, 0.012599439999999998, 0.004999999999999998, 0.004999999999999998], [73.914, 73.914, 73.914, 73.914, 73.914, 73.914, 73.85780360000003, 73.6764156, 72.76927592000001, 69.95706471999999, 63.442905599999996, 51.927884799999994, 37.235478, 16.51327912,3.75994592, 0.6623562399999999, 0.23538464, 0.13759104, 0.013999359999999999, 0.004999999999999998, 0.004999999999999998]]
yerr = [[0.04577510240294409, 0.04577510240294409, 0.04577510240294409, 0.04577510240294409, 0.04577510240294409, 0.04577510240294409, 0.054022952185604135, 0.05507914308911032, 0.0982076078511558, 0.28094892802672433, 0.33462695096933237, 0.6670215390564869, 0.569713656451935, 0.552894980143063, 0.2689066643320123, 0.07489283070431776, 0.06541190241416313, 0.02720496136975019, 0.004995716575467425, 8.673617379884035e-19, 8.673617379884035e-19], [0.0, 0.0, 0.0, 0.0, 0.0, 0.2052613762357208, 0.3539368446015036, 0.5015841128377131, 0.6205342797773227, 0.5685524804821483, 0.6938345171697867, 0.7350952701102259,0.7114645218358187, 0.5409356396315939, 0.23212801166833433, 0.16493677278988575, 0.09207841426544225, 0.052919375780143135, 0.005782417499143414, 8.673617379884035e-19, 8.673617379884035e-19], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027442113673108757, 0.040463478885139036, 0.10328881249619444, 0.2593282664434568, 0.2918727896544517, 0.6450631531825839, 0.6253813228000726, 0.5871620401640372, 0.33913253231595686, 0.0797751540027746, 0.0598146241615744, 0.03560708073367431, 0.0054586738409983775, 8.673617379884035e-19, 8.673617379884035e-19]]
y += [[72.38200000000002, 72.38200000000002, 72.38200000000002, 72.38200000000002, 72.38200000000002, 72.38200000000002, 72.3280036, 72.1476156, 71.261476, 68.50386464, 62.11750584, 50.847085119999996, 36.44287823999999, 16.13508024, 3.6675463199999996, 0.6469562399999998, 0.23258463999999995, 0.13579104, 0.01359936, 0.004999999999999998, 0.004999999999999998], [75.99199999999999, 75.99199999999999, 75.99199999999999, 75.99199999999999, 75.99199999999999, 75.30064496, 73.48276216, 71.41389848, 66.03625111999999, 58.37335808, 47.26688904, 34.74751456, 22.400126479999994, 9.05360312, 2.0316659199999996, 0.30858015999999994, 0.09839383999999998, 0.058996399999999984, 0.01239944, 0.004999999999999998, 0.004999999999999998], [75.99199999999999, 75.99199999999999, 75.99199999999999, 75.99199999999999, 75.99199999999999, 75.99199999999999, 75.9352036, 75.74681559999999, 74.81147608, 71.92926456000001, 65.2417056, 53.39468536, 38.30847848, 16.979080399999997, 3.86714608, 0.67755632, 0.24298463999999997, 0.14139104, 0.01359936, 0.004999999999999998, 0.004999999999999998]]
yerr += [[1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 0.02607519593114877, 0.03502732652498781, 0.09982628449381624, 0.24373966325347507, 0.28047476178589337, 0.6028010942153519, 0.5998872263124454, 0.5630702004908794, 0.3237102311696335, 0.07843521879976112, 0.06193040907010385, 0.03606636025958816, 0.005444010295949115, 8.673617379884035e-19, 8.673617379884035e-19], [1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 0.2075476858897298, 0.36378115823455076, 0.5135045480258117, 0.6210881094204086, 0.5628827149482211, 0.6844891705101478, 0.7273162892080933, 0.7119902209356708, 0.5446230517387025, 0.24545963619605066, 0.1710340016088684, 0.09317349669105694, 0.05361731374248434, 0.00586825767450612, 8.673617379884035e-19, 8.673617379884035e-19], [1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 1.4210854715202004e-14, 0.026982797535614375, 0.03691111056969623, 0.10229619645962341, 0.2649615363214007, 0.31131769377986773, 0.6352789259392053, 0.6266543256617008, 0.5715603340139557, 0.32919465995225605, 0.07952896708008726, 0.06351941573716181, 0.038341265163977054, 0.005444010295949116, 8.673617379884035e-19, 8.673617379884035e-19]]

y = np.array(y);yerr = np.array(yerr)
y = y[[5,4,3,0],4:n_soft];yerr = yerr[[5,4,3,0],4:n_soft]
print((y-y[[0]]).min(axis=1))
line_plot(x, y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/FCC_ea2.eps',
		'Jitter Level','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,linestyles=linestyles,
		yerr=yerr)
exit(0)
y = [[73.88599999999998, 73.6472144, 73.1512468, 72.21711008, 70.78680504, 69.09611928000001, 67.23404520000001, 64.73021056000002, 61.78121008000001, 58.985797520000006, 55.09066232, 51.396512959999995, 47.35518519999999, 42.247730080000004, 37.57504615999999, 32.335805040000004, 26.669987199999998, 20.365017599999995, 13.885058799999996, 7.1679134399999995, 0.004999999999999998], [73.914, 70.16805271999999, 66.50910160000001, 62.96094528, 58.924813199999996, 55.24985888000001, 51.798893199999995, 48.21914088, 44.341197040000004, 40.8306392, 37.029701360000004, 33.273346, 29.782184559999997, 25.783454799999994, 22.230500879999994, 18.710336319999996, 14.876792959999998, 11.172642719999999, 7.426099599999999, 3.8417403199999995, 0.004999999999999998], [73.914, 73.69341440000001, 73.2284468, 72.30391008000001, 70.87120504, 69.22311928, 67.3216452, 64.86221056, 61.93721008, 59.18939752, 55.28346232, 51.58651295999999, 47.5039852, 42.41093007999999, 37.76184615999999, 32.494805039999996, 26.7559872, 20.447217600000002, 13.954858799999997, 7.165513439999998, 0.004999999999999998]]
yerr = [[1.4210854715202004e-14, 0.05222890856344034, 0.06703911463019199, 0.09229484257310165, 0.2812408288925467, 0.20208918390420183, 0.19422717289944724, 0.26320091266098283, 0.23269037474885168, 0.26985074714310114, 0.40338034036797776, 0.3152833978755835, 0.38247435488028036, 0.35055487317970874, 0.5660631155471807, 0.348039396154042, 0.39246181042513717, 0.295761471093988, 0.25723364490492295, 0.3327245715482616, 8.673617379884035e-19], [0.0, 0.21134046508133272, 0.2853341704102587, 0.18904340145334214, 0.3059133858356641, 0.2883092011065797, 0.18034183191111133, 0.6034689293118537, 0.43849994079496785, 0.48040392624566797, 0.45488327221813507, 0.4091682115563911, 0.35046239924690203, 0.5328890492208965, 0.3511448509600707, 0.29898295378238837, 0.308661266317791, 0.334240332965885, 0.22763112037202646, 0.18285092982888973, 8.673617379884035e-19], [0.0, 0.05458673254306435, 0.05032261370795339, 0.09359467120297756, 0.2586605240158727, 0.1569936009239401, 0.205560489132788, 0.2766346276690167, 0.2650772481009289, 0.22804261722250405, 0.35715786616478973, 0.2642820260554503, 0.36038937717569935, 0.3837076306226408, 0.5460807330486789, 0.29945242399297806, 0.34521131661951143, 0.3246834038624087, 0.25619796921730664, 0.29760637423276787, 8.673617379884035e-19]]
y += [[72.38200000000002, 72.1670144, 71.7044468, 70.80731008, 69.40720504000001, 67.78411928000001, 65.91624520000002, 63.545010559999994, 60.661810079999995, 57.94319752, 54.11306232, 50.482112959999995, 46.5311852, 41.53853008, 36.96924616, 31.78020504, 26.223187199999995, 20.051617599999997, 13.655458799999996, 7.021113439999999, 0.004999999999999998], [75.99199999999999, 72.14025272, 68.36950159999999, 64.70894528000001, 60.5728132, 56.815458879999994, 53.259493199999994, 49.59654088, 45.59919703999999, 41.96803919999999, 38.02530136, 34.214946, 30.633984559999995, 26.5144548, 22.834500879999997, 19.23653632, 15.289592959999998, 11.493642719999999, 7.625099599999997, 3.971140319999999, 0.004999999999999998], [75.99199999999999, 75.7678144, 75.2876468, 74.32731007999999, 72.86760504, 71.16211928, 69.2206452, 66.70101056, 63.69381007999999, 60.83599752, 56.81246231999999, 53.02291296, 48.8655852, 43.61953007999999, 38.81484616, 33.38600504, 27.5007872, 21.051217599999994, 14.332658799999995, 7.383513439999999, 0.004999999999999998]]
yerr += [[1.4210854715202004e-14, 0.049306220395890915, 0.051810812871445705, 0.09188143933131214, 0.2700875190825478, 0.16455961452028864, 0.1947669334679587, 0.2483767625394451, 0.2705307039224225, 0.2622439860086191, 0.3808194392351655, 0.2714021329024031, 0.36298520628406833, 0.378197355810484, 0.5773526329020968, 0.2970706064949378, 0.3592747463462703, 0.28825240180036543, 0.2531561508458207, 0.2997313953792203, 8.673617379884035e-19], [1.4210854715202004e-14, 0.23550906065965546, 0.3038049100388449, 0.18662849158970898, 0.33601828895677804, 0.3191669376152773, 0.18005914544131382, 0.6108547898975616, 0.4562148164156765, 0.5053612308361763, 0.4793124263274216, 0.38787312078598035, 0.3311759470581992, 0.5290538849489249, 0.35370746799547487, 0.3107257936580759, 0.32657301068658806, 0.3705900162998805, 0.22195870192858866, 0.1728001468844562, 8.673617379884035e-19], [1.4210854715202004e-14, 0.05280187581637864, 0.051345577515495955, 0.09378167589456762, 0.26151593192799033, 0.16457873450438806, 0.19989051076154565, 0.25866288270560944, 0.2666438745682356, 0.24120180000507765, 0.3752871840116387, 0.27560703337900166, 0.36015791356165944, 0.35350030984716385, 0.5676641379962117, 0.31637846661754726, 0.355283865327679, 0.3416809104759582, 0.2592961208947639, 0.3139033222888447, 8.673617379884035e-19]]


x = [[0.05*i for i in range(n_hard)] for _ in range(4)]
y = np.array(y);yerr = np.array(yerr)
y = y[[5,4,3,0],:n_hard];yerr = yerr[[5,4,3,0],:n_hard]
print((y-y[[0]]).min(axis=1))
line_plot(x, y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/FCC_loss_ea2.eps',
		'Failure Rate (%)','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,linestyles=linestyles,
		yerr=yerr)
envs = ['NR','RR','Ours']
methods_tmp = ['']
y = [75.99,72.38, 73.74]
yerr = np.array([0,0,0.22])
bar_plot(y,yerr,envs,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/ffa_imagenet.eps',
		'#4f646f','FFA (%)',ylim=(72,77),labelsize=32)
y = [  100, 158, 154  ]
bar_plot(y,None,envs,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/flops_imagenet.eps',
		'#4f646f','FLOPS (%)',labelsize=32,ylim=(0,190))

methods = ['NR','RR','Ours']
c_imagenet = [[17.98,5.43,1.41],
[18.87,5.4,1.45],]
y = np.array(c_imagenet)
for i in [0,1]:
	print((y[:,2]-y[:,i])/y[:,i])
groupedbar(y,None,'Consistency (%)', 
	f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/re_imagenet.eps',methods=methods,labelsize=28,xlabel='',
	envs=['Jitter','Fail-stop'],sep=1,width=0.3,bbox_to_anchor=(0.77, 0.9),legloc=None,ncol=1,lgsize=24,
	yticks=range(0,20,2),use_barlabe_y=True)
exit(0)

methods = ['TR','NR','RR','Ours']
flops_vs_nodes = [[200,100,56*2,106],
[300,100,56*3,159],
[400,100,56*4,212]]
y = np.array(flops_vs_nodes)
groupedbar(y,None,'FLOPS (%)', 
	f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/flops_vs_nodes.eps',methods=methods,labelsize=28,xlabel='Cluster Size',
	envs=[2,3,4],sep=1,width=0.2,bbox_to_anchor=(0.17, 1),legloc=None,ncol=1,lgsize=20)

methods = ['NR','RR','Ours']
y = [[94.11,92.21215654952076,93.54],
[94.11,92.21215654952076,93.74],
[94.11,92.21215654952076,93.14]]
yerr =  [[0,0,0.23], 
[0,0,0.23], 
[0,0,0.31]]
y = np.array(y);yerr = np.array(yerr)
groupedbar(y,yerr,'Failure-Free Accuracy (%)', 
	f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/ffa_vs_nodes.eps',methods=methods,labelsize=28,xlabel='Cluster Size',
	envs=[2,3,4],sep=1,width=0.3,bbox_to_anchor=(0.15, 1.02),legloc=None,ncol=1,lgsize=16,yticks=range(92,96),ylim=(92,95))
exit(0)

# multi-node analysis
methods = ['NR','RR','Ours']

re_vs_nodes = [[0.18546325878594247,0.20588857827476037,0.01707268370607029,0.01707268370607029,0.01193889776357826,0.006509584664536749],
[0.266172124600639, 0.3088937699680511,0.01707268370607029,0.017092651757188503,0.0027975239616613414,0.002635782747603843],
[0.29221645367412136,0.3563059105431311,0.017092651757188503,0.017152555910543134,0.007637779552715651,0.006968849840255598]]

y = np.array(re_vs_nodes)*100
for i in range(2):
	y1 = y[:,[0+i,2+i,4+i]]
	for i in [0,1]:
		print((y1[:,2]-y1[:,i])/y1[:,i])
	groupedbar(y1,None,'Consistency (%)', 
		f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/re_vs_nodes{i}.eps',methods=methods,labelsize=28,xlabel='Cluster Size',
		envs=[2,3,4],sep=1,width=0.3,bbox_to_anchor=(0.5, 1.25),use_barlabel_x=True,legloc='upper left',ncol=1,lgsize=20)
exit(0)

envs = ['R20','R32','R44','R56','R110']
methods_tmp = ['NR','RR','Ours']
flops_res = [0.8677293603320655,0.7053025247728113,0.6375213554749708,0.6003251876612296,0.5360070051652971]
acc_par = [[0.9265175718849841, 0.922923322683706, 0.9158346645367412, 0.9186301916932907],
[0.9321086261980831, 0.9285143769968051, 0.9230231629392971, 0.9238218849840255],
[0.9306110223642172, 0.9298123003194888, 0.9238218849840255, 0.9258186900958466],
[0.9385982428115016, 0.9366014376996805, 0.9339057507987221, 0.9327076677316294],
[0.9389976038338658, 0.9381988817891374, 0.9377995207667732, 0.9354033546325878]
]
acc_base = [0.9243,0.9336,0.9379,0.9411,0.9446]
acc_par = np.array(acc_par)
acc_base = np.array(acc_base)
acc_par_mean = acc_par.mean(axis=1)
acc_par_std = acc_par.std(axis=1)
acc_sp = [0.909841214057508,0.9237185303514377,0.9245143769968051,0.9221215654952076,0.9337998402555911]
acc_sp = np.array(acc_sp)
flops_sp = [0.2527390666182668,0.2516129511941694,0.25114302010213746,0.25088513674100354,0.250439214753364]
y = np.concatenate((acc_base.reshape(5,1),acc_sp.reshape(5,1),acc_par_mean.reshape(5,1)),axis=1)*100
yerr = np.concatenate((np.zeros((5,1)),np.zeros((5,1)),acc_par_std.reshape(5,1)),axis=1)*100
groupedbar(y,yerr,'FFA (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/acc_res.eps',methods=methods_tmp,labelsize=24,
	envs=envs,ncol=1,width=.25,sep=1,legloc='lower right',ylim=(90,95),lgsize=20)




x0 = np.array([0.1*i for i in range(11)])
x = [[0.1*i for i in range(11)] for _ in range(2)]
for sn in [2,3,4]:
	methods_tmp = ['Two+','One']
	one = sn*(1-x0)*x0**(sn-1)
	twoormore = 1-x0**sn-one
	y = np.stack((twoormore,one),axis=0)
	if sn==2:
		line_plot(x,y,methods_tmp,colors,
				f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/prob{sn}.eps',
				'Failure Rate (%)','Probability (%)',lbsize=36,linewidth=8,markersize=16,bbox_to_anchor=(0.35,.52),ncol=1,lgsize=32,
				linestyles=linestyles,legloc='best',use_probarrow=True,xticks=[0.2*i for i in range(6)],yticks=[0.2*i for i in range(6)])
	else:
		line_plot(x,y,methods_tmp,colors,
				f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/prob{sn}.eps',
				'Failure Rate (%)','',lbsize=36,linewidth=8,markersize=16,linestyles=linestyles,legloc='best',use_probarrow=True,ncol=0,lgsize=32,
				xticks=[0.2*i for i in range(6)],yticks=[0.2*i for i in range(6)])

n_soft = 11;n_hard = 11;
# soft
# our 2
y = [[0.9361621405750797, 0.9361621405750797, 0.9361621405750797, 0.9361621405750797, 0.9361621405750797, 0.9361621405750797, 0.9310283546325879, 0.928302715654952, 0.9115654952076678, 0.8704392971246007, 0.8034644568690096, 0.6685003993610223, 0.48662939297124586, 0.2595147763578275, 0.12424121405750799, 0.10054512779552716, 0.10054512779552716, 0.10054512779552716, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9357128594249202, 0.9110862619808306, 0.8908785942492013, 0.8381569488817892, 0.7364376996805111, 0.6274420926517571, 0.48725039936102243, 0.347344249201278, 0.19965854632587862, 0.1232288338658147, 0.10026757188498403, 0.10026757188498403, 0.10026757188498403, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.936585463258786, 0.9334045527156551, 0.9168450479233226, 0.8823781948881788, 0.8129053514376995, 0.6865894568690095, 0.5060303514376997, 0.27870607028754, 0.1334664536741214, 0.10055511182108626, 0.10055511182108626, 0.10055511182108626, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9409265175718851, 0.9377256389776356, 0.9288957667731628, 0.8936142172523962, 0.8093889776357825, 0.6444888178913737, 0.3706968849840256, 0.1675199680511182, 0.10401357827476039, 0.1010902555910543, 0.10055511182108626, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.940658945686901, 0.9384684504792332, 0.9196585463258786, 0.8655091853035144, 0.7332148562300318, 0.4475998402555911, 0.19704273162939295, 0.10739217252396165, 0.10132787539936101, 0.10079273162939298, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr = [[0.0015818931565765423, 0.0015818931565765423, 0.0015818931565765423, 0.0015818931565765423, 0.0015818931565765423, 0.0015818931565765423, 0.007950516956585293, 0.009402756606651975, 0.0207778904716307, 0.02601873870311099, 0.023558931316111736, 0.040663586900461166, 0.0582144948670468, 0.056437040845429116, 0.015563681111892422, 0.0010904841391130044, 0.0010904841391130044, 0.0010904841391130044, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.011556293729309922, 0.011891925190609396, 0.018496022790526173, 0.02518926783764432, 0.04295317334658766, 0.04737095813661629, 0.050758024959229736, 0.05090586668309052, 0.04092125306618384, 0.0161616798527303, 0.0008027156549520825, 0.0008027156549520825, 0.0008027156549520825, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.007206944074725552, 0.008852499184261866, 0.0205918559628513, 0.024264865236015995, 0.021985150822558305, 0.03959105855099216, 0.058516116748041136, 0.05815391842397155, 0.02320750608166582, 0.0011111211234452062, 0.0011111211234452062, 0.0011111211234452062, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0008027156549520574, 0.004350268506307805, 0.009198834707969082, 0.017472731159496324, 0.03827052610561039, 0.06653776752906916, 0.07500756485702258, 0.0422339690728622, 0.007634880372308535, 0.0017938424056774941, 0.0011111211234452062, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.001605431309904148, 0.0027368798271916097, 0.01217598070173841, 0.025985757726121644, 0.05647870985657385, 0.08325582487535006, 0.04921398200065732, 0.010965830472881065, 0.0017910629489219492, 0.0012161198878535714, 0.0, 0.0, 0.0]]
# our 3
y += [[0.9406849041533546, 0.9406849041533546, 0.9406849041533546, 0.9406849041533546, 0.9406849041533546, 0.9406849041533546, 0.9406849041533546, 0.9404273162939296, 0.937286341853035, 0.926563498402556, 0.8904492811501598, 0.7992931309904152, 0.6213977635782747, 0.34459265175718856, 0.1542871405750799, 0.10373602236421726, 0.10053514376996806, 0.10053514376996806, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9356928913738018, 0.9110163738019169, 0.890858626198083, 0.8378374600638978, 0.7353614217252396, 0.6265575079872204, 0.4870806709265175, 0.3476637380191693, 0.2002056709265176, 0.12333865814696485, 0.10027755591054313, 0.10027755591054313, 0.10027755591054313, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.936495607028754, 0.9332248402555912, 0.9165355431309903, 0.881571485623003, 0.8125479233226838, 0.6862699680511182, 0.5066273961661342, 0.27932308306709264, 0.13355630990415335, 0.10055511182108626, 0.10055511182108626, 0.10055511182108626, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9376257987220447, 0.9287260383386581, 0.893246805111821, 0.8088099041533546, 0.6442791533546325, 0.37031549520766777, 0.16720047923322684, 0.10402356230031948, 0.10108027156549522, 0.10055511182108626, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057508, 0.9384484824281148, 0.9195786741214057, 0.8651497603833865, 0.7327376198083067, 0.44747803514376994, 0.19641573482428115, 0.1074720447284345, 0.10134784345047924, 0.10082268370607028, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr += [[0.0019144613076450085, 0.0019144613076450085, 0.0019144613076450085, 0.0019144613076450085, 0.0019144613076450085, 0.0019144613076450085, 0.0019144613076450085, 0.002148773847323937, 0.0055255176711369374, 0.010434333914610792, 0.0167217139637815, 0.042028625920111154, 0.06461534053529323, 0.08013724585721935, 0.031196939522469977, 0.007775357319861932, 0.0010740064615502457, 0.0010740064615502457, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.011433815744090537, 0.011828733879114674, 0.018615617233885082, 0.02541205679209715, 0.042655567821735345, 0.04822659134523788, 0.050813401621673186, 0.05005774959648743, 0.04079353605173561, 0.01619306713917164, 0.0008326677316293981, 0.0008326677316293981, 0.0008326677316293981, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007239541060915581, 0.008939477936044047, 0.020747420960426644, 0.024813058229312873, 0.022115932662648732, 0.03954571190223211, 0.05815786428546387, 0.05815979473079301, 0.023312819402994202, 0.0011102236421725309, 0.0011102236421725309, 0.0011102236421725309, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008027156549520573, 0.004258329577667321, 0.00921606877190535, 0.016993442208123138, 0.038028286839334124, 0.06605949822053465, 0.0747304216793538, 0.0420998703275585, 0.007784120296539298, 0.0017696571944327963, 0.0011102236421725309, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016353833865814685, 0.0026381453311676616, 0.01194902677686417, 0.025538158552235644, 0.05602670017552804, 0.08245335711889372, 0.048753299638632557, 0.011140897787994588, 0.0017882780573402189, 0.0012569344953712006, 0.0, 0.0, 0.0]]
# our 4
y += [[0.9345247603833867, 0.9345247603833867, 0.9345247603833867, 0.9345247603833867, 0.9345247603833867, 0.9345247603833867, 0.9345247603833867, 0.9345247603833867, 0.9339896166134185, 0.9310563099041534, 0.9119908146964855, 0.8528474440894568, 0.7065774760383385, 0.4184145367412141, 0.1764157348242812, 0.10675918530351439, 0.10080271565495207, 0.10080271565495207, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9357527955271564, 0.9111761182108626, 0.8908885782747603, 0.8376976837060702, 0.7354812300319489, 0.6267352236421725, 0.4866613418530351, 0.34755391373801914, 0.19983825878594252, 0.12326876996805111, 0.10027755591054313, 0.10027755591054313, 0.10027755591054313, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9365255591054314, 0.9333346645367412, 0.9167452076677314, 0.8821186102236422, 0.8127955271565493, 0.6858506389776358, 0.5058426517571883, 0.27859624600638977, 0.13352635782747604, 0.10054512779552716, 0.10054512779552716, 0.10054512779552716, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.940836661341853, 0.9376257987220447, 0.928805910543131, 0.8935343450479232, 0.808380591054313, 0.6440115814696485, 0.37034744408945686, 0.16748003194888178, 0.10399361022364217, 0.10105031948881789, 0.10054512779552716, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405690894568689, 0.9383785942492013, 0.9196285942492013, 0.8650099840255592, 0.7328773961661341, 0.447220447284345, 0.19672523961661342, 0.10743210862619808, 0.10130790734824283, 0.10080271565495207, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr += [[0.002190487016564894, 0.002190487016564894, 0.002190487016564894, 0.002190487016564894, 0.002190487016564894, 0.002190487016564894, 0.002190487016564894, 0.002190487016564894, 0.0020472618715187195, 0.0027064047081570845, 0.012890280244928377, 0.030928412725281542, 0.056110757447462516, 0.08181597177431865, 0.03592954866565217, 0.009677850263054464, 0.0012269810926168812, 0.0012269810926168812, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.011277071558759861, 0.011745193579417824, 0.01850952400805699, 0.025375162486789126, 0.043383392649159865, 0.04800257810899363, 0.051091313217208605, 0.050641403703752065, 0.040719386688620574, 0.016239794023849938, 0.0008326677316293981, 0.0008326677316293981, 0.0008326677316293981, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.007239159651383999, 0.008888628075875184, 0.02049297300737127, 0.02418939017349594, 0.02228240863539254, 0.0400747047700476, 0.058286439674499396, 0.05830741392113261, 0.023344829522595957, 0.0010904841391130044, 0.0010904841391130044, 0.0010904841391130044, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007727635782747377, 0.004300397390956596, 0.009038229923725726, 0.017249481139072265, 0.038447551991748465, 0.06680925657313624, 0.07555927112795831, 0.04200382448384294, 0.00763070131829235, 0.0017132820585712787, 0.0010904841391130044, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0015754792332268283, 0.002748419268439603, 0.012358142789637728, 0.02571815536748662, 0.05601754597753212, 0.08296926109531734, 0.04870243935930517, 0.011055117418660044, 0.0017295663798930934, 0.0012269810926168812, 0.0, 0.0, 0.0]]
# par 2
y += [[0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9195027955271566, 0.916361821086262, 0.9001417731629392, 0.8658666134185303, 0.7985283546325879, 0.6747284345047923, 0.4974920127795527, 0.2750399361022364, 0.13300718849840257, 0.10050519169329072, 0.10050519169329072, 0.10050519169329072, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9356230031948881, 0.9108666134185303, 0.8906789137380191, 0.8376078274760385, 0.7356888977635783, 0.6268949680511182, 0.48668130990415326, 0.34756389776357827, 0.19991613418530352, 0.12354832268370605, 0.10027755591054313, 0.10027755591054313, 0.10027755591054313, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9365355431309904, 0.9333546325878593, 0.9166054313099041, 0.8815814696485622, 0.8128055111821085, 0.6858706070287539, 0.5061182108626198, 0.278314696485623, 0.13375599041533545, 0.10056509584664537, 0.10056509584664537, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9375559105431309, 0.9288258785942493, 0.8937639776357826, 0.8087599840255593, 0.6444089456869009, 0.36991813099041526, 0.16772963258785942, 0.1040335463258786, 0.1010902555910543, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057508, 0.9384384984025559, 0.9198182907348242, 0.8652096645367411, 0.733324680511182, 0.4475878594249201, 0.1970626996805112, 0.10753194888178912, 0.10135782747603835, 0.1008326677316294, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.007044772864410344, 0.008707577836471801, 0.020186517462215836, 0.02407675566744983, 0.021695211821101137, 0.038189103394798134, 0.05702545049989327, 0.05715779974260247, 0.022739586454417218, 0.001012600722084609, 0.001012600722084609, 0.001012600722084609, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.011516300617118681, 0.011933301623863055, 0.018771435433091458, 0.025571448634478005, 0.042824220073934785, 0.04757934949420338, 0.05121910594981522, 0.051097571743243475, 0.040876990297098244, 0.01626493568176763, 0.0008326677316293981, 0.0008326677316293981, 0.0008326677316293981, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00716892552996913, 0.008789391749593811, 0.020736690240795854, 0.02483750453469489, 0.022162773815996684, 0.04004434332997258, 0.059203232432059966, 0.05848451351469494, 0.02348073221270085, 0.001130412167050239, 0.001130412167050239, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008027156549520573, 0.004385838676700178, 0.009394513719224704, 0.017328614959435877, 0.03881415853488955, 0.06677064837450802, 0.07511074265193961, 0.042133601817288986, 0.0077749396301703974, 0.0017794480326670256, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016353833865814685, 0.0026586018470994037, 0.012051248231448422, 0.026176297951647, 0.056137190694613814, 0.08268571722130168, 0.049086420346364644, 0.011245710694222873, 0.0017964810704380225, 0.001272704430868878, 0.0, 0.0, 0.0]]
# par 3
y += [[0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9237539936102236, 0.9206030351437701, 0.9118230830670926, 0.8773901757188497, 0.7946126198083066, 0.6326277955271564, 0.36457468051118214, 0.16612220447284345, 0.10397364217252396, 0.10103035143769967, 0.10050519169329072, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9356230031948881, 0.9108666134185303, 0.8906789137380191, 0.8376078274760385, 0.7356888977635783, 0.6268949680511182, 0.4869488817891375, 0.3478514376996805, 0.19991613418530352, 0.12354832268370605, 0.10027755591054313, 0.10027755591054313, 0.10027755591054313, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9365355431309904, 0.9333546325878593, 0.9166054313099041, 0.8815814696485622, 0.8128055111821085, 0.686138178913738, 0.5064057507987221, 0.278582268370607, 0.13375599041533545, 0.10056509584664537, 0.10056509584664537, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9375559105431309, 0.9288258785942493, 0.8937639776357826, 0.8087599840255593, 0.6446964856230032, 0.3701857028753993, 0.16772963258785942, 0.1040335463258786, 0.1010902555910543, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057508, 0.9384384984025559, 0.9198182907348242, 0.8652096645367411, 0.7333346645367411, 0.4475878594249201, 0.19707268370607028, 0.10753194888178912, 0.10135782747603835, 0.1008326677316294, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0008027156549520573, 0.004291754367296022, 0.00930907223682859, 0.016816316981063704, 0.03777006543366791, 0.06537943005075132, 0.07252320705530804, 0.040811284189901695, 0.007672909094667608, 0.0017253644124193327, 0.001012600722084609, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.011516300617118681, 0.011933301623863055, 0.018771435433091458, 0.025571448634478005, 0.042824220073934785, 0.04757934949420338, 0.05132584394690321, 0.0510769822069282, 0.040876990297098244, 0.01626493568176763, 0.0008326677316293981, 0.0008326677316293981, 0.0008326677316293981, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00716892552996913, 0.008789391749593811, 0.020736690240795854, 0.02483750453469489, 0.022162773815996684, 0.040240782456441776, 0.05930908028804448, 0.05875314897947808, 0.02348073221270085, 0.001130412167050239, 0.001130412167050239, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008027156549520573, 0.004385838676700178, 0.009394513719224704, 0.017328614959435877, 0.03881415853488955, 0.0671013041402841, 0.07533652243980228, 0.042133601817288986, 0.0077749396301703974, 0.0017794480326670256, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016353833865814685, 0.0026586018470994037, 0.012051248231448422, 0.026176297951647, 0.05670888775017273, 0.082760703361186, 0.04921967381115455, 0.011245710694222873, 0.0017964810704380225, 0.001272704430868878, 0.0, 0.0, 0.0]]
# par 4
y += [[0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9234864217252395, 0.9213458466453675, 0.9028654153354634, 0.8494249201277956, 0.7191413738019169, 0.44055710862619807, 0.19515575079872208, 0.10736222044728434, 0.10130790734824283, 0.10078274760383385, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9356230031948881, 0.9108666134185303, 0.8906789137380191, 0.8376078274760385, 0.7359764376996805, 0.6271525559105431, 0.48668130990415326, 0.34756389776357827, 0.1996385782747604, 0.12354832268370605, 0.10027755591054313, 0.10027755591054313, 0.10027755591054313, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9365355431309904, 0.9333546325878593, 0.9166054313099041, 0.8818490415335465, 0.8128055111821085, 0.6858706070287539, 0.5061182108626198, 0.2780371405750799, 0.13375599041533545, 0.10056509584664537, 0.10056509584664537, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9375559105431309, 0.9288258785942493, 0.8937639776357826, 0.8087599840255593, 0.6444089456869009, 0.36989816293929706, 0.16772963258785942, 0.1040335463258786, 0.1010902555910543, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057508, 0.9384384984025559, 0.9198182907348242, 0.8652096645367411, 0.7330471246006389, 0.44731030351437695, 0.19679512779552716, 0.10753194888178912, 0.10135782747603835, 0.1008326677316294, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.001605431309904148, 0.0026963524205096683, 0.011763627008472412, 0.025490313086122564, 0.05551887229238683, 0.08069847300483154, 0.047916154615680016, 0.010931668385050154, 0.0017601868207663739, 0.0011992739023270887, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.011516300617118681, 0.011933301623863055, 0.018771435433091458, 0.025571448634478005, 0.04276404563238693, 0.047196433244434226, 0.05121910594981522, 0.051097571743243475, 0.040922568440095924, 0.01626493568176763, 0.0008326677316293981, 0.0008326677316293981, 0.0008326677316293981, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00716892552996913, 0.008789391749593811, 0.020736690240795854, 0.02462051795780527, 0.022162773815996684, 0.04004434332997258, 0.059203232432059966, 0.0584910092544927, 0.02348073221270085, 0.001130412167050239, 0.001130412167050239, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008027156549520573, 0.004385838676700178, 0.009394513719224704, 0.017328614959435877, 0.03881415853488955, 0.06677064837450802, 0.07460912825681622, 0.042133601817288986, 0.0077749396301703974, 0.0017794480326670256, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016353833865814685, 0.0026586018470994037, 0.012051248231448422, 0.026176297951647, 0.05649477108583077, 0.08262743267048649, 0.04880557644762104, 0.011245710694222873, 0.0017964810704380225, 0.001272704430868878, 0.0, 0.0, 0.0]]

y,yerr = np.array(y)*100,np.array(yerr)*100
nodes = 2
for sel in [[2,1,15,0],[3,1,20,5],[4,1,20,10]]:
	y1 = y[sel,4:n_soft];yerr1 = yerr[sel,4:n_soft]
	methods_tmp = ['TR','NR','RR','Ours']
	x = [[-1+0.1*i for i in range(4,n_soft)]for _ in range(4)]
	line_plot(x,y1,methods_tmp,colors,
			f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/soft_ea_nodes{nodes}.eps',
			'Jitter Level','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,
			yerr=yerr1)	
	nodes += 1
# hard
y = [[0.9363019169329073, 0.9326737220447283, 0.9261681309904155, 0.9169788338658146, 0.9042671725239618, 0.8814596645367413, 0.8548003194888179, 0.8337160543130991, 0.798594249201278, 0.7647723642172524, 0.7231709265175719, 0.6767292332268371, 0.6501936900958467, 0.5802056709265175, 0.5274261182108625, 0.46292931309904156, 0.39433905750798726, 0.33431309904153356, 0.24523562300319496, 0.17307108626198087, 0.09999999999999999], [0.9411940894568691, 0.9006709265175719, 0.8524800319488817, 0.8144888178913737, 0.7751038338658147, 0.7340654952076677, 0.67810303514377, 0.6511741214057508, 0.5978973642172524, 0.5701497603833865, 0.5325918530351437, 0.46763378594249205, 0.4381629392971246, 0.3936521565495208, 0.3590315495207668, 0.3014297124600639, 0.2696385782747604, 0.230301517571885, 0.17648162939297127, 0.13472643769968054, 0.09999999999999999], [0.9411940894568691, 0.937745607028754, 0.9318690095846645, 0.9229892172523962, 0.9098781948881788, 0.8875099840255594, 0.8606709265175718, 0.8402256389776358, 0.8037859424920126, 0.7696545527156549, 0.7281928913738019, 0.6813817891373802, 0.6546365814696485, 0.5844189297124601, 0.5321485623003195, 0.46674321086261983, 0.3972144568690096, 0.33664936102236426, 0.2473123003194889, 0.17442891373801922, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9400638977635782, 0.9382208466453674, 0.935892571884984, 0.9304712460063899, 0.9187759584664539, 0.902869408945687, 0.8885623003194889, 0.8597583865814699, 0.8414856230031947, 0.7936002396166133, 0.7650499201277955, 0.7125419329073482, 0.6528055111821086, 0.5939117412140574, 0.5140275559105432, 0.4321186102236422, 0.3104512779552716, 0.21243610223642176, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9409165335463259, 0.938290734824281, 0.9390934504792334, 0.9357927316293928, 0.9242272364217252, 0.918113019169329, 0.9025339456869009, 0.8888977635782748, 0.8546265974440894, 0.8358765974440894, 0.794151357827476, 0.7389077476038338, 0.6802316293929712, 0.596273961661342, 0.5136421725239615, 0.37125, 0.2529013578274761, 0.09999999999999999]]
yerr = [[0.0, 0.003232679469800827, 0.006721895262968119, 0.00842487572583564, 0.008134901927732903, 0.00870537069066773, 0.013092925498863617, 0.012673107918681975, 0.018397486065869863, 0.018866247107201035, 0.01866136798822496, 0.019868267455407713, 0.02415092329764275, 0.013815579993989897, 0.0280572098092225, 0.014832873340755198, 0.018444084134533467, 0.01864215199124998, 0.012424224101828418, 0.008321246342186986, 0.0], [1.1102230246251565e-16, 0.0069597535771570365, 0.015125446937568229, 0.017518714939407903, 0.013206165550282875, 0.020330320011702128, 0.02237894632782985, 0.025534139101776564, 0.026913925755291435, 0.025000772633616997, 0.016151432632817393, 0.028374656074505223, 0.026125064134430956, 0.020006601066846555, 0.011797026078106697, 0.021348493695650218, 0.012442484251411768, 0.01876499927860723, 0.014249324398814786, 0.008147931144928894, 0.0], [1.1102230246251565e-16, 0.0031350355350767568, 0.006958692488708289, 0.008142436772168872, 0.008495461498120725, 0.009034378069027916, 0.012739351204582576, 0.01311098419799636, 0.019192596867821682, 0.01885160652041262, 0.01855054399369609, 0.020163293093335456, 0.023679847043820618, 0.01311875747419943, 0.028892859178731536, 0.015367758386209483, 0.01818672129335098, 0.01920620786946041, 0.012699398569217483, 0.00829825309938327, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 0.0013852762606664246, 0.002830120991057784, 0.0033434920035537706, 0.006495874348376657, 0.0067520957293272016, 0.006670327700196999, 0.010824996363497711, 0.013599638597536965, 0.013394509154927453, 0.014130003767069561, 0.013536424328876452, 0.007917615290921769, 0.029413445272071938, 0.018489286329811398, 0.017418659312816474, 0.020817007414061003, 0.012403565099641599, 0.014582124567555805, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0008326677316293774, 0.0030132470610544247, 0.0019863102799945595, 0.0032371438110367127, 0.004922762888838655, 0.007290255612493812, 0.011211880046567493, 0.00937524135725542, 0.014838895621706571, 0.010924068182304389, 0.013897741917512783, 0.026033126000339243, 0.01577495500749436, 0.02165730295288239, 0.019265135567461417, 0.015889984408020478, 0.019450716651676025, 0.0]]
y += [[0.9399960063897763, 0.9401657348242811, 0.9390195686900957, 0.9367571884984025, 0.9343250798722045, 0.9293350638977635, 0.9138258785942492, 0.900892571884984, 0.8823242811501597, 0.8556689297124601, 0.8241214057507985, 0.8028294728434504, 0.7485343450479233, 0.7065435303514376, 0.654966054313099, 0.5853354632587859, 0.49640375399361025, 0.4235642971246006, 0.3254812300319489, 0.22182308306709264, 0.09999999999999999], [0.9410942492012779, 0.8962100638977637, 0.8519608626198083, 0.8165255591054313, 0.7777296325878595, 0.7379712460063896, 0.6777755591054314, 0.6354412939297124, 0.5900818690095847, 0.5579093450479233, 0.5222703674121405, 0.4996904952076676, 0.44107028753993616, 0.39497603833865813, 0.34951876996805115, 0.3085742811501598, 0.2570367412140575, 0.21807108626198085, 0.18462460063897765, 0.1400579073482428, 0.09999999999999999], [0.9410942492012779, 0.9394888178913738, 0.9348302715654953, 0.9258326677316294, 0.9138298722044729, 0.8921665335463258, 0.8597404153354633, 0.8305051916932908, 0.7929492811501596, 0.7691673322683705, 0.7250638977635782, 0.6952835463258785, 0.6374860223642171, 0.5829972044728435, 0.5298282747603833, 0.4732308306709265, 0.38874600638977636, 0.334810303514377, 0.2586261980830672, 0.18538738019169332, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.940836661341853, 0.9376357827476038, 0.9356529552715654, 0.9314616613418532, 0.915972444089457, 0.9029293130990415, 0.8849600638977636, 0.8576357827476038, 0.8264676517571884, 0.8065734824281151, 0.7508706070287539, 0.7097284345047924, 0.6576817092651757, 0.5870626996805111, 0.4990994408945687, 0.4267591853035144, 0.3269488817891374, 0.22283146964856235, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057509, 0.9402615814696486, 0.9375658945686901, 0.9354053514376997, 0.9292611821086261, 0.9199860223642172, 0.9036521565495207, 0.8854173322683707, 0.8707907348242812, 0.8272923322683706, 0.7912400159744409, 0.7428534345047922, 0.6705571086261981, 0.5933366613418529, 0.5022523961661342, 0.38875000000000004, 0.25860822683706075, 0.09999999999999999]]
yerr += [[0.0, 0.0005414435648145597, 0.0013777796105021148, 0.002788898462405699, 0.004516621325515751, 0.006101685249872873, 0.004696952539763973, 0.005311021516774698, 0.010871652330180882, 0.013107185875209107, 0.017273162920830434, 0.02074417018033179, 0.022185067700907155, 0.024867087293651776, 0.01615677947056738, 0.017762104402648443, 0.010614665325765254, 0.019135983481389937, 0.02006971613228766, 0.016833463725008515, 0.0], [0.0, 0.00718655569240537, 0.013945152236419555, 0.013161866940018516, 0.015616080858299063, 0.017680944689895162, 0.03014832936527378, 0.02307728401099896, 0.02624825657296736, 0.01594731485315254, 0.024414466741554793, 0.034938185546443006, 0.017993988095187263, 0.017394882557374276, 0.02327871845257599, 0.011104951888247982, 0.012902653166971303, 0.017070123866907615, 0.012508344880617167, 0.011387009235368548, 0.0], [0.0, 0.001315383905625052, 0.005278459419625607, 0.003974956936471972, 0.00818211307047219, 0.012309671257495403, 0.017609078774857175, 0.006251945784651708, 0.01478408249970118, 0.01458423982673191, 0.017488109456241983, 0.03334872437130384, 0.022252111712762326, 0.01747537451155398, 0.017847682238936378, 0.014019569116702837, 0.021555733058477734, 0.01719994432500281, 0.017016259543734674, 0.013180177715550736, 0.0], [0.0, 0.0, 0.0007727635782747378, 0.002415949349991801, 0.004756599763192114, 0.006023717337671005, 0.004562482954283959, 0.006999575529339042, 0.010104409213436118, 0.01371171684796772, 0.01792576613410159, 0.021266101561065426, 0.02139855252162924, 0.025720748595399207, 0.016430204737468885, 0.019008381614312987, 0.010212401819678295, 0.019520304434589374, 0.01971188763272708, 0.017339189553161386, 0.0], [0.0, 0.0, 0.0, 0.0010904841391129736, 0.0012727044308688464, 0.002080265585421391, 0.003959342295875331, 0.004953398579105959, 0.007956312251896913, 0.009066244858239113, 0.011004197310867441, 0.013122840945873966, 0.01874860027030457, 0.023428915334346784, 0.011263901175692658, 0.02188558387872796, 0.018956927906578307, 0.022326372160841195, 0.01729497264555927, 0.01730145947855376, 0.0]]
y += [[0.9375, 0.9376098242811504, 0.9373003194888179, 0.9369009584664537, 0.9344468849840256, 0.9345786741214057, 0.9287400159744408, 0.9260882587859426, 0.9147384185303513, 0.8998282747603834, 0.8788039137380192, 0.8536561501597444, 0.8306230031948882, 0.7871126198083067, 0.7334165335463257, 0.6641313897763579, 0.5854612619808306, 0.4720806709265176, 0.38166333865814706, 0.25494209265175727, 0.09999999999999999], [0.9410942492012779, 0.9005591054313099, 0.854554712460064, 0.8204053514376998, 0.7662999201277956, 0.7354432907348243, 0.6911561501597443, 0.6506689297124602, 0.6010503194888178, 0.5602256389776358, 0.5316453674121405, 0.4751757188498402, 0.4474760383386581, 0.3854253194888179, 0.3576297923322684, 0.30457068690095845, 0.26639376996805114, 0.21806509584664538, 0.18148162939297124, 0.14338658146964858, 0.09999999999999999], [0.9410942492012779, 0.9402715654952075, 0.9338398562300319, 0.9234045527156549, 0.9038718051118211, 0.8906090255591057, 0.864239217252396, 0.8344868210862618, 0.8076417731629393, 0.7654932108626198, 0.7329492811501597, 0.6856230031948882, 0.6431170127795527, 0.5831269968051118, 0.5322723642172524, 0.45883785942492017, 0.4081110223642172, 0.32064696485623007, 0.26119408945686906, 0.1807088658146965, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9405491214057509, 0.9384085463258787, 0.9337999201277956, 0.9293989616613418, 0.9183007188498402, 0.9065555111821084, 0.8901557507987222, 0.8630011980830672, 0.8329992012779552, 0.7987799520766773, 0.7667511980830671, 0.7128953674121405, 0.6602955271565495, 0.576070287539936, 0.5087420127795526, 0.4027436102236422, 0.3303773961661342, 0.22306709265175723, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9394988019169329, 0.9391613418530351, 0.9334724440894568, 0.9316793130990414, 0.9205990415335463, 0.9063378594249201, 0.8857727635782748, 0.860155750798722, 0.838939696485623, 0.7940714856230031, 0.7411341853035143, 0.6710403354632588, 0.5905431309904154, 0.47716253993610225, 0.38564696485623007, 0.25715854632587865, 0.09999999999999999]]
yerr += [[0.0, 0.000958102383596927, 0.0007797773238724486, 0.0008329334510628683, 0.002885593423488358, 0.0031679934537966387, 0.0031266174401323767, 0.004244150431589769, 0.006886670505459622, 0.008609890217423428, 0.008057395646355882, 0.013485915123252186, 0.010003536048408689, 0.018376579543785007, 0.013902554654366104, 0.015662239865133053, 0.020895680509233242, 0.02295103300049234, 0.01710674500076731, 0.011439442743217107, 0.0], [0.0, 0.012304241159406561, 0.01659933520504075, 0.016592221233154392, 0.019009234486297297, 0.023655264500178953, 0.02606424205292366, 0.016559805285539728, 0.027055216704290227, 0.01463818441048046, 0.03174058596714932, 0.024509755530488175, 0.017211874447981723, 0.01793539966399801, 0.020436563234412775, 0.016362173581683164, 0.01531666929448429, 0.016267222702602075, 0.01752902893657775, 0.00819317534055581, 0.0], [0.0, 0.0012585195891636192, 0.0037637742421805023, 0.007388936851287273, 0.01030024950904621, 0.012644182598634585, 0.01210977091524217, 0.011456085107296378, 0.01986304468692187, 0.01956862173571549, 0.020343062893865692, 0.026567241003961357, 0.013585957496921465, 0.024419703586682627, 0.021919752558263447, 0.016757271777722422, 0.019953356813745785, 0.017593459925232972, 0.015916426202971748, 0.014170895392723504, 0.0], [0.0, 0.0, 0.0010923108020666826, 0.0027268090152144805, 0.006680833960101362, 0.005577824446071641, 0.006887063041350397, 0.009993382396625687, 0.01178930377944336, 0.013234114022494522, 0.019795506917778528, 0.01848671774955087, 0.012560791255871723, 0.02488832790468756, 0.013884327995717354, 0.017392562710434444, 0.02118876096333832, 0.020338477458197335, 0.020600348532264133, 0.012188802630206365, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0024739190469077435, 0.002748691993268106, 0.0026405745060775693, 0.002932398559279646, 0.007787561440127629, 0.008260506414500816, 0.008144073135628324, 0.013631073348135073, 0.010307992477655516, 0.018097395041730184, 0.014511523390347962, 0.016322608669369822, 0.021763065598096473, 0.023172948788172224, 0.017508323358227013, 0.011870277508342858, 0.0]]
y += [[0.9240215654952075, 0.9232288338658148, 0.9157188498402554, 0.9073342651757187, 0.8876976837060703, 0.8753274760383387, 0.8545507188498401, 0.8230311501597443, 0.7895587060702874, 0.7521705271565495, 0.7226497603833865, 0.660948482428115, 0.6381369808306709, 0.5676477635782747, 0.5208027156549521, 0.4473642172523961, 0.4015595047923323, 0.33284145367412143, 0.26164137380191693, 0.17886581469648563, 0.09999999999999999], [0.9410942492012779, 0.9018091054313098, 0.8498262779552717, 0.8257068690095848, 0.7746166134185304, 0.7414596645367413, 0.685615015974441, 0.6445327476038338, 0.6061481629392971, 0.5636761182108627, 0.5260303514376996, 0.462879392971246, 0.4484285143769967, 0.39091054313099044, 0.34775758785942495, 0.3054013578274761, 0.2637579872204473, 0.23126397763578277, 0.1882907348242812, 0.14283346645367415, 0.09999999999999999], [0.9410942492012779, 0.9402715654952077, 0.9323722044728433, 0.9241873003194888, 0.9039516773162941, 0.8915415335463258, 0.8700159744408944, 0.8375878594249201, 0.8037160543130991, 0.7655890575079872, 0.7361182108626197, 0.6727795527156548, 0.6493789936102237, 0.577242412140575, 0.5288498402555911, 0.4546226038338658, 0.4072304313099043, 0.33716453674121405, 0.2648961661341854, 0.1804432907348243, 0.09999999999999999], [0.9410942492012779, 0.9408166932907347, 0.9402515974440895, 0.9384085463258787, 0.9318989616613418, 0.9286162140575082, 0.9201437699680509, 0.9065355431309904, 0.8865575079872204, 0.8581829073482428, 0.837857428115016, 0.7867492012779552, 0.7699500798722045, 0.7056729233226837, 0.6498342651757189, 0.5834784345047924, 0.5045007987220448, 0.43298921725239625, 0.32922523961661343, 0.21766573482428125, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9408067092651757, 0.9410942492012779, 0.9391912939297125, 0.937020766773163, 0.9349500798722044, 0.9322144568690096, 0.9182907348242813, 0.9045247603833866, 0.8914616613418531, 0.8569928115015975, 0.8413238817891374, 0.7867991214057508, 0.7373242811501596, 0.6741793130990417, 0.5925459265175719, 0.5075658945686901, 0.3900758785942492, 0.2520447284345048, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 0.0012128368069627925, 0.002009163037287492, 0.004559026177608105, 0.004777369896755866, 0.010407846282822103, 0.007427135067536107, 0.019248634127235344, 0.011370192827218916, 0.01471354636924572, 0.016277193102212772, 0.01595375820113512, 0.01919176014303523, 0.02240241062057284, 0.01833338272659684, 0.01915234662750271, 0.02713681869023445, 0.03188325684489882, 0.0180093755956672, 0.013659963236110021, 0.0], [0.0, 0.010345833264467601, 0.01127144415383577, 0.011208998220441479, 0.014854433872319023, 0.015369851520777734, 0.014714270302442189, 0.01793048670346185, 0.018314599061651792, 0.017944712063340418, 0.01725911142752746, 0.013269655654528647, 0.021581123881472805, 0.02391488345358719, 0.025094410089816246, 0.02327696925279175, 0.0201780629438965, 0.017579351108637047, 0.009684814395103873, 0.008383231159874981, 0.0], [0.0, 0.0012569344953711665, 0.0020789378517483328, 0.004654561281949599, 0.004800964420190137, 0.010784371549703525, 0.007135823827939953, 0.01899487602826977, 0.011989139324796437, 0.014988820556830674, 0.017329109081774718, 0.01630580050574032, 0.019475204684981352, 0.02294639791117799, 0.018998743451960762, 0.019475320973374056, 0.027872041279210637, 0.032677910332194124, 0.01866876949151682, 0.013746080066002283, 0.0], [0.0, 0.0008326677316293774, 0.001287429933702177, 0.0029170753101952227, 0.0029982977936203565, 0.007894587892452634, 0.006544288021077912, 0.009274423756036935, 0.0074956872345891155, 0.009575261628382141, 0.018136136626103656, 0.014752077236044608, 0.01424235422342263, 0.017780697859066986, 0.016385824722797652, 0.016458639213330035, 0.03130713193536697, 0.024590686083732303, 0.018211125793317254, 0.017452154470294023, 0.0], [0.0, 0.0, 0.0008626198083066973, 0.0, 0.0012480607015504502, 0.0017967030034408194, 0.004184362642854235, 0.005712168422954852, 0.004985099046247042, 0.004980443462684424, 0.009575700514019235, 0.011466745457659152, 0.009685773812643609, 0.016924603618504327, 0.017819813262204918, 0.015965086743074376, 0.024587661264695405, 0.0209390271131603, 0.015833211561441844, 0.016652399778021024, 0.0]]
y += [[0.9240215654952075, 0.9234764376996806, 0.923516373801917, 0.9190974440894568, 0.9165914536741212, 0.9108726038338657, 0.9014896166134184, 0.8912420127795526, 0.8713897763578276, 0.8424640575079871, 0.8176377795527158, 0.785435303514377, 0.7473742012779552, 0.694151357827476, 0.6473622204472844, 0.5689856230031949, 0.4990435303514378, 0.41652156549520775, 0.32073881789137376, 0.21608426517571888, 0.09999999999999999], [0.9410942492012779, 0.8975658945686901, 0.8589696485623003, 0.8034524760383388, 0.7715095846645366, 0.7345746805111821, 0.6860363418530352, 0.6508107028753993, 0.5980850638977636, 0.5585842651757188, 0.5224241214057507, 0.47824480830670923, 0.43038538338658155, 0.3946305910543131, 0.36959464856230034, 0.3067332268370607, 0.26415335463258793, 0.22665535143769966, 0.179163338658147, 0.13858226837060705, 0.09999999999999999], [0.9410942492012779, 0.9378534345047923, 0.9340774760383385, 0.919598642172524, 0.9074500798722045, 0.8894728434504792, 0.8702535942492012, 0.8416453674121407, 0.8045287539936101, 0.7663258785942492, 0.7276257987220449, 0.6838079073482427, 0.6305551118210861, 0.5877136581469649, 0.5356749201277956, 0.4732068690095847, 0.39875199680511186, 0.33152755591054317, 0.25545327476038343, 0.17801517571884984, 0.09999999999999999], [0.9410942492012779, 0.9405690894568689, 0.9405491214057509, 0.9360303514376997, 0.9332947284345048, 0.9276357827476037, 0.9179532747603835, 0.9077955271565497, 0.8866553514376996, 0.8575798722044727, 0.8326337859424922, 0.7996625399361021, 0.7610922523961662, 0.707509984025559, 0.6598921725239618, 0.5791593450479233, 0.5070207667731629, 0.42290135782747607, 0.3255611022364218, 0.21819089456869012, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405391373801917, 0.9386461661341853, 0.9365954472843452, 0.9336421725239615, 0.9295686900958466, 0.9209564696485625, 0.9057048722044728, 0.8866673322683706, 0.8587579872204472, 0.8389376996805111, 0.7868630191693291, 0.7459584664536741, 0.6748841853035144, 0.5937639776357827, 0.5005171725239617, 0.3907288338658147, 0.25475039936102245, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 0.0010904841391129736, 0.0010126007220845707, 0.004134729142521429, 0.00218371341039642, 0.005013247567859326, 0.010185656488035202, 0.009078492393017787, 0.009883430119277361, 0.015834244522226742, 0.008654951533065745, 0.021520655816203327, 0.014040886245107208, 0.022084851307641158, 0.02706114232514029, 0.023405908326654182, 0.017169810530294295, 0.0199736737291584, 0.012012236526286099, 0.016240047400828097, 0.0], [0.0, 0.013099870820638272, 0.012371785510513251, 0.015595527174752868, 0.017539360965651618, 0.02322425173055862, 0.024809475529540007, 0.020850355957943892, 0.010770654419897779, 0.02192585931533004, 0.027229966478054715, 0.025194758032385897, 0.025242475879542387, 0.01168294631638859, 0.025997198693804072, 0.018987917468319335, 0.011808078959992318, 0.008424030425388784, 0.007759972351572618, 0.012148043723821354, 0.0], [0.0, 0.0026776263109509987, 0.004454819845088951, 0.008025523423986746, 0.007499248502279333, 0.01059625210976471, 0.014089726418141699, 0.010577976652659235, 0.015176348881322408, 0.015826847116877575, 0.01775390013438107, 0.019265099658803478, 0.023057026225632066, 0.017779198943035576, 0.02497166375823302, 0.01981493169318772, 0.013645474948625221, 0.014578924095960663, 0.01371948237786555, 0.01796103055417808, 0.0], [0.0, 0.0010505567249903464, 0.0010923108020666826, 0.004063605845931439, 0.0021042612327045727, 0.005074109283084919, 0.010443090097890012, 0.008994831100206737, 0.010225241246562775, 0.016206140160806272, 0.009095345580420564, 0.021915978964612337, 0.013990746504111053, 0.022033501595690418, 0.02692255714792926, 0.02394898335067615, 0.018279675371392475, 0.020551207942726985, 0.01183005987785799, 0.016130233221246546, 0.0], [0.0, 0.0, 0.0, 0.001110223642172503, 0.0014688181342531008, 0.0028886123966537073, 0.00566153798052375, 0.00661455550219548, 0.004736927213219304, 0.009448615305576492, 0.007741513500374126, 0.014541993398687579, 0.011646756738583388, 0.02027060808007669, 0.022839961389295983, 0.025815454234857427, 0.01876654991053932, 0.017831553388960773, 0.013387472764454303, 0.01487180030734817, 0.0]]
y += [[0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9232388178913737, 0.9226737220447285, 0.9213658146964855, 0.917857428115016, 0.9147064696485623, 0.9037000798722046, 0.8955031948881789, 0.86700678913738, 0.8444948083067093, 0.8219828274760383, 0.7737320287539935, 0.7325499201277955, 0.6675379392971246, 0.5834824281150159, 0.4925539137380192, 0.3878274760383387, 0.25537140575079875, 0.09999999999999999], [0.9410942492012779, 0.8939017571884984, 0.8524161341853036, 0.8197324281150159, 0.7710023961661342, 0.7344728434504793, 0.6852895367412141, 0.6469229233226838, 0.6081888977635781, 0.5661920926517572, 0.5187220447284344, 0.47493011182108624, 0.4336721246006389, 0.4088418530351438, 0.3504572683706071, 0.3089636581469649, 0.26171325878594254, 0.22540734824281153, 0.1829492811501598, 0.14669728434504795, 0.09999999999999999], [0.9410942492012779, 0.9384384984025559, 0.933047124600639, 0.9230770766773164, 0.910916533546326, 0.8918510383386582, 0.8645746805111821, 0.8353873801916933, 0.8049760383386582, 0.7751537539936102, 0.7177036741214057, 0.6803913738019169, 0.6292511980830671, 0.5959724440894567, 0.5290475239616613, 0.4603973642172523, 0.39669928115015973, 0.3313877795527157, 0.25884185303514384, 0.18994808306709268, 0.09999999999999999], [0.9410942492012779, 0.940836661341853, 0.9405591054313099, 0.9395686900958466, 0.9359404952076676, 0.9282807507987219, 0.9173582268370607, 0.9040375399361021, 0.8849321086261981, 0.8710563099041533, 0.8307807507987219, 0.7958885782747604, 0.7599241214057508, 0.7116174121405752, 0.6648682108626198, 0.5831829073482427, 0.5106509584664538, 0.41939496805111826, 0.33484225239616616, 0.22585463258785943, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9403115015974441, 0.9397164536741215, 0.9385183706070288, 0.9348702076677317, 0.9316493610223642, 0.9200638977635783, 0.9120766773162939, 0.8828414536741216, 0.8595507188498402, 0.8364696485623003, 0.7879492811501596, 0.746238019169329, 0.6802376198083067, 0.5930471246006389, 0.5002416134185304, 0.39390774760383385, 0.2591653354632588, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0016759322221975759, 0.0013503556046880745, 0.0023861445115810668, 0.005004903882514132, 0.0029662639070488846, 0.004210570758749211, 0.005632525338577535, 0.011664507761251787, 0.008229299014742144, 0.015970530297094956, 0.015792237280473784, 0.01877819658956077, 0.015564395988504016, 0.02197226639453255, 0.027570624298442554, 0.01758242676816212, 0.014954270655301033, 0.0], [0.0, 0.012312152646995918, 0.018789619921083463, 0.02597044881964034, 0.017637285261358708, 0.01810774660309283, 0.026021945684638392, 0.025133962032577797, 0.01972906597419759, 0.024338876322108665, 0.019659805722951944, 0.02133990069893078, 0.01659908442992253, 0.013965223836512133, 0.02685262272788746, 0.01660211893873421, 0.01506489666239101, 0.016357476471373856, 0.016645452807917495, 0.009576183514809327, 0.0], [0.0, 0.0033379869045637544, 0.0053328904067146245, 0.005352793939651994, 0.008071144802442897, 0.011602824210670434, 0.01845875298995951, 0.013260333977530254, 0.018665968852184002, 0.01865017695455214, 0.01903869684991489, 0.019347263079459116, 0.028884131326281818, 0.01639426381113413, 0.02640169655333176, 0.017320967209225956, 0.01512025884695038, 0.024198069097493845, 0.022066002592427265, 0.014432553731494836, 0.0], [0.0, 0.0007727635782747378, 0.001070287539936077, 0.0016999649945689354, 0.0034798699137927594, 0.005182523312911612, 0.012514619961583729, 0.00835432237098087, 0.013219206364698834, 0.008660138245027081, 0.01491078769696457, 0.01974529797649141, 0.02110260260025772, 0.015141660853755776, 0.02530645142618155, 0.015558194644209985, 0.02074099299905427, 0.023809581370065298, 0.014253928495162068, 0.016738644021597957, 0.0], [0.0, 0.0, 0.0, 0.001660516045678519, 0.0013795307866945321, 0.00225318908391696, 0.005120783642201577, 0.002948923542188125, 0.004791823056013743, 0.005981534951262851, 0.012067280125030245, 0.008553838821374398, 0.016051270911673632, 0.015955080745315375, 0.018959546662834476, 0.015317825200466293, 0.022804529892396033, 0.028631694689072337, 0.017311287953456503, 0.015046241206526551, 0.0]]

y,yerr = np.array(y)*100,np.array(yerr)*100
nodes = 2
for sel in [[2,1,15,0],[3,1,20,5],[4,1,20,10]]:
	y1 = y[sel,:n_hard];yerr1 = yerr[sel,:n_hard]
	methods_tmp = ['TR','NR','RR','Ours']
	x = [[5*i for i in range(n_hard)]for _ in range(4)]
	line_plot(x,y1,methods_tmp,colors,
			f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/hard_ea_nodes{nodes}.eps',
			'Failure Rate (%)','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,
			yerr=yerr1)	
	nodes += 1

# FCC 2-nodes no loss analysis
# CIFAR, IMAGENET EA and FR FCC
n_soft = 11;n_hard = 11;
x = [[-1+0.1*i for i in range(4,n_soft)] for _ in range(4)]
# y = [[0.09999999999999999, 0.10054512779552716, 0.10054512779552716, 0.11902755591054312, 0.24644369009584666, 0.4110902555910543, 0.5672503993610223, 0.6601238019169329, 0.7321505591054313, 0.793057108626198, 0.8280910543130989, 0.8515195686900958, 0.8795467252396165, 0.8962060702875398, 0.903905750798722, 0.9113977635782747, 0.915401357827476, 0.9167292332268369, 0.9250139776357826, 0.9292152555910542], [0.09999999999999999, 0.10026757188498403, 0.10026757188498403, 0.12053314696485622, 0.19403953674121407, 0.30107228434504796, 0.40742012779552716, 0.4781529552715654, 0.5490015974440895, 0.6168170926517572, 0.6582428115015976, 0.6940595047923324, 0.7466353833865815, 0.7764436900958467, 0.8043510383386583, 0.8349560702875399, 0.8493550319488816, 0.8560443290734824, 0.8767372204472844, 0.8892432108626197], [0.09999999999999999, 0.10055511182108626, 0.10055511182108626, 0.12967052715654953, 0.265960463258786, 0.43768769968051113, 0.5874161341853036, 0.6758865814696484, 0.7458226837060702, 0.8056110223642173, 0.838514376996805, 0.8573542332268371, 0.8893550319488819, 0.9001757188498403, 0.9084305111821086, 0.9165774760383385, 0.9192432108626198, 0.9202935303514378, 0.9301637380191693, 0.9328494408945687]]
# y += [[0.09999999999999999, 0.10049520766773161, 0.10049520766773161, 0.12854233226837058, 0.2586421725239617, 0.4218510383386582, 0.5637360223642173, 0.6486861022364215, 0.7149021565495206, 0.7717452076677316, 0.8030311501597444, 0.8208426517571885, 0.8512160543130991, 0.8616873003194888, 0.8696825079872204, 0.8772404153354632, 0.8797963258785944, 0.8807268370607029, 0.8902276357827477, 0.8928234824281148]]
# yerr = [[0.0, 0.0010904841391130044, 0.0010904841391130044, 0.012901423965101584, 0.05386465228666159, 0.06330897181706202, 0.061088301211427776, 0.042496575722181885, 0.03791339018992345, 0.0286516040794726, 0.024502583576222894, 0.025234458910697543, 0.025214803986917174, 0.02004981065222387, 0.01913403883501498, 0.0210146786784222, 0.020861977353891028, 0.020812112201925786, 0.012489402392714916, 0.009346223105772366], [0.0, 0.0008027156549520825, 0.0008027156549520825, 0.014877741975549278, 0.041380313427382555, 0.059803487736121326, 0.06442090733061269, 0.054010348189044745, 0.046256730645281255, 0.04990167454459851, 0.04680540980752168, 0.04270990091509234, 0.04092552622057239, 0.04201476212272375, 0.035258841989506476, 0.024606431299643745, 0.024426260082436837, 0.02257806012117962, 0.02338451126124495, 0.01813754932754337], [0.0, 0.0011111211234452062, 0.0011111211234452062, 0.020617885360086453, 0.058052933419270515, 0.06406899866962676, 0.06046125060499545, 0.04090110185496498, 0.03818100531319478, 0.022599839902736295, 0.024208484492046708, 0.02502698385338613, 0.021951503438406496, 0.0199137496482411, 0.019387677509606134, 0.020391848164711325, 0.020533313186606467, 0.020248210447870275, 0.01021375164308125, 0.008686921346931619]]
# yerr += [[0.0, 0.00099443300328881, 0.00099443300328881, 0.019505029461655107, 0.0552931148145234, 0.06093023519257489, 0.05759666397695378, 0.038977482134838315, 0.03628601704797119, 0.022477917272181132, 0.023863743957173233, 0.024283259850336362, 0.021449119587870866, 0.01941129579656337, 0.018824164775408433, 0.019649505483998873, 0.019729050513594238, 0.019467567337130353, 0.009863530877285642, 0.008319727257849869]]

y = [[0.937589856230032, 0.937589856230032, 0.937589856230032, 0.937589856230032, 0.937589856230032, 0.937589856230032, 0.9324560702875401, 0.9297304313099042, 0.9129732428115016, 0.8723322683706071, 0.8049800319488819, 0.6696465654952076, 0.48828873801916933, 0.2604373003194888, 0.12424121405750799, 0.10054512779552716, 0.10054512779552716, 0.10054512779552716, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9357128594249202, 0.9110862619808306, 0.8908785942492013, 0.8381569488817892, 0.7367052715654951, 0.6274420926517571, 0.48753793929712474, 0.3475918530351438, 0.2002136581469649, 0.1232288338658147, 0.10026757188498403, 0.10026757188498403, 0.10026757188498403, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.936585463258786, 0.9334045527156551, 0.9168450479233226, 0.8823781948881788, 0.8129053514376995, 0.6868769968051118, 0.5065555111821085, 0.279241214057508, 0.1334664536741214, 0.10055511182108626, 0.10055511182108626, 0.10055511182108626, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9409265175718851, 0.9377256389776356, 0.9288957667731628, 0.8936142172523962, 0.8093889776357825, 0.6447364217252395, 0.37123202875399364, 0.1675199680511182, 0.10401357827476039, 0.1010902555910543, 0.10055511182108626, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.940658945686901, 0.9384684504792332, 0.9196585463258786, 0.8655091853035144, 0.7331749201277955, 0.44840255591054323, 0.19702276357827475, 0.10739217252396165, 0.10132787539936101, 0.10079273162939298, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr = [[0.00154184519230557, 0.00154184519230557, 0.00154184519230557, 0.00154184519230557, 0.00154184519230557, 0.00154184519230557, 0.00818242981624913, 0.009621731150266646, 0.020934684302132835, 0.026350957571167625, 0.02374048145438981, 0.04038328750134762, 0.05747904030107691, 0.0560016594451472, 0.015563681111892422, 0.0010904841391130044, 0.0010904841391130044, 0.0010904841391130044, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.011556293729309922, 0.011891925190609396, 0.018496022790526173, 0.02518926783764432, 0.04289896786818774, 0.04737095813661629, 0.050856375685426385, 0.05087757640131233, 0.040948927025698516, 0.0161616798527303, 0.0008027156549520825, 0.0008027156549520825, 0.0008027156549520825, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.007206944074725552, 0.008852499184261866, 0.0205918559628513, 0.024264865236015995, 0.021985150822558305, 0.03980791207387507, 0.058693503367118885, 0.05849028854106433, 0.02320750608166582, 0.0011111211234452062, 0.0011111211234452062, 0.0011111211234452062, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0008027156549520574, 0.004350268506307805, 0.009198834707969082, 0.017472731159496324, 0.03827052610561039, 0.06682077740807381, 0.07503336147419472, 0.0422339690728622, 0.007634880372308535, 0.0017938424056774941, 0.0011111211234452062, 0.0, 0.0, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.001605431309904148, 0.0027368798271916097, 0.01217598070173841, 0.025985757726121644, 0.05703008951618122, 0.08281566140731882, 0.049300475613924026, 0.010965830472881065, 0.0017910629489219492, 0.0012161198878535714, 0.0, 0.0, 0.0]]
y += [[0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9240215654952075, 0.9195027955271566, 0.916361821086262, 0.9001417731629392, 0.8661341853035143, 0.7985283546325879, 0.6749860223642171, 0.4977595846645368, 0.27553514376996807, 0.13300718849840257, 0.10050519169329072, 0.10050519169329072, 0.10050519169329072, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9356230031948881, 0.9108666134185303, 0.8909464856230033, 0.8376078274760385, 0.7359764376996805, 0.6271525559105431, 0.4869488817891375, 0.3478514376996805, 0.20019369009584667, 0.12354832268370605, 0.10027755591054313, 0.10027755591054313, 0.10027755591054313, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9365355431309904, 0.9336222044728434, 0.9166054313099041, 0.8818490415335465, 0.8128055111821085, 0.686138178913738, 0.5064057507987221, 0.278582268370607, 0.13375599041533545, 0.10056509584664537, 0.10056509584664537, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9375559105431309, 0.9288258785942493, 0.8937639776357826, 0.8087599840255593, 0.6446964856230032, 0.3704432907348242, 0.16772963258785942, 0.1040335463258786, 0.1010902555910543, 0.10056509584664537, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057508, 0.9384384984025559, 0.9198182907348242, 0.8652096645367411, 0.7336122204472844, 0.4481429712460064, 0.19734025559105434, 0.10779952076677315, 0.10135782747603835, 0.1008326677316294, 0.09999999999999999, 0.09999999999999999, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.007044772864410344, 0.008707577836471801, 0.020186517462215836, 0.023856657240659756, 0.021695211821101137, 0.038374070574890194, 0.05711952506090232, 0.05742207970948171, 0.022739586454417218, 0.001012600722084609, 0.001012600722084609, 0.001012600722084609, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.011516300617118681, 0.011933301623863055, 0.01873585158265425, 0.025571448634478005, 0.04276404563238693, 0.047196433244434226, 0.05132584394690321, 0.0510769822069282, 0.04084833821425771, 0.01626493568176763, 0.0008326677316293981, 0.0008326677316293981, 0.0008326677316293981, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00716892552996913, 0.008659500781297834, 0.020736690240795854, 0.02462051795780527, 0.022162773815996684, 0.040240782456441776, 0.05930908028804448, 0.05875314897947808, 0.02348073221270085, 0.001130412167050239, 0.001130412167050239, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008027156549520573, 0.004385838676700178, 0.009394513719224704, 0.017328614959435877, 0.03881415853488955, 0.0671013041402841, 0.07504738352881136, 0.042133601817288986, 0.0077749396301703974, 0.0017794480326670256, 0.001130412167050239, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016353833865814685, 0.0026586018470994037, 0.012051248231448422, 0.026176297951647, 0.056351249782832145, 0.08254999738495626, 0.0494966678453765, 0.011847754005495594, 0.0017964810704380225, 0.001272704430868878, 0.0, 0.0, 0.0]]

y = np.array(y)*100
yerr = np.array(yerr)*100
y = y[[2,1,5,0],4:n_soft];yerr = yerr[[2,1,5,0],4:n_soft]
# start=2;y = y[:,start:];yerr = yerr[:,start:];x=[[0.1*i for i in range(1+start,21)] for _ in range(4)]
colors_tmp = colors#['k'] + colors[:2] + ['grey'] + colors[2:4]
linestyles_tmp = linestyles#['solid','dashed','dotted','solid','dashdot',(0, (3, 5, 1, 5)),]
methods_tmp = ['TR','NR','RR','Ours']
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/FCC_ea.eps',
		'Jitter Level','Effective Accuracy (%)',lbsize=24,linewidth=8,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr)	

# cifar
# y_ea_loss1 = [[0.9372004792332268, 0.9361701277955271, 0.9269009584664536, 0.9164237220447286, 0.9027735623003196, 0.8834584664536742, 0.8610463258785941, 0.8285802715654953, 0.8152276357827477, 0.759117412140575, 0.7229712460063897, 0.6756689297124601, 0.6312699680511182, 0.5680770766773162, 0.5175319488817891, 0.4741453674121406, 0.4014556709265175, 0.33900359424920135, 0.2573242811501598, 0.18030551118210864, 0.09999999999999999], [0.9411940894568691, 0.902811501597444, 0.8544488817891374, 0.8166713258785941, 0.7749001597444088, 0.7351876996805112, 0.6917911341853034, 0.6364616613418531, 0.6134464856230032, 0.5541433706070287, 0.5137739616613418, 0.46715055910543135, 0.43328873801916934, 0.39498402555910544, 0.3444468849840256, 0.32219249201277955, 0.2572484025559106, 0.23126397763578277, 0.18252396166134188, 0.14374400958466454, 0.09999999999999999], [0.9411940894568691, 0.9401437699680513, 0.9326417731629391, 0.9210263578274761, 0.9081449680511181, 0.8886900958466454, 0.8666873003194888, 0.8341613418530353, 0.8198901757188498, 0.7644788338658146, 0.7271046325878594, 0.6807507987220446, 0.6360423322683705, 0.5725099840255591, 0.5214756389776357, 0.4769908146964855, 0.4042711661341853, 0.34105031948881787, 0.25919129392971246, 0.18125399361022368, 0.09999999999999999]]
# y_ea_loss1 += [[0.9008586261980831, 0.8992831469648562, 0.8935583067092653, 0.8811441693290734, 0.8696525559105431, 0.8490475239616613, 0.8290754792332269, 0.8009005591054313, 0.7716573482428115, 0.7384464856230031, 0.7078993610223642, 0.6664656549520767, 0.6096126198083066, 0.5554253194888179, 0.5133246805111822, 0.453202875399361, 0.38915734824281145, 0.3231869009584665, 0.24736222044728443, 0.17868011182108628, 0.09999999999999999]]
# yerr_ea_loss1 = [[0.0, 0.002383087978836537, 0.003873038115141878, 0.0061453230457425975, 0.012967568025285573, 0.010063407266560125, 0.008824341552946188, 0.01611947452454557, 0.01890696132833872, 0.013174793019248588, 0.019867655361733523, 0.02406030887038924, 0.019344045784808497, 0.026841082314703498, 0.03579666570409922, 0.028536735524548405, 0.023743592117576182, 0.013090604293229717, 0.01902712161252409, 0.0103868882719809, 0.0], [1.1102230246251565e-16, 0.008922661919075328, 0.010855201934208752, 0.014047328679292339, 0.015535675044506014, 0.021138001603641914, 0.02019673453507974, 0.019017540855227846, 0.02697623136127924, 0.0165951113967507, 0.015869314630196252, 0.01722226165574532, 0.016248312393671693, 0.0225139485027758, 0.024553767381638156, 0.01572217242443465, 0.01882411807025787, 0.0094800814188465, 0.01617607815639621, 0.008617508759934122, 0.0], [1.1102230246251565e-16, 0.0021006389776357715, 0.004292221186753114, 0.005502080447956476, 0.012873009197567621, 0.009931106560055136, 0.008520796056967147, 0.016406406989810517, 0.019195156676365066, 0.014236333899824661, 0.019578295223714896, 0.025014428016724065, 0.02032100402565394, 0.026638863608938305, 0.03630411823945728, 0.02905046349658962, 0.0247702550563865, 0.013421182987465093, 0.019594189900178752, 0.009960898934023193, 0.0]]
# yerr_ea_loss1 += [[0.0, 0.0017420439536039302, 0.0029415345396316193, 0.007147753771632083, 0.005778298588183379, 0.009994300024824121, 0.008848309404467163, 0.013715805473285277, 0.013925533885018195, 0.015674429694612985, 0.01324859410913445, 0.018769029529252034, 0.013349791966127953, 0.011702988574271404, 0.03150109653010247, 0.025379522111263788, 0.026080065096160693, 0.01758126474170829, 0.014596163094133774, 0.00866533841732982, 0.0]]

y_ea_loss1 = [[0.9372004792332268, 0.9349400958466456, 0.927969249201278, 0.9149560702875398, 0.904832268370607, 0.8796026357827476, 0.8590814696485622, 0.8305551118210861, 0.8020706869009583, 0.7640994408945687, 0.720357428115016, 0.6860662939297124, 0.6370587060702875, 0.579341054313099, 0.5263198881789137, 0.4674241214057509, 0.405994408945687, 0.324642571884984, 0.25036940894568693, 0.17805910543130993, 0.09999999999999999], [0.9411940894568691, 0.9015814696485623, 0.8597803514376997, 0.812220447284345, 0.7869668530351437, 0.726361821086262, 0.6897623801916932, 0.6501537539936103, 0.6053434504792331, 0.564375, 0.528336661341853, 0.4635243610223642, 0.4434365015974441, 0.3932867412140575, 0.355323482428115, 0.31109225239616617, 0.26583865814696483, 0.22762779552715656, 0.18087859424920133, 0.1402276357827476, 0.09999999999999999], [0.9411940894568691, 0.9391234025559105, 0.9325718849840255, 0.919598642172524, 0.9102735623003196, 0.8848043130990415, 0.8654712460063898, 0.8355171725239616, 0.808001198083067, 0.7687819488817892, 0.7261880990415335, 0.69066892971246, 0.6418310702875399, 0.583694089456869, 0.5295047923322682, 0.47050918530351443, 0.40873003194888174, 0.3267691693290735, 0.25172723642172523, 0.17909744408945688, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.940648961661342, 0.9390235623003195, 0.9343949680511182, 0.9271505591054312, 0.919005591054313, 0.9051297923322684, 0.8895127795527158, 0.862991214057508, 0.8287699680511181, 0.8002595846645368, 0.760211661341853, 0.7087120607028754, 0.6490255591054314, 0.5876417731629393, 0.5132747603833867, 0.41774960063897765, 0.3294948083067093, 0.21179113418530351, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9400838658146965, 0.9381809105431309, 0.9341573482428114, 0.9300738817891375, 0.918905750798722, 0.9071924920127795, 0.8839916134185302, 0.864508785942492, 0.8296825079872203, 0.7897004792332268, 0.7425878594249202, 0.6746345846645367, 0.5980670926517571, 0.498827875399361, 0.3901477635782748, 0.24811301916932912, 0.09999999999999999]]
yerr_ea_loss1 = [[0.0, 0.0012209218253451436, 0.004618166467854866, 0.006070622521024615, 0.011319792899665919, 0.011821421025665253, 0.011952238246100607, 0.019598509017823486, 0.013352760003932386, 0.01700600840312484, 0.016769901768203702, 0.02303654405276731, 0.01673016808589782, 0.01830294090956483, 0.026116362760210142, 0.02155205362959434, 0.023779096515221927, 0.026008198321542438, 0.013325802002358557, 0.014912086029152217, 0.0], [1.1102230246251565e-16, 0.013735754582465409, 0.009284034940147332, 0.012193823579507164, 0.014693734468020584, 0.018622979540834184, 0.01578682014145918, 0.026620882804989733, 0.011650638989682376, 0.022739936173635046, 0.019406857249194998, 0.021124240915792465, 0.02021131778666067, 0.0195188752724254, 0.023881181740275065, 0.01957010054332116, 0.02548691695112473, 0.015513012233975927, 0.0138330760626493, 0.009425064810445236, 0.0], [1.1102230246251565e-16, 0.0010415229037902445, 0.0037130890385951563, 0.005545996341324924, 0.011436019458941385, 0.013111266260471354, 0.012012517336416608, 0.019408791714538035, 0.013095863457929551, 0.015571670082230496, 0.016550547205523538, 0.02300119114911305, 0.016487735887158544, 0.019620191243434722, 0.026668438599553177, 0.022114779242004386, 0.024106954007657628, 0.02575626767787884, 0.014107866190015141, 0.014923194307950511, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 0.001635383386581468, 0.002067152408449825, 0.004201623870433677, 0.004774644959440975, 0.009888971636172594, 0.011814404014580417, 0.008985138432088586, 0.013141214595400673, 0.01632652090028387, 0.019926028065283152, 0.02092079159766647, 0.020268827079122705, 0.02669945550229168, 0.018238983270859058, 0.022656898237435993, 0.03245984236415703, 0.01807184845179222, 0.019021970993666183, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0013604736009145732, 0.0028175825822148496, 0.004252418594502133, 0.0055395883345828794, 0.005717573806361869, 0.008058902581855184, 0.010291743723996003, 0.017581605828474366, 0.01410858121224397, 0.013917324240809497, 0.021265143453742707, 0.014406306236621638, 0.019803353932886996, 0.03665826065109268, 0.021742556467700975, 0.023548823530539654, 0.0]]
y_ea_loss1 += [[0.9240215654952075, 0.9221685303514378, 0.9176297923322684, 0.9032008785942491, 0.890133785942492, 0.871317891373802, 0.8455511182108626, 0.8198083067092652, 0.7921325878594249, 0.7542232428115014, 0.7122983226837062, 0.6721865015974441, 0.6265874600638976, 0.56435303514377, 0.5186761182108626, 0.4488797923322683, 0.3903873801916933, 0.3379892172523961, 0.25507388178913737, 0.17492012779552715, 0.09999999999999999], [0.9410942492012779, 0.8979053514376997, 0.8621485623003196, 0.8051038338658147, 0.7715874600638977, 0.7199860223642172, 0.679674520766773, 0.6484524760383387, 0.6065994408945686, 0.5680650958466453, 0.5141873003194888, 0.48357228434504795, 0.441060303514377, 0.3760642971246007, 0.35793929712460065, 0.3035802715654953, 0.273404552715655, 0.2270127795527157, 0.18977835463258785, 0.13502196485623003, 0.09999999999999999], [0.9410942492012779, 0.9392412140575079, 0.9346725239616613, 0.9200139776357827, 0.9065774760383387, 0.8873422523961662, 0.861116214057508, 0.8343550319488818, 0.8063997603833867, 0.7675419329073483, 0.7253174920127795, 0.6841273961661342, 0.6379692492012781, 0.57316892971246, 0.5274720447284345, 0.4565375399361023, 0.3966373801916933, 0.3435403354632588, 0.258158945686901, 0.1762679712460064, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9408166932907349, 0.9377935303514378, 0.934642571884984, 0.9265455271565495, 0.9186461661341851, 0.9047523961661342, 0.8911441693290737, 0.8628075079872204, 0.8266813099041533, 0.7998222843450478, 0.753781948881789, 0.709776357827476, 0.6551178115015974, 0.5700499201277955, 0.5045826677316294, 0.4292052715654952, 0.32627396166134187, 0.21265974440894575, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408067092651757, 0.9400139776357828, 0.9370607028753994, 0.9343350638977637, 0.92489017571885, 0.921084265175719, 0.904900159744409, 0.8855571086261982, 0.8604632587859425, 0.8241932907348243, 0.7916453674121404, 0.7443430511182109, 0.6589456869009583, 0.5872783546325879, 0.5067751597444089, 0.38067092651757195, 0.24651357827476036, 0.09999999999999999]]
yerr_ea_loss1 += [[1.1102230246251565e-16, 0.0023581718862374893, 0.0038870718377990803, 0.004887215082915404, 0.008904543903450591, 0.008316832083113334, 0.009424090063355532, 0.015256226150572708, 0.011989240924566902, 0.017701503876083168, 0.019331227869699586, 0.022302822953559762, 0.018304864501536556, 0.011884167427673177, 0.018713694981652745, 0.024798331366164963, 0.025218421991375674, 0.014123545970564394, 0.016886057604773488, 0.015347041917350004, 0.0], [0.0, 0.010835785611578722, 0.008990605647283722, 0.018440668075908568, 0.016939778095661513, 0.013707878609241962, 0.022329366262165317, 0.019663523003959436, 0.020538794643574994, 0.023560647059154016, 0.021910983982137148, 0.022566647267381435, 0.021144344701729337, 0.02213337015579413, 0.01736814199130211, 0.020446038351394802, 0.01794146105163237, 0.016239110843838687, 0.01238925274538145, 0.010688662754975663, 0.0], [0.0, 0.002359017141882499, 0.003719648648055333, 0.005014096917823351, 0.008643860969630934, 0.009077171417225494, 0.009726023654086204, 0.015367382172901591, 0.011941810292443688, 0.019008985298931642, 0.020302316906675527, 0.022230833070708327, 0.018517338048706106, 0.013282990298088833, 0.020069110878019224, 0.025188599445656042, 0.025961701383588456, 0.014503754445782606, 0.017373152814627844, 0.015376815860438351, 0.0], [0.0, 0.0, 0.0008326677316293772, 0.0032091590471020506, 0.0033570855030668253, 0.00687847743557408, 0.005303070791045427, 0.010280211314425018, 0.009309439725262473, 0.012435579619092972, 0.019483268192567283, 0.013175308255233392, 0.022192715515843314, 0.014885547161950383, 0.018974518144947636, 0.023204003429355403, 0.02354491366220885, 0.018578245572323578, 0.024266905279544823, 0.01793266625721826, 0.0], [0.0, 0.0, 0.0, 0.0008626198083066973, 0.00248067897724626, 0.0035099153896821747, 0.0038753995462165887, 0.00614417452063475, 0.004729684450891118, 0.00737721641581355, 0.01028258102885954, 0.012163346204267629, 0.016802286730765725, 0.01453841207088222, 0.0125767436459116, 0.022481471773432765, 0.02389855649763157, 0.018672504696556326, 0.025139102288288192, 0.015718197922629004, 0.0]]
x = [[0.05*i for i in range(n_hard)] for _ in range(4)]
y = np.array(y_ea_loss1)*100;yerr = np.array(yerr_ea_loss1)*100
y = y[[2,1,5,0],:n_hard];yerr = yerr[[2,1,5,0],:n_hard]
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/FCC_loss_ea.eps',
		'Failure Rate (%)','Effective Accuracy (%)',lbsize=24,linewidth=8,markersize=0,linestyles=linestyles_tmp,yerr=yerr)

# x0 = np.array([[0.0625*2*i for i in range(1,5)] for _ in range(2)])
# x = 100-x0*100
# methods_tmp = [f'ResNet{v}' for v in [56,50]]
# y = np.array([2.03125/0.03125,40.099609375/.729736328125]).reshape(2,1)
# y = np.repeat(y,4,axis=1)
# y = y / (x0 * 2)
# line_plot(x,y,methods_tmp,colors,
# 		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/comm_cost.eps',
# 		'Partition Ratio (%)','Comm. Cost Reduction Ratio',lbsize=24,linewidth=8,markersize=16,linestyles=linestyles,
# 		use_prob_annot=True)

x0 = np.array([0.1*i for i in range(11)])
x = [(1-x0)*100]
y = [(1-x0**2-(1-x0)**2)*100]
line_plot(x,y,[''],colors,
				f'/home/bo/Dropbox/Research/NSDI24fFaultless/images/conn_loss.eps',
				'Partition Ratio (%)','Connection Loss (%)',lbsize=36,linewidth=8,markersize=16,ncol=0,use_connarrow=True)

# y2 = np.concatenate((re_base[:,1].reshape((1,8)),
# 					re_nobridge[:,1].reshape((1,8)),
# 					re_noonn[:,1].reshape((1,8)),
# 					re_nolab[:,1].reshape((1,8))))*100
# line_plot(x,y2,methods_tmp,colors,
# 		'/home/bo/Dropbox/Research/NSDI24fFaultless/images/ablation_hard.eps',
# 		'FLOPS (%)','Reliability (%)',lbsize=28,linewidth=0,markersize=16,linestyles=linestyles,
# 		lgsize=20,use_arrow=True,arrow_coord=(80,6))	
# ratio of latency
# y = [[0.03125, 2.03125, 32*32*3*4/1024/1024],
# 	[0.729736328125,40.099609375,224*224*3*4/1024/1024 ]]


# a = [0.9345047923322684, 0.9362020766773163, 0.9336062300319489, 0.9326078274760383, 0.9308107028753994, 0.9269169329073482, 0.9281150159744409, 0.9285143769968051]
# a = [0.9396964856230032, 0.9386980830670927, 0.9401956869009584, 0.9352036741214057, 0.9362020766773163, 0.9343051118210862]


# # baseline
# methods_tmp = ['Standalone','Ours']
# y = [[0.21911602480000006,0.006703817600000122],[0.17067866719999998,0.003996000000000066]]
# yerr = [[0.0019785233779630296,0.0009860426520709135],[0.0011066291237713699,0.0002033875583595318]]
# y,yerr = np.array(y),np.array(yerr)
# groupedbar(y,yerr,'$R^{(2)}$', 
# 	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/re_imagenet.eps',methods=methods_tmp,
# 	envs=['Soft','Hard'],ncol=1,sep=.3,width=0.1,legloc='best',use_barlabe_y=True)


# # flop distribution
# flops = [
# 	[100, 0],
# 	[100,100],
# 	[86.77293603320655]*2,
# 	[70.53025247728113]*2,
# 	[63.75213554749708]*2,
# 	[60.03251876612296]*2,
# 	[53.60070051652971]*2
# 	]
# labels = ['Standalone','Replication','Ours\n(ResNet-20)','Ours\n(ResNet-32)','Ours\n(ResNet-44)','Ours\n(ResNet-56)','Ours\n(ResNet-110)']
# filename = '/home/bo/Dropbox/Research/NSDI24fFaultless/images/flops_dist.eps'
# plot_computation_dist(flops,labels,filename,horizontal=False)
# # 77.09%: resnet-50


# different sampling rates
re_vs_sr = [[0.00297523961661339,0.0031948881789138905],
[0.0004992012779553301,0.0019968051118210983],
[0.0015974440894568342,0.004293130990415284],
[0.0026956869009584494,0.0019968051118210983],
 [0.004592651757188482,0.003993610223642197],]

flops_vs_sr = [0.7930,0.6614,0.5956,0.5627,0.5298]
sample_interval = [9,5,3,2,1]
re_vs_sr = np.array(re_vs_sr)
y = re_vs_sr*100
envs = [f'{sample_interval[i]}/9\n{round(flops_vs_sr[i]*200)}' for i in range(5)]
groupedbar(y,None,'Consistency (%)', 
	'/home/bo/Dropbox/Research/NSDI24fFaultless/images/re_vs_sr.eps',methods=['Jitter','Fail-stop'],envs=envs,labelsize=24,
	ncol=1,sep=1,width=0.4,xlabel='Sampling Interval and FLOPS (%)',legloc=None,bbox_to_anchor=(0.25, 1.03))

exit(0)