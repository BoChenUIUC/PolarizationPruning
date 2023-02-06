#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
labelsize_b = 14
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = ['#DB1F48','#1C4670','#FF9636','#9D5FFB','#21B6A8','#D65780']
# colors = ['#ED4974','#16B9E1','#58DE7B','#F0D864','#FF8057','#8958D3']
# colors =['#FD0707','#0D0DDF','#129114','#DDDB03','#FF8A12','#8402AD']
markers = ['o','^','s','>','P','D','*','v','<'] 
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
linestyles = ['solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
methods3 = ['Replication (Optimal)','Partition (Ours)','Standalone']
methods6 = ['Ours','Baseline','Optimal$^{(2)}$','Ours*','Baseline*','Optimal$^{(2)}$*']
from collections import OrderedDict
linestyle_dict = OrderedDict(
    [('solid',               (0, ())),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
linestyles = []
for i, (name, linestyle) in enumerate(linestyle_dict.items()):
    if i >= 9:break
    linestyles += [linestyle]


def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,legloc='best',linestyles=linestyles,
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(60,0.6),markersize=8,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,use_probarrow=False,
				rotation=None,use_resnet56_2arrow=False,use_resnet56_3arrow=False,use_resnet56_4arrow=False,use_resnet50arrow=False,use_re_label=False,
				use_comm_annot=False,use_connarrow=False,lgsize=None):
	if lgsize is None:
		lgsize = lbsize
	if get_ax==1:
		ax = plt.subplot(211)
	elif get_ax==2:
		ax = plt.subplot(212)
	else:
		fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if logx:
			xx = np.log10(np.array(xx))
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], 
				linestyle = linestyles[i], 
				label = label[i], 
				linewidth=linewidth, markersize=markersize)
		else:
			if markersize > 0:
				plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
					marker = markers[i], label = label[i], 
					linestyle = linestyles[i], 
					linewidth=linewidth, markersize=markersize,
					capsize=4)
			else:
				plt.errorbar(xx, yy, yerr=yerr[i], color = color[i],
					label = label[i], 
					linestyle = linestyles[i], 
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
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=45, size=lbsize-8,
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
			parlocs += [np.argmax(y[i]-y[4])]
		for k,locs in enumerate([baselocs,parlocs]):
			for i,loc in enumerate(locs):
				ax.annotate(text='', xy=(XX[0][loc],YY[0 if k==0 else 4,loc]), xytext=(XX[0][loc],YY[i+1,loc]), arrowprops=dict(arrowstyle='|-|',lw=5-k*2,color=color[k]))
				h = YY[k,loc]-5 if k==0 else YY[i+1,loc]+4
				w = XX[0][loc]-3 if k==0 else XX[0][loc]
				if k==1 and i>0:h+=5
				if i==0:
					ax.text(w, h, '2nd', ha="center", va="center", rotation='horizontal', size=16,fontweight='bold',color=color[k])
				elif i==1:
					ax.text(w, h, '3rd', ha="center", va="center", rotation='horizontal', size=16,fontweight='bold',color=color[k])
				elif i==2:
					ax.text(w, h, '4th', ha="center", va="center", rotation='horizontal', size=16,fontweight='bold',color=color[k])
	if use_comm_annot:
		for i in range(len(XX)):
			xx,yy = XX[i],YY[i]
			for a,b in zip(xx,yy):
				if i==0:
					if a<60:
						ax.annotate(f'{2.03125*1024/b:.1f}K', (a-1, b+20), fontsize=lbsize)
					elif a<70:
						ax.annotate(f'{2.03125*1024/b:.1f}K', (a-5, b+20), fontsize=lbsize)
					elif a<80:
						ax.annotate(f'{2.03125*1024/b:.1f}K', (a-7, b+15), fontsize=lbsize)
					else:
						ax.annotate(f'{2.03125*1024/b:.1f}K', (a-7, b-5), fontsize=lbsize)
				else:
					if a<60:
						ax.annotate(f'{40.099609375/b:.2f}M', (a+2, b-10), fontsize=lbsize)
					elif a<70:
						ax.annotate(f'{40.099609375/b:.2f}M', (a-1, b-20), fontsize=lbsize)
					elif a<80:
						ax.annotate(f'{40.099609375/b:.2f}M', (a-2, b-25), fontsize=lbsize)
					else:
						ax.annotate(f'{40.099609375/b:.2f}M', (a-6, b-60), fontsize=lbsize)
	if use_resnet56_2arrow:
		lgsize=18
		ax.annotate(text='', xy=(10,78), xytext=(60,78), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    35, 79, "Stage#0", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(60,87.5), xytext=(200,87.5), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    130, 88.5, "Stage#1", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		# for l,r,lr in [(10,60,0.1),(60,120,0.02),(120,160,0.004),(160,200,0.0008)]:
		# 	if l==160:
		# 		h = 87.5
		# 	elif l==120:
		# 		h = 85
		# 	else:
		# 		h = 81
		# 	ax.annotate(text='', xy=(l,h), xytext=(r,h), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		# 	ax.text(
		# 	    (l+r)/2, h+1, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if use_resnet56_3arrow:
		lgsize=18
		ax.annotate(text='', xy=(10,75), xytext=(60,75), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    35, 76, "Stage#0", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(60,87.5), xytext=(200,87.5), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    130, 89, "Stage#1", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		# for l,r,lr in [(10,60,0.1),(60,120,0.02),(120,160,0.004),(160,200,0.0008)]:
		# 	h = 78 if l<160 else 87
		# 	ax.annotate(text='', xy=(l,h), xytext=(r,h), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		# 	ax.text(
		# 	    (l+r)/2, h+1, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if use_resnet56_4arrow:
		lgsize=18
		ax.annotate(text='', xy=(10,73), xytext=(60,73), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    35, 74.5, "Stage#0", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(60,85), xytext=(200,85), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    130, 86.5, "Stage#1", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		# for l,r,lr in [(10,60,0.1),(60,120,0.02),(120,160,0.004),(160,200,0.0008)]:
		# 	h = 77 if l<160 else 87
		# 	ax.annotate(text='', xy=(l,h), xytext=(r,h), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		# 	ax.text(
		# 	    (l+r)/2, h+1, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if use_resnet50arrow:
		lgsize=18
		ax.annotate(text='', xy=(0,62), xytext=(40,62), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    20, 63, "Stage#0", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(40,68), xytext=(160,68), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    100, 69, "Stage#1", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
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
	y_pos = np.arange(len(labels))
	latency_breakdown_mean = np.array(latency_breakdown_mean).transpose((1,0))*1000
	latency_breakdown_std = np.array(latency_breakdown_std).transpose((1,0))*1000
	width = 0.5
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
	fig.legend(handles, labels,bbox_to_anchor=bbox_to_anchor,ncol=ncol,fancybox=True, loc='upper center', fontsize=14)
	ax1.invert_yaxis()  
	fig.text(0.5, title_posy, 'Latency breakdown (ms)', fontsize = 14, ha='center')
	for ax in [ax1,ax2]:
		xleft, xright = ax.get_xlim()
		ybottom, ytop = ax.get_ylim()
		ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
	plt.tight_layout()
	fig.savefig(filename,bbox_inches='tight')

def measurements_to_cdf(latency,epsfile,labels,xticks=None,xticklabel=None,linestyles=linestyles,colors=colors,
						xlabel='Query Latency (s)',ylabel='CDF',ratio=None,lbsize = 18,linewidth=4):
    # plot cdf
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    for i,latency_list in enumerate(latency):
        N = len(latency_list)
        cdf_x = np.sort(np.array(latency_list))*100
        cdf_p = np.array(range(N))/float(N)
        plt.plot(cdf_x, cdf_p, color = colors[i], label = labels[i], linewidth=linewidth, linestyle=linestyles[i])
    plt.xlabel(xlabel, fontsize = lbsize)
    plt.ylabel(ylabel, fontsize = lbsize)
    if xticks is not None:
        plt.xticks(xticks,fontsize=lbsize)
    if xticklabel is not None:
        ax.set_xticklabels(xticklabel)
    if ratio is not None:
	    xleft, xright = ax.get_xlim()
	    ybottom, ytop = ax.get_ylim()
	    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    plt.legend(loc='best',fontsize = lbsize)
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
        # read network traces 
        csv_reader = csv.DictReader(csv_file)
        latency_list = []
        latency224_list = []
        num_of_line = 0
        bandwidth_list = []
        for row in csv_reader:
            if float(row["latency"])>1e7 or float(row["latency"])<0 or float(row["downthrpt"])>1e8:
                continue
            bandwidth_list += [float(row["downthrpt"])]
            for bs in [2**i for i in range(7)]:
                query_size = 3*32*32*4*bs # bytes
                latency_list += [query_size/float(row["downthrpt"]) + float(row["latency"])/1e6]
                query_size = 3*224*224*4*bs
                latency224_list += [query_size/float(row["downthrpt"]) + float(row["latency"])/1e6]
            num_of_line += 1
            if num_of_line==10000:break
        all_latency_list += np.array(latency_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
        all_latency_list += np.array(latency224_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
    all_latency_list = np.array(all_latency_list)
    all_latency_list = all_latency_list.mean(axis=-1).reshape(4,7)
    query_size = 3*32*32*4*np.array([2**(i) for i in range(7)])
    bw = query_size/all_latency_list[0]/1e6*8
    print(bw.mean(),bw.std(),'MBps',np.array(bandwidth_list).mean()*8,np.array(bandwidth_list).std()*8)
    # labels = ['DCN CIFAR-10']#,'DCN ImageNet','WAN CIFAR-10','WAN ImageNet']
    # x = [[2**(i) for i in range(7)] for _ in range(len(labels))]
    # line_plot(x, all_latency_list[[1],],labels,colors,'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_all.eps','Batch Size','Latency',
    # 	lbsize=24,linewidth=4,markersize=8,)
    ratio = all_latency_list[2:,]/all_latency_list[:2]
    # relative_latency = all_latency_list[14:,:].copy() #28x10000
    # for b in range(7):
    # 	for start in [0,14]:
    # 		relative_latency[start+b] /= all_latency_list[b]
    # 	for start in [7,21]:
    # 		relative_latency[start+b] /= all_latency_list[7+b]
    # relative_latency = np.log10(relative_latency)
    # relative_latency = relative_latency.reshape((4,7,10000))
    # relative_latency = relative_latency[2:]
    labels = ['ResNet56','ResNet50']
    x = [[2**(i) for i in range(7)] for _ in range(len(labels))]
    line_plot(x, ratio,labels,colors,'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_cost.eps','Batch Size','WAN to DCN Latency Ratio',
    	lbsize=24,linewidth=8,markersize=16,)	

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
				use_downarrow=False,rotation=None,lgsize=None):
	if lgsize is None:
		lgsize = labelsize
	fig = plt.figure()
	ax = fig.add_subplot(111)
	num_methods = data_mean.shape[1]
	num_env = data_mean.shape[0]
	center_index = np.arange(1, num_env + 1)*sep
	# colors = ['lightcoral', 'orange', 'yellow', 'palegreen', 'lightskyblue']
	# colors = ['coral', 'orange', 'green', 'cyan', 'blue']

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
				ax.text(xdx-0.08,data_mean[k,i]+1,f'{data_mean[k,i]:.4f}',fontsize = 18, rotation='vertical',fontweight='bold')
		if use_downarrow:
			if i==1:
				for j in range(2,data_mean.shape[0]):
					ax.annotate(text='', xy=(x_index[j],data_mean[j,i]), xytext=(x_index[j],200), arrowprops=dict(arrowstyle='<->',lw=4))
					ax.text(x_index[j]-0.04, 160, '$\downarrow$'+f'{200-data_mean[j,i]:.0f}%', ha="center", va="center", rotation='vertical', size=labelsize ,fontweight='bold')
					# ax.text(center_index[j]-0.02,data_mean[j,i]+5,'$\downarrow$'+f'{200-data_mean[j,i]:.0f}%',fontsize = 16, fontweight='bold')
			else:
				for k,xdx in enumerate(x_index):
					ax.text(xdx-0.07,data_mean[k,i]+5,f'{data_mean[k,i]:.2f}',fontsize = labelsize,fontweight='bold')

	if ncol>0:
		if legloc is None:
			plt.legend(bbox_to_anchor=bbox_to_anchor, fancybox=True,
			           loc='upper center', ncol=ncol, fontsize=lgsize)
		else:
			plt.legend(fancybox=True,
			           loc=legloc, ncol=ncol, fontsize=lgsize)
	fig.savefig(path, bbox_inches='tight')
	plt.close()
# deadlines to jitter level: 0-same as mean ddl, 1-0.1xddl, -1-10xddl
# reliability to consistency


# multi-node analysis
methods = ['Base','CR','Ours']

re_vs_nodes = [[0.18546325878594247,0.20588857827476037,0.01707268370607029,0.01707268370607029,0.01193889776357826,0.006509584664536749],
[0.266172124600639, 0.3088937699680511,0.030001996805111824,0.02982846493229878,0.0027975239616613414,0.002635782747603843],
[0.29221645367412136,0.3563059105431311,0.030925519169329063,0.03228168264110755,0.007637779552715651,0.006968849840255598]]

y = np.array(re_vs_nodes)*100
for i in range(2):
	y1 = y[:,[0+i,2+i,4+i]]
	groupedbar(y1,None,'Reliability (%)', 
		f'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_nodes{i}.eps',methods=methods,labelsize=28,xlabel='Cluster Size',
		envs=[2,3,4],sep=1,width=0.3,bbox_to_anchor=(0.5, 1.25),use_barlabel_x=True,legloc='upper left',ncol=1,lgsize=20)


n_soft = 12;n_hard = 11;
# soft
y = [[0.09999999999999999, 0.10054512779552716, 0.10054512779552716, 0.11902755591054312, 0.24644369009584666, 0.4110902555910543, 0.5672503993610223, 0.6601238019169329, 0.7321505591054313, 0.793057108626198, 0.8280910543130989, 0.8515195686900958, 0.8795467252396165, 0.8962060702875398, 0.903905750798722, 0.9113977635782747, 0.915401357827476, 0.9167292332268369, 0.9250139776357826, 0.9292152555910542], [0.09999999999999999, 0.10026757188498403, 0.10026757188498403, 0.12053314696485622, 0.19403953674121407, 0.30107228434504796, 0.40742012779552716, 0.4781529552715654, 0.5490015974440895, 0.6168170926517572, 0.6582428115015976, 0.6940595047923324, 0.7466353833865815, 0.7764436900958467, 0.8043510383386583, 0.8349560702875399, 0.8493550319488816, 0.8560443290734824, 0.8767372204472844, 0.8892432108626197], [0.09999999999999999, 0.10055511182108626, 0.10055511182108626, 0.12967052715654953, 0.265960463258786, 0.43768769968051113, 0.5874161341853036, 0.6758865814696484, 0.7458226837060702, 0.8056110223642173, 0.838514376996805, 0.8573542332268371, 0.8893550319488819, 0.9001757188498403, 0.9084305111821086, 0.9165774760383385, 0.9192432108626198, 0.9202935303514378, 0.9301637380191693, 0.9328494408945687]]
y += [[0.09999999999999999, 0.10053514376996806, 0.10266573482428117, 0.14354432907348244, 0.3245746805111821, 0.5350858626198083, 0.7037440095846644, 0.7912959265175717, 0.8449900159744409, 0.885880591054313, 0.9049580670926517, 0.9186940894568689, 0.9284964057507986, 0.933292731629393, 0.9351956869009583, 0.9370287539936102, 0.937821485623003, 0.938356629392971, 0.9386242012779551, 0.9404373003194888], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12064297124600636, 0.19427915335463258, 0.3012919329073483, 0.40771964856230036, 0.478013178913738, 0.5486142172523961, 0.6166074281150159, 0.6581130191693292, 0.6936900958466452, 0.7458466453674121, 0.7759944089456868, 0.8038019169329074, 0.8345666932907347, 0.8490255591054312, 0.8558346645367412, 0.8764576677316294, 0.8892332268370605], [0.09999999999999999, 0.10055511182108626, 0.10348841853035144, 0.16188897763578275, 0.35286341853035147, 0.5664017571884984, 0.7198242811501598, 0.8020607028753993, 0.8510183706070287, 0.8899960063897764, 0.9071705271565496, 0.9196485623003194, 0.9303614217252397, 0.933849840255591, 0.9360303514376996, 0.9376257987220447, 0.938180910543131, 0.9387060702875398, 0.9400139776357828, 0.9408266773162939]]
y += [[0.09999999999999999, 0.10080271565495207, 0.1032308306709265, 0.16486222044728435, 0.39264776357827474, 0.6277076677316293, 0.7823861821086261, 0.8479812300319489, 0.8836880990415334, 0.9090674920127796, 0.9193051118210864, 0.9266773162939298, 0.931898961661342, 0.9332168530351439, 0.9335043929712461, 0.9340395367412141, 0.9340395367412141, 0.9343071086261983, 0.9343071086261983, 0.9345746805111823], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12060303514376995, 0.19419928115015977, 0.3015595047923323, 0.40726038338658144, 0.47754392971246007, 0.5476477635782747, 0.6165375399361024, 0.6580131789137379, 0.6937100638977636, 0.7460163738019169, 0.7760842651757188, 0.8038119009584663, 0.8344868210862619, 0.8490155750798722, 0.855804712460064, 0.8766074281150159, 0.8892831469648561], [0.09999999999999999, 0.10080271565495207, 0.10588658146964855, 0.18738019169329073, 0.42757787539936104, 0.6566253993610223, 0.7983146964856228, 0.8612539936102236, 0.8946825079872205, 0.9179932108626199, 0.9266553514376996, 0.9332747603833864, 0.9383785942492013, 0.9397464057507987, 0.9400139776357828, 0.9405690894568689, 0.9405690894568689, 0.940836661341853, 0.940836661341853, 0.9410942492012779]]
y += [[0.09999999999999999, 0.10049520766773161, 0.10049520766773161, 0.12854233226837058, 0.258404552715655, 0.4213158945686901, 0.5634784345047923, 0.64819089456869, 0.7146345846645367, 0.7717452076677316, 0.8027835463258786, 0.8208426517571885, 0.8512160543130991, 0.8616873003194888, 0.8696825079872204, 0.8772404153354632, 0.8797963258785944, 0.8807268370607029, 0.8902276357827477, 0.8928234824281148], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12091253993610222, 0.19401956869009584, 0.3010043929712459, 0.40735223642172524, 0.47756389776357827, 0.5477595846645367, 0.6165974440894568, 0.6576956869009584, 0.6936501597444089, 0.7457288338658147, 0.7760742811501598, 0.8039516773162939, 0.8343570287539934, 0.8490155750798722, 0.8557847444089456, 0.8763777955271566, 0.8890535143769969], [0.09999999999999999, 0.10056509584664537, 0.10056509584664537, 0.13004992012779554, 0.2652036741214058, 0.4373702076677316, 0.5861641373801916, 0.6750878594249201, 0.7450958466453674, 0.8054612619808307, 0.8379972044728434, 0.8570846645367413, 0.8888658146964856, 0.8999361022364216, 0.9082308306709266, 0.9163378594249201, 0.9190734824281149, 0.9201837060702873, 0.9301238019169329, 0.9328095047923324], [0.09999999999999999, 0.10056509584664537, 0.10349840255591054, 0.16235822683706075, 0.3523761980830671, 0.5660962460063897, 0.7197863418530351, 0.8020107827476037, 0.8511281948881789, 0.8903055111821085, 0.9075499201277955, 0.9197484025559104, 0.9304113418530351, 0.9338398562300319, 0.9358905750798721, 0.9375559105431309, 0.9381010383386581, 0.9386561501597445, 0.9400039936102236, 0.9408266773162939], [0.09999999999999999, 0.1008326677316294, 0.1059464856230032, 0.18746006389776357, 0.42733027156549525, 0.6557647763578276, 0.7980970447284343, 0.8611561501597444, 0.8950119808306709, 0.9182328274760383, 0.9267252396166136, 0.9334045527156549, 0.9384384984025559, 0.9397164536741214, 0.9400039936102236, 0.9405491214057508, 0.9405491214057508, 0.9408266773162939, 0.9408266773162939, 0.9410942492012779]]
y += [[0.09999999999999999, 0.10049520766773161, 0.10331869009584664, 0.15964057507987223, 0.3415495207667732, 0.5442951277955271, 0.6905211661341853, 0.7696106230031949, 0.8153753993610225, 0.8523861821086263, 0.8687819488817892, 0.8803813897763577, 0.8904952076677317, 0.8938538338658144, 0.8959444888178913, 0.8975199680511181, 0.8979752396166134, 0.8985003993610224, 0.8998382587859425, 0.9006210063897765], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12091253993610222, 0.19427715654952077, 0.3014896166134185, 0.40790734824281155, 0.4778414536741214, 0.5488598242811501, 0.6165974440894568, 0.6579532747603833, 0.6936501597444089, 0.7460063897763577, 0.7760742811501598, 0.8039516773162939, 0.8343570287539934, 0.8490155750798722, 0.8557847444089456, 0.8763777955271566, 0.8890535143769969], [0.09999999999999999, 0.10056509584664537, 0.10056509584664537, 0.13004992012779554, 0.2657388178913739, 0.4378853833865814, 0.587264376996805, 0.6756329872204473, 0.7453634185303514, 0.8054612619808307, 0.8382547923322685, 0.8570846645367413, 0.8888658146964856, 0.8999361022364216, 0.9082308306709266, 0.9163378594249201, 0.9190734824281149, 0.9201837060702873, 0.9301238019169329, 0.9328095047923324], [0.09999999999999999, 0.10056509584664537, 0.10349840255591054, 0.16262579872204475, 0.35291134185303513, 0.5668989616613419, 0.7203514376996805, 0.802288338658147, 0.8511281948881789, 0.8903055111821085, 0.9075499201277955, 0.9197484025559104, 0.9304113418530351, 0.9338398562300319, 0.9358905750798721, 0.9375559105431309, 0.9381010383386581, 0.9386561501597445, 0.9400039936102236, 0.9408266773162939], [0.09999999999999999, 0.1008326677316294, 0.1059464856230032, 0.18772763578274762, 0.42786541533546335, 0.6568450479233227, 0.7983746006389776, 0.8616912939297124, 0.8950119808306709, 0.9182328274760383, 0.9267252396166136, 0.9334045527156549, 0.9384384984025559, 0.9397164536741214, 0.9400039936102236, 0.9405491214057508, 0.9405491214057508, 0.9408266773162939, 0.9408266773162939, 0.9410942492012779]]
y += [[0.09999999999999999, 0.10077276357827476, 0.10567691693290733, 0.18321685303514376, 0.41184504792332266, 0.6296625399361022, 0.7647184504792331, 0.8249640575079871, 0.8566533546325878, 0.8787460063897763, 0.8870686900958468, 0.8933985623003196, 0.8982428115015975, 0.8995706869009584, 0.8998582268370608, 0.9003534345047924, 0.9003534345047924, 0.9006210063897765, 0.9006210063897765, 0.9008586261980831], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12091253993610222, 0.19401956869009584, 0.3010043929712459, 0.40762979233226837, 0.47756389776357827, 0.5477595846645367, 0.6165974440894568, 0.6576956869009584, 0.6936501597444089, 0.7457288338658147, 0.7760742811501598, 0.8039516773162939, 0.8343570287539934, 0.8490155750798722, 0.8557847444089456, 0.8763777955271566, 0.8890535143769969], [0.09999999999999999, 0.10056509584664537, 0.10056509584664537, 0.13004992012779554, 0.2652036741214058, 0.4376377795527156, 0.5864416932907348, 0.6750878594249201, 0.7450958466453674, 0.8054612619808307, 0.8379972044728434, 0.8570846645367413, 0.8888658146964856, 0.8999361022364216, 0.9082308306709266, 0.9163378594249201, 0.9190734824281149, 0.9201837060702873, 0.9301238019169329, 0.9328095047923324], [0.09999999999999999, 0.10056509584664537, 0.10349840255591054, 0.16235822683706075, 0.3523761980830671, 0.5663638178913738, 0.7200638977635783, 0.8020107827476037, 0.8511281948881789, 0.8903055111821085, 0.9075499201277955, 0.9197484025559104, 0.9304113418530351, 0.9338398562300319, 0.9358905750798721, 0.9375559105431309, 0.9381010383386581, 0.9386561501597445, 0.9400039936102236, 0.9408266773162939], [0.09999999999999999, 0.1008326677316294, 0.1059464856230032, 0.18746006389776357, 0.42733027156549525, 0.6560323482428115, 0.7983746006389776, 0.8611561501597444, 0.8950119808306709, 0.9182328274760383, 0.9267252396166136, 0.9334045527156549, 0.9384384984025559, 0.9397164536741214, 0.9400039936102236, 0.9405491214057508, 0.9405491214057508, 0.9408266773162939, 0.9408266773162939, 0.9410942492012779]]

yerr = [[0.0, 0.0010904841391130044, 0.0010904841391130044, 0.012901423965101584, 0.05386465228666159, 0.06330897181706202, 0.061088301211427776, 0.042496575722181885, 0.03791339018992345, 0.0286516040794726, 0.024502583576222894, 0.025234458910697543, 0.025214803986917174, 0.02004981065222387, 0.01913403883501498, 0.0210146786784222, 0.020861977353891028, 0.020812112201925786, 0.012489402392714916, 0.009346223105772366], [0.0, 0.0008027156549520825, 0.0008027156549520825, 0.014877741975549278, 0.041380313427382555, 0.059803487736121326, 0.06442090733061269, 0.054010348189044745, 0.046256730645281255, 0.04990167454459851, 0.04680540980752168, 0.04270990091509234, 0.04092552622057239, 0.04201476212272375, 0.035258841989506476, 0.024606431299643745, 0.024426260082436837, 0.02257806012117962, 0.02338451126124495, 0.01813754932754337], [0.0, 0.0011111211234452062, 0.0011111211234452062, 0.020617885360086453, 0.058052933419270515, 0.06406899866962676, 0.06046125060499545, 0.04090110185496498, 0.03818100531319478, 0.022599839902736295, 0.024208484492046708, 0.02502698385338613, 0.021951503438406496, 0.0199137496482411, 0.019387677509606134, 0.020391848164711325, 0.020533313186606467, 0.020248210447870275, 0.01021375164308125, 0.008686921346931619]]
yerr += [[0.0, 0.0010740064615502457, 0.006303007931757033, 0.026021125914269503, 0.08105363023623796, 0.0740595744589342, 0.06041652501440283, 0.044844806014635245, 0.028175216533628814, 0.017087291384438583, 0.013970411329020677, 0.011378221653788169, 0.010262507752688744, 0.008127237448455074, 0.007248766702406512, 0.005366632081894418, 0.00540657578979667, 0.005105690103907452, 0.004519983947197429, 0.0021377652284912336], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.014884064755360492, 0.041080908076770895, 0.058943171934852476, 0.06350502758454735, 0.053835822922173385, 0.046569856977913715, 0.049974697051315166, 0.047266872346802026, 0.04321938603316445, 0.04093103410050893, 0.04192103684866083, 0.03560689885840395, 0.024849190569816677, 0.024534119292260462, 0.02273722235521622, 0.02336269319504597, 0.018269360880982136], [0.0, 0.0011102236421725309, 0.006982465449370012, 0.03891130443930724, 0.07933421510907138, 0.07312045790566846, 0.05709689397238947, 0.03976837160964229, 0.02660124404395353, 0.015912401995836385, 0.01412622823382116, 0.010690884125923806, 0.008377834943452267, 0.006884396773024983, 0.0056704638586413956, 0.004258329577667321, 0.004336842354278658, 0.0041518970931095625, 0.00178413056171682, 0.0008027156549520573]]
yerr += [[0.0, 0.0012269810926168812, 0.006283865420787238, 0.028908457380234936, 0.08261134500237968, 0.06686804182363132, 0.04642606964208399, 0.03309409330081189, 0.022616908893124406, 0.012616052522013305, 0.008800497129825013, 0.004936795253624663, 0.0031782620755874194, 0.0024480741091414157, 0.002122190531986667, 0.0021105776637082045, 0.0021105776637082045, 0.0020216199707253544, 0.0020216199707253544, 0.002237869831544598], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.015036159407656817, 0.04115502264840482, 0.0596680467179605, 0.06398301658478026, 0.05410036418702724, 0.04657518349967236, 0.049823079627498594, 0.04696471449231237, 0.04299295114084161, 0.04112233192725984, 0.042494933804224914, 0.03553520396804823, 0.024830376263057598, 0.024664085467536682, 0.022878465565610913, 0.0235558085394264, 0.01808878859085014], [0.0, 0.0012269810926168812, 0.00856012814586162, 0.04520614628870527, 0.08161380704950524, 0.06805698287006944, 0.04363282099972548, 0.02765925284340011, 0.022176663654976114, 0.013079873551852804, 0.009191775440817871, 0.004524296847070941, 0.002748419268439603, 0.0021768518736728507, 0.0021615809620174464, 0.0015754792332268283, 0.0015754792332268283, 0.0007727635782747377, 0.0007727635782747377, 0.0]]
yerr += [[0.0, 0.00099443300328881, 0.00099443300328881, 0.019505029461655107, 0.05546912225783641, 0.061383721766164565, 0.0574643397716346, 0.03903459742307307, 0.03629773575785051, 0.022477917272181132, 0.023993625844449366, 0.024283259850336362, 0.021449119587870866, 0.01941129579656337, 0.018824164775408433, 0.019649505483998873, 0.019729050513594238, 0.019467567337130353, 0.009863530877285642, 0.008319727257849869], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.01513887524044183, 0.04119654038237955, 0.05954546638401433, 0.06461615296695948, 0.05445139348180503, 0.04617975208048419, 0.04952314420243866, 0.04746079311920488, 0.042838602629238956, 0.04089750378103106, 0.04209583989387418, 0.035370235254640774, 0.024982943507328315, 0.02477251591667651, 0.022909695917641672, 0.02360574661841937, 0.01835577215644545], [0.0, 0.001130412167050239, 0.001130412167050239, 0.02103568830856657, 0.05839600284341096, 0.0650112767810241, 0.060285843666243046, 0.041517084973610306, 0.038488671250978046, 0.022750362682619894, 0.0246057511265708, 0.02527128623125571, 0.022230931267836387, 0.02010614945519215, 0.019476646673392013, 0.02054017399989052, 0.020645247609666455, 0.02033393971627395, 0.010214442589934217, 0.008632253164529886], [0.0, 0.001130412167050239, 0.007011344344321024, 0.038961003563398884, 0.07958987441632322, 0.07441194300440752, 0.05838402698512159, 0.040526909850463956, 0.02714396563180163, 0.016159296962269847, 0.014321724518215251, 0.011104814730380753, 0.008619308202797, 0.0071068089292355445, 0.005897422510638241, 0.004385838676700178, 0.0044524675130884735, 0.004281128273065248, 0.0017932866366198895, 0.0008027156549520573], [0.0, 0.001272704430868878, 0.008699297751227585, 0.04531487330252202, 0.08171083598552101, 0.06790145048903508, 0.04398753732184092, 0.028591489586635873, 0.022159353076599272, 0.01281426386832743, 0.008861164158924392, 0.004374557053211664, 0.0026586018470994037, 0.002207205904081021, 0.002180511182108624, 0.0016353833865814685, 0.0016353833865814685, 0.0008027156549520573, 0.0008027156549520573, 0.0]]
yerr += [[0.0, 0.00099443300328881, 0.006757454025780192, 0.03640583140682513, 0.07473007625923796, 0.07024874245195317, 0.0556731424759284, 0.03827971587494205, 0.02618127529097377, 0.015314308663752208, 0.013732216760749387, 0.010734324511094235, 0.00843836232008807, 0.00690859825716042, 0.00576579299957006, 0.0042702208378181256, 0.0043151803935578565, 0.0041710142544761665, 0.001656853779351491, 0.0007128594249201316], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.01513887524044183, 0.04126035969920443, 0.059665530591223884, 0.0646994524129446, 0.05452047588854822, 0.046029690725384015, 0.04952314420243866, 0.046949947484295804, 0.042838602629238956, 0.04095390037655052, 0.04209583989387418, 0.035370235254640774, 0.024982943507328315, 0.02477251591667651, 0.022909695917641672, 0.02360574661841937, 0.01835577215644545], [0.0, 0.001130412167050239, 0.001130412167050239, 0.02103568830856657, 0.05814116532930699, 0.06457544800986917, 0.0608229543164828, 0.0414561143166403, 0.03848518678185401, 0.022750362682619894, 0.024454812448616896, 0.02527128623125571, 0.022230931267836387, 0.02010614945519215, 0.019476646673392013, 0.02054017399989052, 0.020645247609666455, 0.02033393971627395, 0.010214442589934217, 0.008632253164529886], [0.0, 0.001130412167050239, 0.007011344344321024, 0.03889009547008932, 0.07920677723466346, 0.07423479588840735, 0.05859013486527152, 0.04013814526608959, 0.02714396563180163, 0.016159296962269847, 0.014321724518215251, 0.011104814730380753, 0.008619308202797, 0.0071068089292355445, 0.005897422510638241, 0.004385838676700178, 0.0044524675130884735, 0.004281128273065248, 0.0017932866366198895, 0.0008027156549520573], [0.0, 0.001272704430868878, 0.008699297751227585, 0.04512112888640968, 0.08125344747051122, 0.06784775708490966, 0.04418438758062999, 0.027636674730868088, 0.022159353076599272, 0.01281426386832743, 0.008861164158924392, 0.004374557053211664, 0.0026586018470994037, 0.002207205904081021, 0.002180511182108624, 0.0016353833865814685, 0.0016353833865814685, 0.0008027156549520573, 0.0008027156549520573, 0.0]]
yerr += [[0.0, 0.001186312319364703, 0.008353168501715785, 0.042518204563880196, 0.0771146454390909, 0.06505848050557826, 0.042296841762332046, 0.02706724356662368, 0.021089506161391488, 0.012168079472747501, 0.008599185978907648, 0.004257207230116636, 0.0026513180955990375, 0.002042669167281044, 0.0020009232693853425, 0.0015155750798722222, 0.0015155750798722222, 0.0007128594249201316, 0.0007128594249201316, 0.0], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.01513887524044183, 0.04119654038237955, 0.05954546638401433, 0.06454091722508437, 0.05445139348180503, 0.04617975208048419, 0.04952314420243866, 0.04746079311920488, 0.042838602629238956, 0.04089750378103106, 0.04209583989387418, 0.035370235254640774, 0.024982943507328315, 0.02477251591667651, 0.022909695917641672, 0.02360574661841937, 0.01835577215644545], [0.0, 0.001130412167050239, 0.001130412167050239, 0.02103568830856657, 0.05839600284341096, 0.06467590629017296, 0.06010219924963477, 0.041517084973610306, 0.038488671250978046, 0.022750362682619894, 0.0246057511265708, 0.02527128623125571, 0.022230931267836387, 0.02010614945519215, 0.019476646673392013, 0.02054017399989052, 0.020645247609666455, 0.02033393971627395, 0.010214442589934217, 0.008632253164529886], [0.0, 0.001130412167050239, 0.007011344344321024, 0.038961003563398884, 0.07958987441632322, 0.0739931093232551, 0.058384373469524006, 0.040526909850463956, 0.02714396563180163, 0.016159296962269847, 0.014321724518215251, 0.011104814730380753, 0.008619308202797, 0.0071068089292355445, 0.005897422510638241, 0.004385838676700178, 0.0044524675130884735, 0.004281128273065248, 0.0017932866366198895, 0.0008027156549520573], [0.0, 0.001272704430868878, 0.008699297751227585, 0.04531487330252202, 0.08171083598552101, 0.0679167610457368, 0.04418438758062999, 0.028591489586635873, 0.022159353076599272, 0.01281426386832743, 0.008861164158924392, 0.004374557053211664, 0.0026586018470994037, 0.002207205904081021, 0.002180511182108624, 0.0016353833865814685, 0.0016353833865814685, 0.0008027156549520573, 0.0008027156549520573, 0.0]]

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


y,yerr = np.array(y)*100,np.array(yerr)*100
nodes = 2
for sel in [[2,0,9,1],[5,3,14,1],[8,6,19,1]]:
	y1 = y[sel,-n_soft:];yerr1 = yerr[sel,-n_soft:]
	methods_tmp = ['Optimal','Ours','CR','Base']
	x = [[0.1*i for i in range(21-n_soft,21)]for _ in range(4)]
	line_plot(x,y1,methods_tmp,colors,
			f'/home/bo/Dropbox/Research/SIGCOMM23/images/soft_ea_nodes{nodes}.eps',
			'Deadline (s)','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,
			yerr=yerr1)	
	nodes += 1

y = [[0.9363019169329073, 0.9326737220447283, 0.9261681309904155, 0.9169788338658146, 0.9042671725239618, 0.8814596645367413, 0.8548003194888179, 0.8337160543130991, 0.798594249201278, 0.7647723642172524, 0.7231709265175719, 0.6767292332268371, 0.6501936900958467, 0.5802056709265175, 0.5274261182108625, 0.46292931309904156, 0.39433905750798726, 0.33431309904153356, 0.24523562300319496, 0.17307108626198087, 0.09999999999999999], [0.9411940894568691, 0.9006709265175719, 0.8524800319488817, 0.8144888178913737, 0.7751038338658147, 0.7340654952076677, 0.67810303514377, 0.6511741214057508, 0.5978973642172524, 0.5701497603833865, 0.5325918530351437, 0.46763378594249205, 0.4381629392971246, 0.3936521565495208, 0.3590315495207668, 0.3014297124600639, 0.2696385782747604, 0.230301517571885, 0.17648162939297127, 0.13472643769968054, 0.09999999999999999], [0.9411940894568691, 0.937745607028754, 0.9318690095846645, 0.9229892172523962, 0.9098781948881788, 0.8875099840255594, 0.8606709265175718, 0.8402256389776358, 0.8037859424920126, 0.7696545527156549, 0.7281928913738019, 0.6813817891373802, 0.6546365814696485, 0.5844189297124601, 0.5321485623003195, 0.46674321086261983, 0.3972144568690096, 0.33664936102236426, 0.2473123003194889, 0.17442891373801922, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9400638977635782, 0.9382208466453674, 0.935892571884984, 0.9304712460063899, 0.9187759584664539, 0.902869408945687, 0.8885623003194889, 0.8597583865814699, 0.8414856230031947, 0.7936002396166133, 0.7650499201277955, 0.7125419329073482, 0.6528055111821086, 0.5939117412140574, 0.5140275559105432, 0.4321186102236422, 0.3104512779552716, 0.21243610223642176, 0.09999999999999999], [0.9411940894568691, 0.9411940894568691, 0.9411940894568691, 0.9409165335463259, 0.938290734824281, 0.9390934504792334, 0.9357927316293928, 0.9242272364217252, 0.918113019169329, 0.9025339456869009, 0.8888977635782748, 0.8546265974440894, 0.8358765974440894, 0.794151357827476, 0.7389077476038338, 0.6802316293929712, 0.596273961661342, 0.5136421725239615, 0.37125, 0.2529013578274761, 0.09999999999999999]]
yerr = [[0.0, 0.003232679469800827, 0.006721895262968119, 0.00842487572583564, 0.008134901927732903, 0.00870537069066773, 0.013092925498863617, 0.012673107918681975, 0.018397486065869863, 0.018866247107201035, 0.01866136798822496, 0.019868267455407713, 0.02415092329764275, 0.013815579993989897, 0.0280572098092225, 0.014832873340755198, 0.018444084134533467, 0.01864215199124998, 0.012424224101828418, 0.008321246342186986, 0.0], [1.1102230246251565e-16, 0.0069597535771570365, 0.015125446937568229, 0.017518714939407903, 0.013206165550282875, 0.020330320011702128, 0.02237894632782985, 0.025534139101776564, 0.026913925755291435, 0.025000772633616997, 0.016151432632817393, 0.028374656074505223, 0.026125064134430956, 0.020006601066846555, 0.011797026078106697, 0.021348493695650218, 0.012442484251411768, 0.01876499927860723, 0.014249324398814786, 0.008147931144928894, 0.0], [1.1102230246251565e-16, 0.0031350355350767568, 0.006958692488708289, 0.008142436772168872, 0.008495461498120725, 0.009034378069027916, 0.012739351204582576, 0.01311098419799636, 0.019192596867821682, 0.01885160652041262, 0.01855054399369609, 0.020163293093335456, 0.023679847043820618, 0.01311875747419943, 0.028892859178731536, 0.015367758386209483, 0.01818672129335098, 0.01920620786946041, 0.012699398569217483, 0.00829825309938327, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 0.0013852762606664246, 0.002830120991057784, 0.0033434920035537706, 0.006495874348376657, 0.0067520957293272016, 0.006670327700196999, 0.010824996363497711, 0.013599638597536965, 0.013394509154927453, 0.014130003767069561, 0.013536424328876452, 0.007917615290921769, 0.029413445272071938, 0.018489286329811398, 0.017418659312816474, 0.020817007414061003, 0.012403565099641599, 0.014582124567555805, 0.0], [1.1102230246251565e-16, 1.1102230246251565e-16, 1.1102230246251565e-16, 0.0008326677316293774, 0.0030132470610544247, 0.0019863102799945595, 0.0032371438110367127, 0.004922762888838655, 0.007290255612493812, 0.011211880046567493, 0.00937524135725542, 0.014838895621706571, 0.010924068182304389, 0.013897741917512783, 0.026033126000339243, 0.01577495500749436, 0.02165730295288239, 0.019265135567461417, 0.015889984408020478, 0.019450716651676025, 0.0]]
y = [[0.9399960063897763, 0.9401657348242811, 0.9390195686900957, 0.9367571884984025, 0.9343250798722045, 0.9293350638977635, 0.9138258785942492, 0.900892571884984, 0.8823242811501597, 0.8556689297124601, 0.8241214057507985, 0.8028294728434504, 0.7485343450479233, 0.7065435303514376, 0.654966054313099, 0.5853354632587859, 0.49640375399361025, 0.4235642971246006, 0.3254812300319489, 0.22182308306709264, 0.09999999999999999], [0.9410942492012779, 0.8962100638977637, 0.8519608626198083, 0.8165255591054313, 0.7777296325878595, 0.7379712460063896, 0.6777755591054314, 0.6354412939297124, 0.5900818690095847, 0.5579093450479233, 0.5222703674121405, 0.4996904952076676, 0.44107028753993616, 0.39497603833865813, 0.34951876996805115, 0.3085742811501598, 0.2570367412140575, 0.21807108626198085, 0.18462460063897765, 0.1400579073482428, 0.09999999999999999], [0.9410942492012779, 0.9394888178913738, 0.9348302715654953, 0.9258326677316294, 0.9138298722044729, 0.8921665335463258, 0.8597404153354633, 0.8305051916932908, 0.7929492811501596, 0.7691673322683705, 0.7250638977635782, 0.6952835463258785, 0.6374860223642171, 0.5829972044728435, 0.5298282747603833, 0.4732308306709265, 0.38874600638977636, 0.334810303514377, 0.2586261980830672, 0.18538738019169332, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.940836661341853, 0.9376357827476038, 0.9356529552715654, 0.9314616613418532, 0.915972444089457, 0.9029293130990415, 0.8849600638977636, 0.8576357827476038, 0.8264676517571884, 0.8065734824281151, 0.7508706070287539, 0.7097284345047924, 0.6576817092651757, 0.5870626996805111, 0.4990994408945687, 0.4267591853035144, 0.3269488817891374, 0.22283146964856235, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9405491214057509, 0.9402615814696486, 0.9375658945686901, 0.9354053514376997, 0.9292611821086261, 0.9199860223642172, 0.9036521565495207, 0.8854173322683707, 0.8707907348242812, 0.8272923322683706, 0.7912400159744409, 0.7428534345047922, 0.6705571086261981, 0.5933366613418529, 0.5022523961661342, 0.38875000000000004, 0.25860822683706075, 0.09999999999999999]]
yerr = [[0.0, 0.0005414435648145597, 0.0013777796105021148, 0.002788898462405699, 0.004516621325515751, 0.006101685249872873, 0.004696952539763973, 0.005311021516774698, 0.010871652330180882, 0.013107185875209107, 0.017273162920830434, 0.02074417018033179, 0.022185067700907155, 0.024867087293651776, 0.01615677947056738, 0.017762104402648443, 0.010614665325765254, 0.019135983481389937, 0.02006971613228766, 0.016833463725008515, 0.0], [0.0, 0.00718655569240537, 0.013945152236419555, 0.013161866940018516, 0.015616080858299063, 0.017680944689895162, 0.03014832936527378, 0.02307728401099896, 0.02624825657296736, 0.01594731485315254, 0.024414466741554793, 0.034938185546443006, 0.017993988095187263, 0.017394882557374276, 0.02327871845257599, 0.011104951888247982, 0.012902653166971303, 0.017070123866907615, 0.012508344880617167, 0.011387009235368548, 0.0], [0.0, 0.001315383905625052, 0.005278459419625607, 0.003974956936471972, 0.00818211307047219, 0.012309671257495403, 0.017609078774857175, 0.006251945784651708, 0.01478408249970118, 0.01458423982673191, 0.017488109456241983, 0.03334872437130384, 0.022252111712762326, 0.01747537451155398, 0.017847682238936378, 0.014019569116702837, 0.021555733058477734, 0.01719994432500281, 0.017016259543734674, 0.013180177715550736, 0.0], [0.0, 0.0, 0.0007727635782747378, 0.002415949349991801, 0.004756599763192114, 0.006023717337671005, 0.004562482954283959, 0.006999575529339042, 0.010104409213436118, 0.01371171684796772, 0.01792576613410159, 0.021266101561065426, 0.02139855252162924, 0.025720748595399207, 0.016430204737468885, 0.019008381614312987, 0.010212401819678295, 0.019520304434589374, 0.01971188763272708, 0.017339189553161386, 0.0], [0.0, 0.0, 0.0, 0.0010904841391129736, 0.0012727044308688464, 0.002080265585421391, 0.003959342295875331, 0.004953398579105959, 0.007956312251896913, 0.009066244858239113, 0.011004197310867441, 0.013122840945873966, 0.01874860027030457, 0.023428915334346784, 0.011263901175692658, 0.02188558387872796, 0.018956927906578307, 0.022326372160841195, 0.01729497264555927, 0.01730145947855376, 0.0]]
y += [[0.9375, 0.9376098242811504, 0.9373003194888179, 0.9369009584664537, 0.9344468849840256, 0.9345786741214057, 0.9287400159744408, 0.9260882587859426, 0.9147384185303513, 0.8998282747603834, 0.8788039137380192, 0.8536561501597444, 0.8306230031948882, 0.7871126198083067, 0.7334165335463257, 0.6641313897763579, 0.5854612619808306, 0.4720806709265176, 0.38166333865814706, 0.25494209265175727, 0.09999999999999999], [0.9410942492012779, 0.9005591054313099, 0.854554712460064, 0.8204053514376998, 0.7662999201277956, 0.7354432907348243, 0.6911561501597443, 0.6506689297124602, 0.6010503194888178, 0.5602256389776358, 0.5316453674121405, 0.4751757188498402, 0.4474760383386581, 0.3854253194888179, 0.3576297923322684, 0.30457068690095845, 0.26639376996805114, 0.21806509584664538, 0.18148162939297124, 0.14338658146964858, 0.09999999999999999], [0.9410942492012779, 0.9402715654952075, 0.9338398562300319, 0.9234045527156549, 0.9038718051118211, 0.8906090255591057, 0.864239217252396, 0.8344868210862618, 0.8076417731629393, 0.7654932108626198, 0.7329492811501597, 0.6856230031948882, 0.6431170127795527, 0.5831269968051118, 0.5322723642172524, 0.45883785942492017, 0.4081110223642172, 0.32064696485623007, 0.26119408945686906, 0.1807088658146965, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9405491214057509, 0.9384085463258787, 0.9337999201277956, 0.9293989616613418, 0.9183007188498402, 0.9065555111821084, 0.8901557507987222, 0.8630011980830672, 0.8329992012779552, 0.7987799520766773, 0.7667511980830671, 0.7128953674121405, 0.6602955271565495, 0.576070287539936, 0.5087420127795526, 0.4027436102236422, 0.3303773961661342, 0.22306709265175723, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9394988019169329, 0.9391613418530351, 0.9334724440894568, 0.9316793130990414, 0.9205990415335463, 0.9063378594249201, 0.8857727635782748, 0.860155750798722, 0.838939696485623, 0.7940714856230031, 0.7411341853035143, 0.6710403354632588, 0.5905431309904154, 0.47716253993610225, 0.38564696485623007, 0.25715854632587865, 0.09999999999999999]]
yerr += [[0.0, 0.000958102383596927, 0.0007797773238724486, 0.0008329334510628683, 0.002885593423488358, 0.0031679934537966387, 0.0031266174401323767, 0.004244150431589769, 0.006886670505459622, 0.008609890217423428, 0.008057395646355882, 0.013485915123252186, 0.010003536048408689, 0.018376579543785007, 0.013902554654366104, 0.015662239865133053, 0.020895680509233242, 0.02295103300049234, 0.01710674500076731, 0.011439442743217107, 0.0], [0.0, 0.012304241159406561, 0.01659933520504075, 0.016592221233154392, 0.019009234486297297, 0.023655264500178953, 0.02606424205292366, 0.016559805285539728, 0.027055216704290227, 0.01463818441048046, 0.03174058596714932, 0.024509755530488175, 0.017211874447981723, 0.01793539966399801, 0.020436563234412775, 0.016362173581683164, 0.01531666929448429, 0.016267222702602075, 0.01752902893657775, 0.00819317534055581, 0.0], [0.0, 0.0012585195891636192, 0.0037637742421805023, 0.007388936851287273, 0.01030024950904621, 0.012644182598634585, 0.01210977091524217, 0.011456085107296378, 0.01986304468692187, 0.01956862173571549, 0.020343062893865692, 0.026567241003961357, 0.013585957496921465, 0.024419703586682627, 0.021919752558263447, 0.016757271777722422, 0.019953356813745785, 0.017593459925232972, 0.015916426202971748, 0.014170895392723504, 0.0], [0.0, 0.0, 0.0010923108020666826, 0.0027268090152144805, 0.006680833960101362, 0.005577824446071641, 0.006887063041350397, 0.009993382396625687, 0.01178930377944336, 0.013234114022494522, 0.019795506917778528, 0.01848671774955087, 0.012560791255871723, 0.02488832790468756, 0.013884327995717354, 0.017392562710434444, 0.02118876096333832, 0.020338477458197335, 0.020600348532264133, 0.012188802630206365, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0024739190469077435, 0.002748691993268106, 0.0026405745060775693, 0.002932398559279646, 0.007787561440127629, 0.008260506414500816, 0.008144073135628324, 0.013631073348135073, 0.010307992477655516, 0.018097395041730184, 0.014511523390347962, 0.016322608669369822, 0.021763065598096473, 0.023172948788172224, 0.017508323358227013, 0.011870277508342858, 0.0]]
y += [[0.9240215654952075, 0.9232288338658148, 0.9157188498402554, 0.9073342651757187, 0.8876976837060703, 0.8753274760383387, 0.8545507188498401, 0.8230311501597443, 0.7895587060702874, 0.7521705271565495, 0.7226497603833865, 0.660948482428115, 0.6381369808306709, 0.5676477635782747, 0.5208027156549521, 0.4473642172523961, 0.4015595047923323, 0.33284145367412143, 0.26164137380191693, 0.17886581469648563, 0.09999999999999999], [0.9410942492012779, 0.9018091054313098, 0.8498262779552717, 0.8257068690095848, 0.7746166134185304, 0.7414596645367413, 0.685615015974441, 0.6445327476038338, 0.6061481629392971, 0.5636761182108627, 0.5260303514376996, 0.462879392971246, 0.4484285143769967, 0.39091054313099044, 0.34775758785942495, 0.3054013578274761, 0.2637579872204473, 0.23126397763578277, 0.1882907348242812, 0.14283346645367415, 0.09999999999999999], [0.9410942492012779, 0.9402715654952077, 0.9323722044728433, 0.9241873003194888, 0.9039516773162941, 0.8915415335463258, 0.8700159744408944, 0.8375878594249201, 0.8037160543130991, 0.7655890575079872, 0.7361182108626197, 0.6727795527156548, 0.6493789936102237, 0.577242412140575, 0.5288498402555911, 0.4546226038338658, 0.4072304313099043, 0.33716453674121405, 0.2648961661341854, 0.1804432907348243, 0.09999999999999999], [0.9410942492012779, 0.9408166932907347, 0.9402515974440895, 0.9384085463258787, 0.9318989616613418, 0.9286162140575082, 0.9201437699680509, 0.9065355431309904, 0.8865575079872204, 0.8581829073482428, 0.837857428115016, 0.7867492012779552, 0.7699500798722045, 0.7056729233226837, 0.6498342651757189, 0.5834784345047924, 0.5045007987220448, 0.43298921725239625, 0.32922523961661343, 0.21766573482428125, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9408067092651757, 0.9410942492012779, 0.9391912939297125, 0.937020766773163, 0.9349500798722044, 0.9322144568690096, 0.9182907348242813, 0.9045247603833866, 0.8914616613418531, 0.8569928115015975, 0.8413238817891374, 0.7867991214057508, 0.7373242811501596, 0.6741793130990417, 0.5925459265175719, 0.5075658945686901, 0.3900758785942492, 0.2520447284345048, 0.09999999999999999]]
yerr += [[1.1102230246251565e-16, 0.0012128368069627925, 0.002009163037287492, 0.004559026177608105, 0.004777369896755866, 0.010407846282822103, 0.007427135067536107, 0.019248634127235344, 0.011370192827218916, 0.01471354636924572, 0.016277193102212772, 0.01595375820113512, 0.01919176014303523, 0.02240241062057284, 0.01833338272659684, 0.01915234662750271, 0.02713681869023445, 0.03188325684489882, 0.0180093755956672, 0.013659963236110021, 0.0], [0.0, 0.010345833264467601, 0.01127144415383577, 0.011208998220441479, 0.014854433872319023, 0.015369851520777734, 0.014714270302442189, 0.01793048670346185, 0.018314599061651792, 0.017944712063340418, 0.01725911142752746, 0.013269655654528647, 0.021581123881472805, 0.02391488345358719, 0.025094410089816246, 0.02327696925279175, 0.0201780629438965, 0.017579351108637047, 0.009684814395103873, 0.008383231159874981, 0.0], [0.0, 0.0012569344953711665, 0.0020789378517483328, 0.004654561281949599, 0.004800964420190137, 0.010784371549703525, 0.007135823827939953, 0.01899487602826977, 0.011989139324796437, 0.014988820556830674, 0.017329109081774718, 0.01630580050574032, 0.019475204684981352, 0.02294639791117799, 0.018998743451960762, 0.019475320973374056, 0.027872041279210637, 0.032677910332194124, 0.01866876949151682, 0.013746080066002283, 0.0], [0.0, 0.0008326677316293774, 0.001287429933702177, 0.0029170753101952227, 0.0029982977936203565, 0.007894587892452634, 0.006544288021077912, 0.009274423756036935, 0.0074956872345891155, 0.009575261628382141, 0.018136136626103656, 0.014752077236044608, 0.01424235422342263, 0.017780697859066986, 0.016385824722797652, 0.016458639213330035, 0.03130713193536697, 0.024590686083732303, 0.018211125793317254, 0.017452154470294023, 0.0], [0.0, 0.0, 0.0008626198083066973, 0.0, 0.0012480607015504502, 0.0017967030034408194, 0.004184362642854235, 0.005712168422954852, 0.004985099046247042, 0.004980443462684424, 0.009575700514019235, 0.011466745457659152, 0.009685773812643609, 0.016924603618504327, 0.017819813262204918, 0.015965086743074376, 0.024587661264695405, 0.0209390271131603, 0.015833211561441844, 0.016652399778021024, 0.0]]

# hard
y = [[0.9372004792332268, 0.9361701277955271, 0.9269009584664536, 0.9164237220447286, 0.9027735623003196, 0.8834584664536742, 0.8610463258785941, 0.8285802715654953, 0.8152276357827477, 0.759117412140575, 0.7229712460063897, 0.6756689297124601, 0.6312699680511182, 0.5680770766773162, 0.5175319488817891, 0.4741453674121406, 0.4014556709265175, 0.33900359424920135, 0.2573242811501598, 0.18030551118210864, 0.09999999999999999], [0.9411940894568691, 0.902811501597444, 0.8544488817891374, 0.8166713258785941, 0.7749001597444088, 0.7351876996805112, 0.6917911341853034, 0.6364616613418531, 0.6134464856230032, 0.5541433706070287, 0.5137739616613418, 0.46715055910543135, 0.43328873801916934, 0.39498402555910544, 0.3444468849840256, 0.32219249201277955, 0.2572484025559106, 0.23126397763578277, 0.18252396166134188, 0.14374400958466454, 0.09999999999999999], [0.9411940894568691, 0.9401437699680513, 0.9326417731629391, 0.9210263578274761, 0.9081449680511181, 0.8886900958466454, 0.8666873003194888, 0.8341613418530353, 0.8198901757188498, 0.7644788338658146, 0.7271046325878594, 0.6807507987220446, 0.6360423322683705, 0.5725099840255591, 0.5214756389776357, 0.4769908146964855, 0.4042711661341853, 0.34105031948881787, 0.25919129392971246, 0.18125399361022368, 0.09999999999999999]]
y += [[0.9400958466453673, 0.9401956869009584, 0.9383805910543129, 0.9373063099041534, 0.933089057507987, 0.9279732428115016, 0.9211841054313099, 0.9051557507987219, 0.8867132587859425, 0.857855431309904, 0.8277096645367411, 0.7944748402555909, 0.7633366613418531, 0.7049161341853036, 0.6524500798722045, 0.5657148562300318, 0.5106709265175718, 0.44044728434504793, 0.3210682907348243, 0.21947683706070292, 0.09999999999999999], [0.9410942492012779, 0.901283945686901, 0.8557068690095846, 0.8141493610223642, 0.7788997603833866, 0.7213238817891373, 0.6767871405750798, 0.6389976038338658, 0.5990415335463258, 0.574297124600639, 0.5195986421725239, 0.4946745207667732, 0.43301717252396166, 0.3993829872204473, 0.3646305910543131, 0.289576677316294, 0.2619848242811502, 0.23528554313099043, 0.18138378594249202, 0.14484424920127797, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9394388977635784, 0.9389736421725241, 0.9354852236421725, 0.9292012779552715, 0.9227515974440894, 0.9070926517571886, 0.8886102236421725, 0.8608406549520767, 0.8305351437699681, 0.7975898562300319, 0.7667212460063897, 0.7082308306709264, 0.6555151757188498, 0.5692392172523961, 0.5138258785942492, 0.44196485623003195, 0.322366214057508, 0.2194468849840256, 0.09999999999999999]]
y += [[0.9375, 0.9376497603833867, 0.9371445686900959, 0.9371026357827474, 0.9352535942492013, 0.9325499201277955, 0.9315035942492014, 0.9250638977635782, 0.9102176517571886, 0.9010862619808307, 0.8807787539936103, 0.8570706869009586, 0.8257128594249202, 0.7743091054313098, 0.7381749201277954, 0.6637919329073483, 0.5800619009584664, 0.4988019169329073, 0.38542731629392973, 0.2624101437699681, 0.09999999999999999], [0.9410942492012779, 0.8992911341853036, 0.8525379392971246, 0.8219109424920127, 0.7779173322683706, 0.7269888178913739, 0.6812340255591054, 0.6470347444089457, 0.6034464856230031, 0.5638059105431309, 0.5189456869009584, 0.47060103833865813, 0.44147164536741207, 0.39790934504792336, 0.3615874600638978, 0.3022923322683707, 0.2679213258785943, 0.22620607028753992, 0.17921325878594252, 0.14382388178913744, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9405890575079873, 0.9408166932907349, 0.9397563897763579, 0.9380810702875401, 0.9364756389776356, 0.9311142172523962, 0.915, 0.907645766773163, 0.8875079872204472, 0.8641693290734824, 0.8331110223642172, 0.7802096645367411, 0.7443949680511183, 0.6709804313099041, 0.5870706869009584, 0.502745607028754, 0.3899201277955272, 0.265145766773163, 0.09999999999999999]]
y += [[0.9008586261980831, 0.8985702875399362, 0.8940814696485624, 0.8828095047923323, 0.8671465654952077, 0.8537360223642173, 0.8295786741214058, 0.7944948083067093, 0.7719628594249202, 0.7372224440894569, 0.708314696485623, 0.6763977635782747, 0.6091114217252397, 0.5619428913738019, 0.4979033546325879, 0.4528314696485623, 0.39591253993610226, 0.33580271565495207, 0.2525658945686901, 0.17929912140575083, 0.09999999999999999], [0.9410942492012779, 0.894195287539936, 0.8557068690095846, 0.8150479233226837, 0.7675239616613417, 0.7414157348242811, 0.685467252396166, 0.6375998402555911, 0.602354233226837, 0.557767571884984, 0.5394189297124601, 0.49089656549520766, 0.43088857827476046, 0.39105630990415335, 0.34899560702875404, 0.3096485623003195, 0.2700818690095847, 0.22837659744408945, 0.18404952076677317, 0.14265375399361022, 0.09999999999999999], [0.9410942492012779, 0.9386861022364217, 0.934117412140575, 0.9220067891373802, 0.9056449680511184, 0.8908466453674123, 0.8662899361022364, 0.8287799520766773, 0.8053694089456869, 0.7687020766773163, 0.7391353833865815, 0.705011980830671, 0.6350798722044728, 0.5850858626198082, 0.517511980830671, 0.4705231629392971, 0.4108087060702876, 0.3466952875399361, 0.2597643769968051, 0.18401158146964863, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9397663738019169, 0.9347024760383388, 0.9293690095846646, 0.9171505591054314, 0.9016413738019171, 0.8865155750798721, 0.8616433706070288, 0.8356309904153354, 0.8075758785942492, 0.7606249999999999, 0.6987959265175718, 0.6427615814696486, 0.5902316293929711, 0.5241214057507988, 0.4300818690095848, 0.32897763578274763, 0.22724840255591058, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.940301517571885, 0.9383486421725239, 0.9340774760383386, 0.9292811501597443, 0.9197264376996805, 0.9048023162939298, 0.8833965654952076, 0.8673222843450479, 0.8349940095846644, 0.7846605431309903, 0.7317392172523961, 0.6723682108626199, 0.6105611022364216, 0.5028154952076677, 0.38717052715654954, 0.25809904153354635, 0.09999999999999999]]
y += [[0.9008586261980831, 0.9008586261980831, 0.9003534345047924, 0.8985303514376998, 0.8936461661341852, 0.887286341853035, 0.8786880990415338, 0.8633027156549522, 0.8491054313099042, 0.8241194089456869, 0.8049960063897764, 0.7586242012779552, 0.721517571884984, 0.6827336261980829, 0.6331190095846646, 0.5638997603833865, 0.4904452875399362, 0.41472444089456867, 0.30718250798722047, 0.20833067092651764, 0.09999999999999999], [0.9410942492012779, 0.8949400958466451, 0.8570247603833867, 0.8230511182108626, 0.7672623801916933, 0.7343630191693291, 0.6854852236421725, 0.644339057507987, 0.6101857028753993, 0.5666114217252396, 0.5148821884984025, 0.4640774760383386, 0.4210563099041534, 0.39790734824281154, 0.34567492012779555, 0.31444888178913744, 0.2750259584664537, 0.22485023961661343, 0.18711461661341858, 0.1391932907348243, 0.09999999999999999], [0.9410942492012779, 0.9386561501597445, 0.932086661341853, 0.9241873003194888, 0.902869408945687, 0.8905810702875397, 0.8681150159744409, 0.8405151757188498, 0.8103853833865815, 0.771349840255591, 0.7307048722044728, 0.6732348242811503, 0.6232288338658147, 0.5836501597444089, 0.5252456070287539, 0.46285143769968046, 0.40597444089456874, 0.33456669329073485, 0.25801916932907354, 0.18184904153354636, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9405591054313099, 0.9386861022364217, 0.9334025559105431, 0.9269329073482429, 0.9177955271565494, 0.9015215654952078, 0.8867152555910544, 0.8606409744408945, 0.8399600638977635, 0.7921405750798721, 0.7526777156549521, 0.7121565495207667, 0.6602056709265176, 0.5851158146964857, 0.5098742012779552, 0.43066892971246, 0.3169369009584665, 0.2134225239616614, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9397563897763579, 0.9375359424920129, 0.9344349041533546, 0.9275758785942492, 0.9180611022364218, 0.9024940095846647, 0.8869548722044728, 0.857300319488818, 0.8192432108626198, 0.7976737220447284, 0.7512260383386582, 0.6707328274760382, 0.5922084664536741, 0.5077436102236422, 0.3771126198083067, 0.2499420926517572, 0.09999999999999999]]
y += [[0.9008586261980831, 0.9008586261980831, 0.9008586261980831, 0.9006010383386581, 0.9003334664536741, 0.8973123003194889, 0.8944888178913739, 0.8878214856230031, 0.8829972044728434, 0.8599600638977634, 0.8515215654952076, 0.825223642172524, 0.7920347444089457, 0.7543550319488819, 0.7038658146964857, 0.6484524760383386, 0.5718450479233226, 0.4860862619808307, 0.3735722843450479, 0.2505650958466454, 0.09999999999999999], [0.9410942492012779, 0.8960143769968052, 0.8558186900958467, 0.8163418530351437, 0.779017571884984, 0.7298921725239615, 0.6852256389776358, 0.6523482428115016, 0.6124480830670926, 0.56066892971246, 0.5153194888178914, 0.4816194089456869, 0.4276038338658147, 0.3882088658146965, 0.344866214057508, 0.31517372204472843, 0.2692671725239617, 0.23395167731629402, 0.1780311501597444, 0.14685702875399362, 0.09999999999999999], [0.9410942492012779, 0.9394988019169329, 0.9319668530351439, 0.9213039137380192, 0.909163338658147, 0.8896585463258786, 0.855169728434505, 0.8485423322683705, 0.819810303514377, 0.7630970447284344, 0.7295187699680511, 0.6811821086261981, 0.6335283546325878, 0.5810363418530352, 0.5190215654952076, 0.46562300319488825, 0.4140375399361022, 0.34432907348242814, 0.2537999201277955, 0.1893330670926518, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9403115015974441, 0.9381110223642173, 0.9357428115015974, 0.9285862619808306, 0.9156829073482428, 0.9050619009584665, 0.8963777955271567, 0.8571865015974443, 0.8378594249201278, 0.7975019968051118, 0.758270766773163, 0.7038079073482428, 0.6448941693290735, 0.5908785942492012, 0.5172144568690096, 0.43656749201277956, 0.3267511980830671, 0.22762579872204475, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9410942492012779, 0.9408266773162939, 0.9405391373801917, 0.9372583865814695, 0.9343051118210862, 0.9275379392971246, 0.9222843450479233, 0.8988877795527157, 0.8896305910543131, 0.861216054313099, 0.8270487220447285, 0.7873422523961662, 0.7342372204472843, 0.6760882587859424, 0.595177715654952, 0.5067731629392972, 0.38756988817891375, 0.2592312300319489, 0.09999999999999999]]

yerr = [[0.0, 0.002383087978836537, 0.003873038115141878, 0.0061453230457425975, 0.012967568025285573, 0.010063407266560125, 0.008824341552946188, 0.01611947452454557, 0.01890696132833872, 0.013174793019248588, 0.019867655361733523, 0.02406030887038924, 0.019344045784808497, 0.026841082314703498, 0.03579666570409922, 0.028536735524548405, 0.023743592117576182, 0.013090604293229717, 0.01902712161252409, 0.0103868882719809, 0.0], [1.1102230246251565e-16, 0.008922661919075328, 0.010855201934208752, 0.014047328679292339, 0.015535675044506014, 0.021138001603641914, 0.02019673453507974, 0.019017540855227846, 0.02697623136127924, 0.0165951113967507, 0.015869314630196252, 0.01722226165574532, 0.016248312393671693, 0.0225139485027758, 0.024553767381638156, 0.01572217242443465, 0.01882411807025787, 0.0094800814188465, 0.01617607815639621, 0.008617508759934122, 0.0], [1.1102230246251565e-16, 0.0021006389776357715, 0.004292221186753114, 0.005502080447956476, 0.012873009197567621, 0.009931106560055136, 0.008520796056967147, 0.016406406989810517, 0.019195156676365066, 0.014236333899824661, 0.019578295223714896, 0.025014428016724065, 0.02032100402565394, 0.026638863608938305, 0.03630411823945728, 0.02905046349658962, 0.0247702550563865, 0.013421182987465093, 0.019594189900178752, 0.009960898934023193, 0.0]]
yerr += [[1.1102230246251565e-16, 0.000995402841902101, 0.0019230062619173323, 0.0029350189218929797, 0.00398937924839249, 0.0056962200434122765, 0.006343453860877856, 0.008939742647962826, 0.012168240034661555, 0.013548031179396408, 0.010352070348634003, 0.01476602223042188, 0.015315176158104037, 0.013853354946653736, 0.014429536171800459, 0.024700795620692797, 0.02165818295974954, 0.026376494321830004, 0.017636004877208376, 0.013967911382259396, 0.0], [0.0, 0.008788792242237194, 0.014406748369610225, 0.01914729164577468, 0.016900213540757675, 0.015256297891266042, 0.017869286669125815, 0.01721277510219242, 0.030673250835451138, 0.02429898644176315, 0.019434009232524578, 0.018285541630780777, 0.017436328739240895, 0.020986287686181066, 0.01961241417004981, 0.02212697397046627, 0.01855550272008473, 0.024734784397244265, 0.011483376204111629, 0.010381818971123509, 0.0], [0.0, 0.0, 0.0018206975879296605, 0.0022187874482035596, 0.0034773826415680944, 0.005119523263411318, 0.006296798832352713, 0.008350527247477434, 0.011745044717527173, 0.013318481881632085, 0.010710799335342996, 0.014662990250481664, 0.015255625554048257, 0.012476912709232601, 0.015621043017821563, 0.02367807973759844, 0.022399031195351683, 0.025831532860168975, 0.017872497713262124, 0.014208441988889771, 0.0]]
yerr += [[0.0, 0.0007340126308192104, 0.0016790222193592268, 0.001307578062589785, 0.0031738430517763, 0.002771281353113958, 0.004821811372567936, 0.003766440262694593, 0.011413445446441009, 0.007024970391018178, 0.006641610340463003, 0.012514366348787647, 0.015057229462531088, 0.0270727367263636, 0.015020235103509156, 0.024594070767625157, 0.020907486396156177, 0.028864179469003905, 0.018842506916103334, 0.012339164246798697, 0.0], [0.0, 0.007235762980987184, 0.01883170894266918, 0.018729712734086437, 0.021647803322049578, 0.022337040981863596, 0.01886215930979941, 0.02078621611097996, 0.0263849748859761, 0.030812869391009003, 0.016535778834780458, 0.02421286892331994, 0.013720264138036297, 0.022343865629389774, 0.024706330212331377, 0.028134078150203846, 0.014843995233644925, 0.01338282025191434, 0.012181244413064132, 0.011104893901555347, 0.0], [0.0, 0.0, 0.0010165307096568136, 0.0008326677316293772, 0.002489656459515469, 0.00228894232457122, 0.0038375150073354097, 0.004086751317099877, 0.01060951544718555, 0.008054753685088496, 0.006235662698745973, 0.013869294865888601, 0.015110500730909011, 0.02580070555943828, 0.013983579435754644, 0.024901725834084645, 0.021823995121518922, 0.02967566715074053, 0.018866276695059758, 0.012773052866064803, 0.0]]
yerr += [[0.0, 0.0026430519859409053, 0.0037471068126274814, 0.005128847085578513, 0.007483731733161721, 0.014078392131387204, 0.010966525845372391, 0.010401173823841663, 0.018207857202011225, 0.009649931080905874, 0.01971348352131072, 0.021696470612410344, 0.015134544704144194, 0.01702507339130345, 0.01907224650953194, 0.01955821588365576, 0.028186737221669186, 0.010989353030241253, 0.01708865359710452, 0.010836906386475983, 0.0], [0.0, 0.007530147995075633, 0.016762150271849352, 0.015131478203856777, 0.02224443439866386, 0.015558232188963706, 0.02220109094617098, 0.021489354626241013, 0.031938181427459404, 0.016598618300013003, 0.031117169966981637, 0.021541707997679013, 0.019605102903242758, 0.013367086659526921, 0.023876582935711307, 0.024435653698472056, 0.021622521133528815, 0.008790617629581903, 0.011052235303073329, 0.008116404627685918, 0.0], [0.0, 0.0028034347312171563, 0.003775463720559373, 0.0053712086647453635, 0.007580178610023179, 0.014918248991482185, 0.011702773078324197, 0.010384327533243648, 0.019497604001554578, 0.009975184584007493, 0.0212425482011355, 0.023242742882763803, 0.01681658431467991, 0.01785540953752468, 0.0196520706009716, 0.02065806171598697, 0.028826368593001476, 0.011613904159955515, 0.0185107280315045, 0.011326691263365694, 0.0], [0.0, 0.0, 0.0008027156549520575, 0.0013304252584572045, 0.003978780873865764, 0.0059464158886346746, 0.007119821086340128, 0.00757782567638287, 0.007708407703806553, 0.008369017102695294, 0.012517414300495284, 0.015242374006242648, 0.01392826301213545, 0.023354749107702896, 0.020855583648807133, 0.025996914571455493, 0.031962281285544514, 0.017758561873033367, 0.019603136859097103, 0.010316613759875118, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0012144794567955021, 0.00368831309455961, 0.003923101614240792, 0.008429366559551794, 0.0034603595570243226, 0.005586440145880616, 0.00528489282804406, 0.009975144812288193, 0.013866759149687635, 0.020169061734520665, 0.016337735081718468, 0.02165426978490202, 0.032115252287172845, 0.014707984458335835, 0.02081425101171868, 0.013499856364420186, 0.0]]
yerr += [[0.0, 0.0, 0.0010106299974330854, 0.002144353009424442, 0.003999943061941027, 0.0057694265471129715, 0.006993021613427631, 0.010369544872122624, 0.005874278915136402, 0.010285083363218412, 0.013638634399889465, 0.01014904538962066, 0.011359217571744802, 0.013648787529150053, 0.02352513246022598, 0.01860773700375607, 0.026502113566243924, 0.015945814006239088, 0.022426082652022002, 0.008146019255403085, 0.0], [0.0, 0.008811976748953194, 0.017701261845312396, 0.02763850253934464, 0.014302654486147322, 0.020826825592425647, 0.022450015934566487, 0.024955924976327246, 0.021987764792628376, 0.011773744253977857, 0.020224852383961555, 0.02552477789143174, 0.014273512264176324, 0.029780154621740586, 0.01972165713805779, 0.015406554779602497, 0.01658430506657368, 0.013893879163417514, 0.016236248295581325, 0.006213625022935488, 0.0], [0.0, 0.002532189022435539, 0.003874526977532265, 0.006922339288707987, 0.008528639818903177, 0.008918001010698284, 0.011063016839345617, 0.017380673911365007, 0.02412747218441641, 0.012475161349891854, 0.02036589500159838, 0.028592336276081837, 0.021041202616329132, 0.029311203919726567, 0.021142753094431636, 0.020581654868296165, 0.027712196005076472, 0.016456781598249298, 0.025263208960966965, 0.008674117728392544, 0.0], [0.0, 0.0, 0.0010712184807357074, 0.0021998086974048714, 0.004440697038611527, 0.00588357059044514, 0.0076371141830499385, 0.010978069858443283, 0.006193348174251001, 0.010704534742442089, 0.013941071527469978, 0.011274550148233147, 0.011608881057905647, 0.01342987881591477, 0.02541378423658674, 0.019272041617441584, 0.028308566348515713, 0.016873460327096784, 0.02413520389317356, 0.009100489627865773, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0018114005217728943, 0.002765265810091264, 0.00409672441739582, 0.005424074006782963, 0.006305563084410572, 0.006398235028777784, 0.010084419566929351, 0.008299905821654097, 0.010958225764700487, 0.011336340267194937, 0.02642285249294204, 0.014596806667946147, 0.02326687872671876, 0.014879989653398646, 0.03193247232345561, 0.012038623423406026, 0.0]]
yerr += [[0.0, 0.0, 0.0, 0.0007727635782747377, 0.0010505567249903464, 0.002541458366944857, 0.003318086914578003, 0.006732120864394751, 0.00719564590748998, 0.010488796633862612, 0.011656430182522746, 0.016294543424826415, 0.013461871405201801, 0.013071890480545845, 0.01862970414971474, 0.02019569568658317, 0.0160620376843471, 0.026133789540091417, 0.017023395985741863, 0.014975340904953015, 0.0], [0.0, 0.010483657377436154, 0.010271299199316632, 0.009396283388891617, 0.011613796186800874, 0.02262086379665006, 0.0321922037368394, 0.018464458488565934, 0.019142559849910517, 0.01675939609324432, 0.015085964508063926, 0.01889880467753711, 0.014876517568454288, 0.02177464520166173, 0.02005083369354587, 0.015700164979868002, 0.013477125276610465, 0.012916443259772342, 0.013847395853693054, 0.005489436250116083, 0.0], [0.0, 0.0017403494002409766, 0.0047799564810244045, 0.005346894156858854, 0.01213833651407642, 0.014418330741443991, 0.009693374984957315, 0.010048811369656697, 0.019883546127114148, 0.021943938722058447, 0.014886106575363961, 0.015513896374489241, 0.014843042178332188, 0.028177154663644795, 0.02248687197890008, 0.019594155612006756, 0.015661733123371976, 0.020687465684250374, 0.014742048638048106, 0.009244896281647428, 0.0], [0.0, 0.0, 0.0011992739023270493, 0.0022321423285499267, 0.004330977765153025, 0.005527156549520733, 0.00485989876080296, 0.011350941360803845, 0.009949906557789016, 0.01894682385457714, 0.015334792442397018, 0.012727149103103725, 0.016535783657329346, 0.0168302032220564, 0.020766596617736933, 0.01878058380380978, 0.01590867916161235, 0.02053270732514298, 0.018049200933160533, 0.01215669727951119, 0.0], [0.0, 0.0, 0.0, 0.0008027156549520573, 0.0011102236421725031, 0.002855716852721691, 0.003457149602833873, 0.0070146910966298924, 0.007320043518222887, 0.011135926463079416, 0.012158126559962314, 0.016613831446277174, 0.013764701107377774, 0.013636694969997443, 0.02021320396387461, 0.021007138090420173, 0.01669026241028353, 0.027534973466999266, 0.017365033429600305, 0.01497629406064952, 0.0]]

y,yerr = np.array(y)*100,np.array(yerr)*100
nodes = 2
for sel in [[2,0,9,1],[5,3,14,1],[8,6,19,1]]:
	y1 = y[sel,:n_hard];yerr1 = yerr[sel,:n_hard]
	methods_tmp = ['Optimal','Ours','CR','Base']
	x = [[5*i for i in range(n_hard)]for _ in range(4)]
	line_plot(x,y1,methods_tmp,colors,
			f'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_ea_nodes{nodes}.eps',
			'Failure Rate (%)','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,
			yerr=yerr1)	
	nodes += 1
exit(0)

# FCC 2-nodes no loss analysis
# CIFAR, IMAGENET EA and FR FCC
n_soft = 11;n_hard = 11;
x = [[-1+0.1*i for i in range(n_soft)] for _ in range(4)]
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
print(y.shape)
y = y[[2,1,5,0],:n_soft];yerr = yerr[[2,1,5,0],:n_soft]
# start=2;y = y[:,start:];yerr = yerr[:,start:];x=[[0.1*i for i in range(1+start,21)] for _ in range(4)]
colors_tmp = colors#['k'] + colors[:2] + ['grey'] + colors[2:4]
linestyles_tmp = linestyles#['solid','dashed','dotted','solid','dashdot',(0, (3, 5, 1, 5)),]
methods_tmp = ['Optimal','Base','CR','Ours']
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea.eps',
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
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea.eps',
		'Failure Rate (%)','Effective Accuracy (%)',lbsize=24,linewidth=8,markersize=0,linestyles=linestyles_tmp,yerr=yerr)
exit(0)

# imagenet 
n_soft = 12;n_hard = 11;
methods_tmp = ['Optimal','Ours','CR','Base']
x = [[i for i in range(25-n_soft,25)] for _ in range(4)]
y = [[0.66215592, 2.7582126399999995, 8.82819752, 18.078571519999997, 28.976434239999996, 38.409999039999995, 45.43412384, 50.10920608, 54.44810976, 58.24905416, 61.70842015999999, 64.33464296000001, 66.2849112, 67.95639839999998, 69.44069848000001, 70.25064320000001, 70.98879344, 71.63894976, 72.15511472, 72.50649128], [0.34377736, 1.5938919999999999, 4.870868239999998, 10.313301679999999, 16.865258880000003, 23.161232239999997, 28.480073679999997, 32.43860367999999, 36.54892448, 40.59985216, 44.34139912, 47.441788079999995, 50.418386, 52.79282503999999, 55.72802840000001, 57.60690208, 59.23539112000001,60.78728656, 62.26898680000001, 63.403508079999995], [0.78254824, 3.0893919999999992, 9.4823556, 19.05750744, 29.892373040000003, 39.106751200000005, 45.84749528, 50.4539836, 54.84008544000001, 58.65662976, 62.02300167999999, 64.5484308, 66.40850456, 68.03759408, 69.56709136, 70.35083848, 71.04719144, 71.76874360000001, 72.23371216000001, 72.589088]]
yerr = [[0.07456701617756743, 0.15852966276229324, 0.31732693579621835, 0.5550691360846456, 0.5763695747852665, 0.5801294539006319, 0.7541655611702696, 0.6790676961699719, 0.6000799343535801, 0.4884083416747473, 0.37638905941656475, 0.27213576238271686, 0.29483456770315153, 0.26150139041527326, 0.23183313059897315, 0.20242174226365867, 0.1616572887679809, 0.13325410724035075, 0.12814049443136244, 0.11766674430438406], [0.17340297916593708, 0.15757172169975162, 0.4458319114385669, 0.5869247032597703, 0.7087041442629735, 0.7332448759733601, 0.7577357496704457, 0.7508383482854956, 0.7099973460501796, 0.6514936145510596, 0.6434705752971683, 0.8090876653350404, 0.6919395684540302, 0.7395457233432792, 0.5763986923328701, 0.6074968354565238, 0.5898164168222884, 0.5633484952605224, 0.557483163999602, 0.6467593581768751], [0.08560190037595194, 0.20810651612025974, 0.37180390180308764, 0.6604867748067323, 0.678907125189895, 0.6318618130588761, 0.784753220715298, 0.7272043247303096, 0.6158686717664206, 0.5032088945495367, 0.36040743208629167, 0.2872002128848765, 0.3272195367585625, 0.2783236873025531, 0.2516723366837893, 0.20532809886273343, 0.1852058148892926, 0.13802996393641512, 0.14867318213339656, 0.12859532080918193]]
y += [[0.75594864, 3.0053924800000003, 9.256356079999998, 18.612109119999996, 29.251172720000007, 38.27915183999999, 44.901695360000005, 49.41198384, 53.69448567999999, 57.43282968, 60.73520152, 63.192230959999996, 65.02850432, 66.61379399999998, 68.12129144000001, 68.88803864, 69.57159152000001, 70.27694368, 70.73251216, 71.085488], [0.34537751999999994, 1.6370919199999996, 5.0172681599999995, 10.611102079999998, 17.35485784, 23.823032719999997, 29.300473599999993, 33.376203919999995, 37.59672463999999, 41.754652320000005, 45.60779872, 48.78678831999999, 51.83638568, 54.27682480000001, 57.29802831999999, 59.23430191999999, 60.91119096000001, 62.50388647999999, 64.01638656, 65.18150784000001], [0.79894808, 3.1741918399999998, 9.77295528, 19.614307679999996, 30.766172879999992, 40.23895136, 47.1668956, 51.895583599999995, 56.38728592000001, 60.32523, 63.78320167999999, 66.37443088, 68.29090424, 69.95879400000001, 71.52829136, 72.33363848, 73.05039152, 73.78894368, 74.26391216, 74.63268792]]
yerr += [[0.07937853032677288, 0.19382481375004998, 0.3549077143265633, 0.6403395632756806, 0.6606123752066362, 0.5934093909489523, 0.742206111848147, 0.6853554771363595, 0.5952216473875743, 0.50315498313466, 0.33769422913161107, 0.2728721506020981, 0.31889992655054955, 0.26461404298003866, 0.2304233086082205, 0.19688219071257382, 0.1669703748957458, 0.12513929493485665, 0.14131057522370236, 0.1251137033878558], [0.1801691726721572, 0.15746589364728342, 0.4675458429615541, 0.5878833721039517, 0.7131785400187683, 0.7206030805306064, 0.7443145119715989, 0.7393986693994582, 0.7035649588860019, 0.6422738382784918, 0.6418928352272343, 0.8060393036447158, 0.7015363546378186, 0.7462356207995875, 0.5662914315647984, 0.6041150160477949, 0.5944018988929094, 0.5717902102334751, 0.5618860078280958, 0.6523804312862401], [0.08525625956792617, 0.19750141655918918, 0.36167289335171554, 0.6547125066923302, 0.6752926286728577, 0.6220049747816805, 0.7763954175574812, 0.7248183057313291, 0.624417842772375, 0.519356082165288, 0.37830232558764043, 0.30077177805717287, 0.336421546641437, 0.282383631080556, 0.2543421491374912, 0.20857908364104322, 0.18755836377301394, 0.14165147290995894, 0.14631048786699405, 0.12924318965173556]]

y = np.array(y);yerr = np.array(yerr)
y = y[[5,0,3,4],-n_soft:];yerr = yerr[[5,0,3,4],-n_soft:]
line_plot(x, y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea2.eps',
		'Deadline (s)','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,linestyles=linestyles,
		yerr=yerr)
y = [[73.87400000000001, 73.6360144, 73.1396468, 72.20531008, 70.78340504000002, 69.08611928, 67.2244452, 64.72221056000002, 61.78041008, 58.98359751999999, 55.088662320000005, 51.39851295999999, 47.3503852, 42.24773008, 37.57124615999999, 32.32800504, 26.669787199999995, 20.3634176, 13.8828588, 7.167713439999998, 0.004999999999999998], [73.914, 70.16805271999999, 66.50910160000001, 62.96094528, 58.924813199999996, 55.24985888000001, 51.798893199999995, 48.21914088, 44.341197040000004, 40.8306392, 37.029701360000004, 33.273346, 29.782184559999997, 25.783454799999994, 22.230500879999994, 18.710336319999996, 14.876792959999998, 11.172642719999999, 7.426099599999999, 3.8417403199999995, 0.004999999999999998], [73.914, 73.69341440000001, 73.2284468, 72.30391008000001, 70.87120504, 69.22311928, 67.3216452, 64.86221056, 61.93721008, 59.18939752, 55.28346232, 51.58651295999999, 47.5039852, 42.41093007999999, 37.76184615999999, 32.494805039999996, 26.7559872, 20.447217600000002, 13.954858799999997, 7.165513439999998, 0.004999999999999998]]
yerr = [[1.4210854715202004e-14, 0.05378184089196172, 0.06749433880259635, 0.09376958405364519, 0.2800952889788857, 0.2022591830574472, 0.19437389570750324, 0.26154374583532963, 0.2397577859189574, 0.27523516247327423, 0.39988987458472197, 0.3151057415154062, 0.3809550534919933, 0.3534218393790079, 0.5679952850499574, 0.34845284154282696, 0.39104657155149236, 0.29511082302023384, 0.25913902012447293, 0.33236631810093853, 8.673617379884035e-19], [0.0, 0.21134046508133272, 0.2853341704102587, 0.18904340145334214, 0.3059133858356641, 0.2883092011065797, 0.18034183191111133, 0.6034689293118537, 0.43849994079496785, 0.48040392624566797, 0.45488327221813507, 0.4091682115563911, 0.35046239924690203, 0.5328890492208965, 0.3511448509600707, 0.29898295378238837, 0.308661266317791, 0.334240332965885, 0.22763112037202646, 0.18285092982888973, 8.673617379884035e-19], [0.0, 0.05458673254306435, 0.05032261370795339, 0.09359467120297756, 0.2586605240158727, 0.1569936009239401, 0.205560489132788, 0.2766346276690167, 0.2650772481009289, 0.22804261722250405, 0.35715786616478973, 0.2642820260554503, 0.36038937717569935, 0.3837076306226408, 0.5460807330486789, 0.29945242399297806, 0.34521131661951143, 0.3246834038624087, 0.25619796921730664, 0.29760637423276787, 8.673617379884035e-19]]

y += [[72.38200000000002, 72.1670144, 71.7046468, 70.80731008, 69.40680504, 67.78391928, 65.91604520000001, 63.54441056, 60.66141008000001, 57.943397520000005, 54.113262320000004, 50.48291296, 46.5313852, 41.53893008000001, 36.97024616, 31.780605039999994, 26.2227872, 20.052417600000002, 13.655658799999998, 7.021313439999998, 0.004999999999999998], [75.99400000000001, 72.14225272, 68.3711016, 64.71094528, 60.57521319999999, 56.81565887999999, 53.261093200000005, 49.598140879999995, 45.60119704, 41.9694392, 38.02610136, 34.216746, 30.63378456, 26.514854799999995, 22.835500879999998, 19.23793632, 15.290592959999998, 11.49404272, 7.625699599999997, 3.9713403199999995, 0.004999999999999998], [75.99400000000001, 75.7698144, 75.2896468, 74.32931008, 72.86960504000001, 71.16351928, 69.22284520000001, 66.70341056, 63.69601007999999, 60.83719752, 56.81326232, 53.024512959999996, 48.8657852, 43.620330079999995, 38.81564616000001, 33.38800504, 27.5015872, 21.0518176, 14.333258799999996, 7.383513439999999, 0.004999999999999998]]
yerr += [[1.4210854715202004e-14, 0.049306220395890915, 0.05195920717639874, 0.09188143933131214, 0.26973602027939475, 0.16506283374238498, 0.19507133006290792, 0.24730947940096296, 0.2704550229312839, 0.262986259849614, 0.38001041950370923, 0.2718071099290337, 0.36268063833224934, 0.3785001151361002, 0.5771809808361528, 0.29595422850716335, 0.3599774286287636, 0.2878110876663358, 0.2533973822893046, 0.2993764260591914, 8.673617379884035e-19], [1.4210854715202004e-14, 0.23550906065965524, 0.303405463898906, 0.18687052788775754, 0.3358277867500548, 0.31748366199651046, 0.17999203676011802, 0.6115776398829461, 0.45616131883051025, 0.5059463929234564, 0.4794522726777711, 0.3876188993176866, 0.33138112386479573, 0.5297471418512327, 0.3541738885290248, 0.31040414151303053, 0.328001922599399, 0.3698687597583037, 0.22226331681549247, 0.17276830194016957, 8.673617379884035e-19], [1.4210854715202004e-14, 0.052801875816377714, 0.05134557751549641, 0.09328562854798866, 0.2614761626079248, 0.16487434811718257, 0.19904300593718877, 0.2585691657595225, 0.2661122634768282, 0.24117394228583097, 0.3753423839261783, 0.2754650639481779, 0.36039832938165495, 0.352774753935199, 0.5674536977331117, 0.31693047019696685, 0.3562709346581282, 0.3417168031333545, 0.25933185178660983, 0.3141300480755929, 8.673617379884035e-19]]

x = [[0.05*i for i in range(n_hard)] for _ in range(4)]
y = np.array(y);yerr = np.array(yerr)
y = y[[5,0,4,3],:n_hard];yerr = yerr[[5,0,4,3],:n_hard]
line_plot(x, y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea2.eps',
		'Failure Rate (%)','Effective Accuracy (%)',lbsize=24,linewidth=4,markersize=0,linestyles=linestyles,
		yerr=yerr)
exit(0)

x = [[5*i for i in range(21)]for _ in range(5)]
y = [[0.8957627795527155, 0.8517651757188498, 0.8090595047923321, 0.7735822683706071, 0.7221285942492013, 0.691517571884984, 0.6490095846645367, 0.6029273162939297, 0.5602416134185303, 0.5173462460063898, 0.48724640575079875, 0.4363218849840256, 0.38952476038338657, 0.3393550319488818, 0.31298322683706076, 0.2747703674121406, 0.22207468051118212, 0.18240415335463261, 0.14094049520766774, 0.09999999999999999], [0.9389436900958467, 0.9327096645367412, 0.9219668530351438, 0.9096026357827476, 0.883426517571885, 0.8675299520766775, 0.8387220447284346, 0.7953873801916932, 0.7715415335463258, 0.7225039936102237, 0.6893610223642173, 0.6380251597444089, 0.5819189297124601, 0.5162260383386582, 0.47091653354632584, 0.41250599041533553, 0.3273941693290735, 0.265491214057508, 0.18740814696485625, 0.09999999999999999], [0.9408266773162939, 0.9405591054313097, 0.9381709265175718, 0.9347224440894569, 0.9279133386581468, 0.9181908945686901, 0.9031988817891374, 0.8845347444089458, 0.8644009584664536, 0.8393170926517571, 0.801920926517572, 0.749129392971246, 0.7044189297124601, 0.6362100638977635, 0.5899420926517571, 0.5227376198083067, 0.41543730031948883, 0.3366453674121407, 0.22356230031948882, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.940023961661342, 0.9392511980830671, 0.9363278753993611, 0.9344948083067092, 0.9302615814696485, 0.9201138178913739, 0.9052875399361024, 0.8918390575079872, 0.8627715654952077, 0.8249940095846645, 0.7863258785942492, 0.7312519968051119, 0.6771405750798722, 0.6024860223642172, 0.4888238817891374, 0.39297723642172533, 0.2562579872204473, 0.09999999999999999], [0.898817891373802, 0.8929632587859425, 0.8826697284345049, 0.8703753993610224, 0.8460463258785943, 0.8304492811501596, 0.8028394568690096, 0.7618710063897763, 0.7391533546325878, 0.6926218051118211, 0.6609764376996805, 0.6121765175718851, 0.5579772364217253, 0.49667731629392964, 0.45305511182108626, 0.3973103035143771, 0.3167911341853036, 0.25841253993610225, 0.1836341853035144, 0.09999999999999999]]
yerr = [[0.006997876375020307, 0.016734437743554532, 0.012127500296505958, 0.017237607923604327, 0.01652817916156893, 0.017862496663828824, 0.019094398530911494, 0.01293758123754703, 0.02790298260476726, 0.029083235857071756, 0.019613131810753536, 0.024548452749052943, 0.012422231004442159, 0.015561299532737535, 0.020248306445012344, 0.017593245190660217, 0.013815886487961736, 0.010064554627632874, 0.00901415000465792, 0.0], [0.0020199744070912577, 0.0032324253766238334, 0.00653287651965923, 0.009172708278275014, 0.011921123855137186, 0.01059779721918944, 0.017090001119459443, 0.012361923551600719, 0.02400840721149313, 0.026234013096169042, 0.0228978001598712, 0.03175155848795646, 0.02244152268715682, 0.025468525848158535, 0.029358407348361502, 0.02099587933965674, 0.024345903249069753, 0.017092721271991466, 0.013484202410392266, 0.0], [0.0008027156549520575, 0.0010712184807357074, 0.0022035031668343617, 0.005239027964211368, 0.00393279828078102, 0.005460873192321837, 0.010205587905032077, 0.008860327405948483, 0.020381674123960886, 0.015519384629006138, 0.01613874370173752, 0.025357654777092082, 0.016296640957360668, 0.020385145055574323, 0.026988731096102322, 0.026800322050731698, 0.024095142805632887, 0.017292520880111212, 0.01813718868803247, 0.0], [0.0, 0.0, 0.0024308623529587818, 0.0024002403129800703, 0.0020382685888009405, 0.004076723499975016, 0.004875208983659424, 0.003370617083741069, 0.005564876243903203, 0.0051922542858615466, 0.00958072502904603, 0.019763452711440668, 0.016496599841994884, 0.019692192854194834, 0.02522283850573193, 0.022579075887578987, 0.024949860614209174, 0.012598416304351604, 0.020184203882597448, 0.0], [0.001927342112372416, 0.003179089331609038, 0.006199251512525477, 0.008842385139736059, 0.01141126165629694, 0.01041494363648053, 0.01664410498867549, 0.011930115136505527, 0.023236953811564296, 0.025148276409804056, 0.021797757967920994, 0.03150124050809064, 0.022916120965365556, 0.024505531034889692, 0.028665699102147366, 0.0206405564153535, 0.022501503135747496, 0.016330672689323523, 0.012477112118501896, 0.0]]
y = [[0.9412]+l for l in y]
yerr = [[0]+l for l in yerr]
y = np.array(y)[[0,1,2,3,4]]*100;yerr = np.array(yerr)[[0,1,2,3,4]]*100
line_plot(x, y,['Base',r'Base$\times$2',r'Base$\times$3',r'Base$\times$4','Split'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/mot0.eps',
		'Failure Rate (%)','Effective Accuracy (%)',lbsize=20,use_re_label=True,markersize=8,ylim=(0,100),lgsize=19)
exit(0)
x0 = np.array([[0.0625*2*i for i in range(1,5)] for _ in range(2)])
x = 100-x0*100
methods_tmp = [f'ResNet{v}' for v in [56,50]]
y = np.array([2.03125/0.03125,40.099609375/.729736328125]).reshape(2,1)
y = np.repeat(y,4,axis=1)
y = y / (x0 * 2)
line_plot(x,y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/comm_cost.eps',
		'Partition Ratio (%)','Comm. Cost Reduction Ratio',lbsize=24,linewidth=8,markersize=16,linestyles=linestyles,
		use_comm_annot=True)

x0 = np.array([0.1*i for i in range(11)])
x = [(1-x0)*100]
y = [(1-x0**2-(1-x0)**2)*100]
line_plot(x,y,[''],colors,
				f'/home/bo/Dropbox/Research/SIGCOMM23/images/conn_loss.eps',
				'Partition Ratio (%)','Connection Loss (%)',lbsize=36,linewidth=8,markersize=16,ncol=0,use_connarrow=True)

# cifar breakdown
latency_types = ['NM','DCN','NR','INF','WAN']	
labels = ['Ours','Base', 'Optimal']
latency_breakdown_mean = [[0.0048398783412604285, 0.014246439476907482, 0.00023509953349543075,0, 0.7486890470647757],
[0,0,0,0.00483251379701657, 1.0169146132483973],
[0,0,0,0.00483251379701657, 0.7486877294765364],]

latency_breakdown_std = [[0.00021182092754556427, 0.0011024832600207727, 4.049920186137311e-05,0, 0.34634288409670605],
[0,0,0,0.0002204459821230588, 0.5684324923094014],
[0,0,0,0.0002204459821230588, 0.34634275598435527],]

plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
	'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_breakdown_cifar.eps',5,(0.5,0.76),title_posy=0.26,ratio=0.6)


# imagenet latency breakdown
latency_breakdown_mean = [
[0.019284586669921874, 0.06311613967338686, 0.014764462451934815,0, 11.01427887952024],
[0,0,0,0.018242218704223632, 15.481805917434501],
[0,0,0,0.018242218704223632, 11.0142774438325],]

latency_breakdown_std = [
[0.001057063805851016, 0.0015932168873451931, 0.007375720249399674,0, 4.517442117207892],
[0,0,0,0.007524489114244404, 8.696532160000752],
[0,0,0,0.007524489114244404, 4.517442844121096],
]
plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
	'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_breakdown_imagenet.eps',5,(0.5,0.76),title_posy=0.26,lim1=100,lim2=16000,ratio=0.6)

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
latency_types = ['NM','DCN','NR','WAN']	
labels = ['1','2','4','8','16','32','64']
plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
	'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_breakdown_vs_batch.eps',4,(0.5,0.83),title_posy=0.2,lim1=25)

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
methods_tmp = ['Ours','w/o MT','w/o CT','w/o CT+MT']
y1 = np.concatenate((re_base[:,0].reshape((1,8)),
					re_nobridge[:,0].reshape((1,8)),
					re_noonn[:,0].reshape((1,8)),
					re_nolab[:,0].reshape((1,8))))*100
line_plot(x,y1,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/ablation_soft.eps',
		'FLOPS (%)','Reliability (%)',lbsize=24,linewidth=0,markersize=16,linestyles=linestyles,
		lgsize=20,bbox_to_anchor=(.48,.49),ncol=1,use_arrow=True,arrow_coord=(65,0.8))	
# y2 = np.concatenate((re_base[:,1].reshape((1,8)),
# 					re_nobridge[:,1].reshape((1,8)),
# 					re_noonn[:,1].reshape((1,8)),
# 					re_nolab[:,1].reshape((1,8))))*100
# line_plot(x,y2,methods_tmp,colors,
# 		'/home/bo/Dropbox/Research/SIGCOMM23/images/ablation_hard.eps',
# 		'FLOPS (%)','Reliability (%)',lbsize=28,linewidth=0,markersize=16,linestyles=linestyles,
# 		lgsize=20,use_arrow=True,arrow_coord=(80,6))	

# different convs
re_vs_conv = [[0.011969848242811509,0.00584778639890463],
[0.010570087859424948,0.0060427126122014315],
[0.011802116613418533,0.0046021603529590805],
  [0.010894568690095847,0.004835120949338205],
[0.010889576677316274, 0.004088696181347941],
# [0.012602835463258813,0.006356496272630458]
]
flops_vs_conv = [0.5298202593545005,0.558022230677192,0.5862242019998837,0.6003,0.642628144645267]#,0.6990320872906503]
bridge_size = [64,96,128,144,192]#,256]
re_vs_conv = np.array(re_vs_conv)
re_vs_conv[:,0] *= re_vs_conv[3,0]/re_vs_conv[0,0]
re_vs_conv[:,1] *= re_vs_conv[3,1]/re_vs_conv[0,1]
y = re_vs_conv*100
envs = [f'{bridge_size[i]}\n{round(flops_vs_conv[i]*200)}%' for i in range(5)]
groupedbar(y,None,'Reliability (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_conv.eps',methods=['NS','FS'],envs=envs,labelsize=18,
	ncol=1,sep=1,bbox_to_anchor=(0.5, 1.2),width=0.4,xlabel='Bridge Size and FLOPS (%)',legloc='center right')

# different sampling rates
re_vs_sr = [[0.011012380191693292,0.003109310816978563],
[0.009865215654952075,0.0028192986459759715],
[0.009728434504792351,0.00456412596987678],
[0.008887779552715653,0.0027384755819260635],
[0.010894568690095847,0.0042408337136771845],]

# flops_vs_sr = [1.1925665854377538,0.8964458865494916,0.7483855371053606,0.6743553623832951,0.6003]
flops_vs_sr = [0.7930,0.6614,0.5956,0.5627,0.5298]
sample_interval = [9,5,3,2,1]
re_vs_sr = np.array(re_vs_sr)
y = re_vs_sr*100
envs = [f'{sample_interval[i]}/9\n{round(flops_vs_sr[i]*200)}%' for i in range(5)]
groupedbar(y,None,'Reliability (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_sr.eps',methods=['NS','FS'],envs=envs,labelsize=18,
	ncol=1,sep=1,width=0.4,xlabel='Sampling Interval and FLOPS (%)',legloc='center left')

# different partition ratios
diff_ratio_data = [[0.006060303514377014, -0.001283660429027841],
[0.007938298722044735,2.3771489426446934e-05],
[0.0085513178913738,0.0013074319184542707],
[0.011028354632587872,0.002904876007911146],
[0.010889576677316302,0.004559371671991471],
[0.010039936102236427,0.006133044272021897],
[0.013776956869009583,0.010430929560322535],
[0.012270367412140598,0.01332154267457781]
]
diff_ratio_data = np.array(diff_ratio_data)*100
y = diff_ratio_data
# flops_base = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
flops_base = [0.9355,0.8224,0.7171,0.6196,0.5298,0.4479,0.3737,0.3073]
envs = [f'{round(100-6.25*i)}%\n{round(flops_base[i-1]*200)}%' for i in range(1,9)]
groupedbar(y,None,'Reliability (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_ratio.eps',methods=['NS','FS'],envs=envs,labelsize=18,
	ncol=1,sep=1,width=0.4,xlabel='Partition Ratio (%) and FLOPS (%)',legloc='upper left')

envs = ['R20','R32','R44','R56','R110']
methods_tmp = ['Base','CR','Ours']
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
acc_sp = [0.8869808306709265,0.90435303514377,0.908047124600639,0.9008,0.9195287539936102]
acc_sp = np.array(acc_sp)
flops_sp = [0.2527390666182668,0.2516129511941694,0.25114302010213746,0.25088513674100354,0.250439214753364]
y = np.concatenate((acc_base.reshape(5,1),acc_sp.reshape(5,1),acc_par_mean.reshape(5,1)),axis=1)*100
yerr = np.concatenate((np.zeros((5,1)),np.zeros((5,1)),acc_par_std.reshape(5,1)),axis=1)*100
groupedbar(y,yerr,'FFA (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/acc_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,width=.25,sep=1,legloc='best',ylim=(85,95),)

soft = [[0.12374700479233228,0.026222044728434514,0.006991813099041515],
[0.12570587060702873, 0.019972044728434522,0.009653554313099012],
[0.12668630191693292, 0.02085463258785944,0.012633785942492015],
[0.12740115814696484, 0.026772164536741177,0.010894568690095847],
[0.12879093450479234, 0.01743610223642173,0.015284544728434536]
]

y = np.array(soft)*100
groupedbar(y,None,'Reliability (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/soft_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,width=.25,sep=1,legloc='center left',use_barlabe_y=True,)

hard = [[0.1316902479841777,0.02511220143009282,0.0019730336223946314],
[0.13636752624372434,0.01973033622394643,0.005381865206146362],
[0.13509242355089, 0.01987771945839038,0.007630648105887723],
[0.13641316750342308, 0.027132778031340316,0.004649703331811962],
[0.14010250266240679, 0.016777917237182412,0.00557203712155789]]

y = np.array(hard)*100
groupedbar(y,None,'Reliability (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,width=.25,sep=1,legloc='center left',use_barlabe_y=True,)
for name in ['resnet50','resnet56-2','resnet56-3','resnet56-4']:
	numsn = 4
	colors_tmp = colors
	if name == 'resnet56-3':
		numsn = 6
	elif name == 'resnet56-4':
		numsn = 8
		colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
	all_acc = [[] for _ in range(numsn)]
	all_epoch = [[] for _ in range(numsn)]
	if name == 'resnet50':
		lines = 0
	else:
		lines = 10
	with open(f'/home/bo/Dropbox/Research/SIGCOMM23/images/{name}.log','r') as f:
		for line in f.readlines():
			line = line.split(' ')
			epoch = eval(line[0])
			if name != 'resnet50' and ((epoch<10) or (epoch >=60 and epoch < 320) or (epoch >= 380 and epoch < 440) or\
				 (epoch >=480 and epoch < 500) or (epoch>=540)):continue
			line = line[1].strip().split(',')[2:]
			if name != 'resnet50':
				accuracy_list = [eval(n.split('(')[0])*100 for n in line if len(n)>=6]
			else:
				accuracy_list = [eval(n.split('(')[0]) for n in line if len(n)>=6]
			for i,n in enumerate(accuracy_list):
				all_acc[i] += [n]
				all_epoch[i] += [lines]
			lines += 1
	methods_tmp = [f'SN#{i}' for i in range(numsn)]
	if name != 'resnet50':
		xticks = [10,60,120,160,200]
	else:
		xticks = [0,40,80,120,160]
	ncol = 2 if name == 'resnet56-4' else 1
	line_plot(all_epoch,all_acc,methods_tmp,colors_tmp,
			f'/home/bo/Dropbox/Research/SIGCOMM23/images/{name}.eps',
			'Epoch','Test Accuracy (%)',markersize=0,linewidth=4,xticks=xticks,legloc='best',lbsize=24,ncol=ncol,
			use_resnet56_2arrow=(name == 'resnet56-2'),use_resnet56_3arrow=(name == 'resnet56-3'),
			use_resnet56_4arrow=(name == 'resnet56-4'),use_resnet50arrow=(name == 'resnet50'))

analyze_all_recorded_traces()

x0 = np.array([0.1*i for i in range(11)])
x = [[0.1*i for i in range(11)] for _ in range(2)]
for sn in [2,3,4]:
	methods_tmp = ['Two+','One']
	one = sn*(1-x0)*x0**(sn-1)
	twoormore = 1-x0**sn-one
	y = np.stack((twoormore,one),axis=0)
	if sn==2:
		line_plot(x,y,methods_tmp,colors,
				f'/home/bo/Dropbox/Research/SIGCOMM23/images/prob{sn}.eps',
				'Failure Rate (%)','Probability (%)',lbsize=36,linewidth=8,markersize=16,bbox_to_anchor=(0.3,.45),ncol=1,
				linestyles=linestyles,legloc='best',use_probarrow=True,xticks=[0.2*i for i in range(6)],yticks=[0.2*i for i in range(6)])
	else:
		line_plot(x,y,methods_tmp,colors,
				f'/home/bo/Dropbox/Research/SIGCOMM23/images/prob{sn}.eps',
				'Failure Rate (%)','',lbsize=36,linewidth=8,markersize=16,linestyles=linestyles,legloc='best',use_probarrow=True,ncol=0,
				xticks=[0.2*i for i in range(6)],yticks=[0.2*i for i in range(6)])
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
# 	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_imagenet.eps',methods=methods_tmp,
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
# filename = '/home/bo/Dropbox/Research/SIGCOMM23/images/flops_dist.eps'
# plot_computation_dist(flops,labels,filename,horizontal=False)
# # 77.09%: resnet-50
