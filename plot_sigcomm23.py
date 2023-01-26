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


def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,legloc='best',linestyles=linestyles,
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(0.4,30),markersize=8,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,use_doublearrow=False,
				rotation=None,use_resnet56_2arrow=False,use_resnet56_3arrow=False,use_resnet56_4arrow=False,use_resnet50arrow=False,use_re_label=False):
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
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=-45, size=lbsize-8,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	if use_doublearrow:
		ax.annotate(text='', xy=(10,81), xytext=(10,18), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    7, 48, "4.5X more likely", ha="center", va="center", rotation='vertical', size=lbsize,fontweight='bold')
	if use_re_label:
		baselocs = [];parlocs = []
		for i in range(2,5):
			baselocs += [np.argmax(y[i]-y[0])]
			parlocs += [np.argmax(y[i]-y[1])]
		for k,locs in enumerate([baselocs,parlocs]):
			for i,loc in enumerate(locs):
				ax.annotate(text='', xy=(XX[0][loc],YY[k,loc]), xytext=(XX[0][loc],YY[i+2,loc]), arrowprops=dict(arrowstyle='|-|',lw=5-k*2,color=color[k]))
				h = YY[k,loc]-5 if k==0 else YY[i+2,loc]+4
				w = XX[0][loc]-3 if k==0 else XX[0][loc]
				if i==0:
					ax.text(w, h, '2nd', ha="center", va="center", rotation='horizontal', size=16,fontweight='bold',color=color[k])
				elif i==1:
					ax.text(w, h, '3rd', ha="center", va="center", rotation='horizontal', size=16,fontweight='bold',color=color[k])
				elif i==2:
					ax.text(w, h, '4th', ha="center", va="center", rotation='horizontal', size=16,fontweight='bold',color=color[k])
	if use_resnet56_2arrow:
		ax.annotate(text='', xy=(10,78), xytext=(60,78), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    35, 79, "Train larger SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(60,78), xytext=(200,78), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    130, 79, "Train all SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		for l,r,lr in [(10,60,0.1),(60,120,0.02),(120,160,0.004),(160,200,0.0008)]:
			ax.annotate(text='', xy=(l,81), xytext=(r,81), arrowprops=dict(arrowstyle='<->',lw=linewidth))
			ax.text(
			    (l+r)/2, 82, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if use_resnet56_3arrow:
		ax.annotate(text='', xy=(10,75), xytext=(60,75), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    35, 76, "Train larger SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(60,85), xytext=(200,85), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    130, 86, "Train all SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		for l,r,lr in [(10,60,0.1),(60,120,0.02),(120,160,0.004),(160,200,0.0008)]:
			h = 78 if l<160 else 87
			ax.annotate(text='', xy=(l,h), xytext=(r,h), arrowprops=dict(arrowstyle='<->',lw=linewidth))
			ax.text(
			    (l+r)/2, h+1, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if use_resnet56_4arrow:
		ax.annotate(text='', xy=(10,73), xytext=(60,73), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    35, 74, "Train larger SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(60,85), xytext=(200,85), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    130, 86, "Train all SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		for l,r,lr in [(10,60,0.1),(60,120,0.02),(120,160,0.004),(160,200,0.0008)]:
			h = 77 if l<160 else 87
			ax.annotate(text='', xy=(l,h), xytext=(r,h), arrowprops=dict(arrowstyle='<->',lw=linewidth))
			ax.text(
			    (l+r)/2, h+1, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if use_resnet50arrow:
		ax.annotate(text='', xy=(0,62), xytext=(40,62), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    20, 63, "Train larger SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		ax.annotate(text='', xy=(40,62), xytext=(160,62), arrowprops=dict(arrowstyle='<->',lw=linewidth))
		ax.text(
		    100, 63, "Train all SNs", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
		for l,r,lr in [(0,40,0.1),(40,80,0.1),(80,120,0.01),(120,160,0.001)]:
			ax.annotate(text='', xy=(l,64), xytext=(r,64), arrowprops=dict(arrowstyle='<->',lw=linewidth))
			ax.text(
			    (l+r)/2, 64.5, f"lr={lr}", ha="center", va="center", rotation='horizontal', size=lbsize,fontweight='bold')
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lbsize)
		else:
			if bbox_to_anchor is None:
				plt.legend(loc=legloc,fontsize = lbsize,ncol=ncol)
			else:
				plt.legend(loc=legloc,fontsize = lbsize,ncol=ncol,bbox_to_anchor=bbox_to_anchor)
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
				label=latency_types[i], hatch=hatches[i], align='center')
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
    trace_filenames += [f'../WAN/{12*i:06d}' for i in [1,2,4,8,16,32,64]]
    trace_filenames += [f'../WAN-768/{768*i:06d}' for i in [1,2,4,8,16,32,64]]
    latency_mean_list = []
    latency_std_list = []
    trpt_mean_list = []
    trpt_std_list = []
    all_latency_list = []
    for tidx,filename in enumerate(trace_filenames):
        latency_list = []
        bandwidth_list = []
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
        for row in csv_reader:
            if float(row["latency"])>1e7 or float(row["latency"])<0 or float(row["downthrpt"])>1e8:
                continue
            for bs in [2**i for i in range(7)]:
                query_size = 3*32*32*4*bs # bytes
                latency_list += [query_size/float(row["downthrpt"]) + float(row["latency"])/1e6]
                query_size = 3*256*256*4*bs
                latency224_list += [query_size/float(row["downthrpt"]) + float(row["latency"])/1e6]
            num_of_line += 1
            if num_of_line==10000:break
        all_latency_list += np.array(latency_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
        all_latency_list += np.array(latency224_list).reshape((num_of_line,7)).transpose((1,0)).tolist()
    all_latency_list = np.array(all_latency_list)
    relative_latency = all_latency_list[14:,:].copy()
    for b in range(7):
    	for start in [0,14]:
    		relative_latency[start+b] /= all_latency_list[b]
    	for start in [7,21]:
    		relative_latency[start+b] /= all_latency_list[7+b]
    relative_latency = np.log10(relative_latency)
    relative_latency = relative_latency.reshape((4,7,10000))
    relative_latency = relative_latency[2:]
    labels = ['CIFAR-10','ImageNet']
    x = [[2**(i) for i in range(7)] for _ in range(len(labels))]
    y = relative_latency.mean(axis=-1)
    yerr = relative_latency.std(axis=-1)
    line_plot(x, y,labels,colors,'/home/bo/Dropbox/Research/SIGCOMM23/images/wdlr_vs_bs.eps','Query Batch Size','WDLR',
    	yerr=yerr,yticks=[1,2,3],yticklabel=[10,100,1000])	

    # throughput = all_latency_list[14:,:].copy()
    # throughput = 1/throughput 
    # throughput = throughput.reshape((4,7,100,100)).mean(axis=-1)
    # throughput *= np.array([2**i for i in range(7)]).reshape((1,7,1))
    # throughput = np.log10(throughput)
    # y = throughput.mean(axis=-1)
    # yerr = throughput.std(axis=-1)
    # line_plot(x, y,labels,colors,'/home/bo/Dropbox/Research/SIGCOMM23/images/throughput_vs_bs.eps','Query Batch Size','Query Throughput (1/s)',
    # 	yerr=yerr,yticks=[1,2,3],yticklabel=[10,100,1000])	

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
				use_downarrow=False,rotation=None):
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
					ax.text(xdx-0.07,data_mean[k,i]+.03,f'{data_mean[k,i]:.4f}',fontsize = 18, rotation='vertical',fontweight='bold')
		if use_barlabe_y and i==2:
			for k,xdx in enumerate(x_index):
				ax.text(xdx-0.08,data_mean[k,i]+.01,f'{data_mean[k,i]:.4f}',fontsize = 18, rotation='vertical',fontweight='bold')
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
			           loc='upper center', ncol=ncol, fontsize=labelsize)
		else:
			plt.legend(fancybox=True,
			           loc=legloc, ncol=ncol, fontsize=labelsize)
	fig.savefig(path, bbox_inches='tight')
	plt.close()

# baseline
flops_base = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
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
flops_noonn = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
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


x = np.array([flops_base,flops_nobridge,flops_noonn,flops_nolab])*100
# Model Transform; Collaborative Training
methods_tmp = ['Ours (w/ C+T)','w/o Transform','w/o Collaboration','Standalone (w/o C+T)']
y = np.concatenate((re_base[:,0].reshape((1,8)),
					re_nobridge[:,0].reshape((1,8)),
					re_noonn[:,0].reshape((1,8)),
					re_nolab[:,0].reshape((1,8))))*100
line_plot(x,y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/ablation_soft.eps',
		'FLOPS (%)','Reliability (%)',lbsize=16,linewidth=4,markersize=8,linestyles=linestyles,)	
y = np.concatenate((re_base[:,1].reshape((1,8)),
					re_nobridge[:,1].reshape((1,8)),
					re_noonn[:,1].reshape((1,8)),
					re_nolab[:,1].reshape((1,8))))*100
line_plot(x,y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/ablation_hard.eps',
		'FLOPS (%)','Reliability (%)',lbsize=16,linewidth=4,markersize=8,linestyles=linestyles,)	
exit(0)
x = [[5*i for i in range(21)]for _ in range(5)]
y = [[0.8957627795527155, 0.8517651757188498, 0.8090595047923321, 0.7735822683706071, 0.7221285942492013, 0.691517571884984, 0.6490095846645367, 0.6029273162939297, 0.5602416134185303, 0.5173462460063898, 0.48724640575079875, 0.4363218849840256, 0.38952476038338657, 0.3393550319488818, 0.31298322683706076, 0.2747703674121406, 0.22207468051118212, 0.18240415335463261, 0.14094049520766774, 0.09999999999999999], [0.9389436900958467, 0.9327096645367412, 0.9219668530351438, 0.9096026357827476, 0.883426517571885, 0.8675299520766775, 0.8387220447284346, 0.7953873801916932, 0.7715415335463258, 0.7225039936102237, 0.6893610223642173, 0.6380251597444089, 0.5819189297124601, 0.5162260383386582, 0.47091653354632584, 0.41250599041533553, 0.3273941693290735, 0.265491214057508, 0.18740814696485625, 0.09999999999999999], [0.9408266773162939, 0.9405591054313097, 0.9381709265175718, 0.9347224440894569, 0.9279133386581468, 0.9181908945686901, 0.9031988817891374, 0.8845347444089458, 0.8644009584664536, 0.8393170926517571, 0.801920926517572, 0.749129392971246, 0.7044189297124601, 0.6362100638977635, 0.5899420926517571, 0.5227376198083067, 0.41543730031948883, 0.3366453674121407, 0.22356230031948882, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.940023961661342, 0.9392511980830671, 0.9363278753993611, 0.9344948083067092, 0.9302615814696485, 0.9201138178913739, 0.9052875399361024, 0.8918390575079872, 0.8627715654952077, 0.8249940095846645, 0.7863258785942492, 0.7312519968051119, 0.6771405750798722, 0.6024860223642172, 0.4888238817891374, 0.39297723642172533, 0.2562579872204473, 0.09999999999999999], [0.898817891373802, 0.8929632587859425, 0.8826697284345049, 0.8703753993610224, 0.8460463258785943, 0.8304492811501596, 0.8028394568690096, 0.7618710063897763, 0.7391533546325878, 0.6926218051118211, 0.6609764376996805, 0.6121765175718851, 0.5579772364217253, 0.49667731629392964, 0.45305511182108626, 0.3973103035143771, 0.3167911341853036, 0.25841253993610225, 0.1836341853035144, 0.09999999999999999]]
yerr = [[0.006997876375020307, 0.016734437743554532, 0.012127500296505958, 0.017237607923604327, 0.01652817916156893, 0.017862496663828824, 0.019094398530911494, 0.01293758123754703, 0.02790298260476726, 0.029083235857071756, 0.019613131810753536, 0.024548452749052943, 0.012422231004442159, 0.015561299532737535, 0.020248306445012344, 0.017593245190660217, 0.013815886487961736, 0.010064554627632874, 0.00901415000465792, 0.0], [0.0020199744070912577, 0.0032324253766238334, 0.00653287651965923, 0.009172708278275014, 0.011921123855137186, 0.01059779721918944, 0.017090001119459443, 0.012361923551600719, 0.02400840721149313, 0.026234013096169042, 0.0228978001598712, 0.03175155848795646, 0.02244152268715682, 0.025468525848158535, 0.029358407348361502, 0.02099587933965674, 0.024345903249069753, 0.017092721271991466, 0.013484202410392266, 0.0], [0.0008027156549520575, 0.0010712184807357074, 0.0022035031668343617, 0.005239027964211368, 0.00393279828078102, 0.005460873192321837, 0.010205587905032077, 0.008860327405948483, 0.020381674123960886, 0.015519384629006138, 0.01613874370173752, 0.025357654777092082, 0.016296640957360668, 0.020385145055574323, 0.026988731096102322, 0.026800322050731698, 0.024095142805632887, 0.017292520880111212, 0.01813718868803247, 0.0], [0.0, 0.0, 0.0024308623529587818, 0.0024002403129800703, 0.0020382685888009405, 0.004076723499975016, 0.004875208983659424, 0.003370617083741069, 0.005564876243903203, 0.0051922542858615466, 0.00958072502904603, 0.019763452711440668, 0.016496599841994884, 0.019692192854194834, 0.02522283850573193, 0.022579075887578987, 0.024949860614209174, 0.012598416304351604, 0.020184203882597448, 0.0], [0.001927342112372416, 0.003179089331609038, 0.006199251512525477, 0.008842385139736059, 0.01141126165629694, 0.01041494363648053, 0.01664410498867549, 0.011930115136505527, 0.023236953811564296, 0.025148276409804056, 0.021797757967920994, 0.03150124050809064, 0.022916120965365556, 0.024505531034889692, 0.028665699102147366, 0.0206405564153535, 0.022501503135747496, 0.016330672689323523, 0.012477112118501896, 0.0]]
y = [[0.9412]+l for l in y]
yerr = [[0]+l for l in yerr]
y = np.array(y)[[0,4,1,2,3]]*100;yerr = np.array(yerr)[[0,4,1,2,3]]*100
line_plot(x, y,['Se.','St.',r'Rn.($\times$2)',r'Rn.($\times$3)',r'Rn.($\times$4)'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/mot0.eps',
		'Failure Rate (%)','Effective Accuracy (%)',lbsize=20,use_re_label=True,markersize=8,ylim=(0,100))
exit(0)

# # 1. loss brings down accuracy
# x = [[5*i for i in range(6)]for _ in range(4)]
# y = [[94.12,89.984824,85.529952, 81.327276,75.978435,72.654353],
# [94.12,93.84984,93.394968,92.176917,91.282947,89.128594],
# [91.53,87.02364624000002, 82.06531632, 77.92634296000001, 73.15500176000002, 68.793842],
# [91.53,91.33101080000003, 90.60644991999999, 89.49291112, 87.95659496, 85.86170872]] 

# yerr = [[0,0.779019, 1.106963, 1.908686, 2.466583, 0.698803],
# [0,0.212719,0.429682, 0.626182, 0.943877, 1.152293],
# [0,0.22001012256037583, 0.31734498384907517, 0.36730227485946276, 0.42945146053601585,0.326857638718318],
# [0,0.05626240955778068, 0.09197665318038978, 0.17145129561051317, 0.24627422058304022,0.29693415323216665]]
# line_plot(x, y,['CIFAR-10 (se.)','CIFAR-10 (rn.)','ImageNet (se.)','ImageNet (rn.)'],colors,
# 		'/home/bo/Dropbox/Research/SIGCOMM23/images/mot1.eps',
# 		'Failure Rate (%)','Effective Accuracy (%)',yerr=yerr,lbsize=20)
# # 2. replication helps but hurts flops
# # replication makes sure consistent performance with one or two servers
# # this is not necessary
# # with even 10% loss, the chance is 4.5X
# # simply with pruning does not change this fact, we want to exploit beyond replication: partitioning
# # 
# x = [[10*i for i in range(11)]for _ in range(3)]
# y = [[(0.1*i)**2*100 for i in range(11)],
# [(1-.1*i)*(2*0.1*i)*100 for i in range(11)],
# [(1-.1*i)**2*100 for i in range(11)]
# ]
# line_plot(x, y,['None alive','One alive','Two alive'],colors,
# 		'/home/bo/Dropbox/Research/SIGCOMM23/images/mot2.eps',
# 		'Failure Rate (%)','Probability (%)',use_doublearrow=True,lbsize=20)
# # 3. partition
# # two capacity, different accuracy cdf
# # our_correctness = [[1.0, 0.9375, 0.96875, 0.9375, 0.9375, 0.96875, 0.96875, 0.875, 0.9375, 0.90625, 0.90625, 0.96875, 0.90625, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.875, 0.9375, 1.0, 0.96875, 0.90625, 0.96875, 0.875, 0.9375, 0.96875, 0.9375, 0.9375, 1.0, 0.9375, 0.9375, 1.0, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.90625, 0.96875, 0.96875, 0.96875, 0.90625, 1.0, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.9375, 1.0, 0.9375, 0.9375, 0.96875, 0.90625, 0.90625, 0.96875, 0.84375, 0.90625, 0.96875, 1.0, 0.90625, 0.90625, 1.0, 0.9375, 1.0, 0.875, 0.96875, 0.96875, 0.9375, 1.0, 0.9375, 0.90625, 0.875, 0.9375, 0.96875, 0.875, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.96875, 0.9375, 0.96875, 0.96875, 0.875, 0.96875, 0.9375, 0.78125, 0.90625, 0.9375, 0.9375, 0.84375, 0.90625, 1.0, 1.0, 1.0, 0.9375, 0.84375, 0.96875, 0.90625, 1.0, 0.96875, 0.96875, 0.96875, 0.9375, 0.875, 0.96875, 0.96875, 0.90625, 0.96875, 0.9375, 0.96875, 0.9375, 1.0, 0.9375, 0.96875, 0.96875, 0.9375, 0.90625, 1.0, 0.9375, 0.90625, 1.0, 0.96875, 0.90625, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.9375, 0.90625, 1.0, 0.96875, 0.9375, 1.0, 1.0, 1.0, 0.9375, 0.875, 0.875, 0.90625, 0.9375, 0.96875, 0.90625, 0.9375, 0.9375, 0.90625, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.9375, 0.96875, 1.0, 0.90625, 0.90625, 0.90625, 0.90625, 0.875, 0.9375, 0.96875, 0.96875, 0.84375, 0.96875, 0.96875, 0.90625, 0.84375, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.90625, 0.90625, 0.9375, 0.9375, 0.96875, 0.875, 1.0, 0.96875, 0.96875, 0.9375, 0.96875, 0.90625, 1.0, 0.96875, 0.90625, 0.875, 0.9375, 0.875, 0.9375, 0.9375, 1.0, 0.9375, 1.0, 0.9375, 1.0, 0.96875, 1.0, 1.0, 0.96875, 0.96875, 0.9375, 0.96875, 0.875, 0.96875, 0.96875, 1.0, 0.9375, 0.90625, 0.9375, 0.96875, 0.84375, 0.90625, 0.9375, 0.96875, 0.90625, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 0.96875, 0.96875, 0.90625, 0.9375, 0.90625, 0.9375, 0.90625, 0.84375, 0.96875, 0.9375, 0.96875, 1.0, 0.9375, 0.96875, 0.96875, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625, 0.96875, 0.9375, 1.0, 0.9375, 0.90625, 0.9375, 1.0, 0.9375, 0.96875, 1.0, 0.96875, 0.9375, 0.9375, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625, 0.90625, 0.9375, 0.96875, 0.9375, 1.0, 0.90625, 1.0, 0.96875, 0.96875, 0.9375, 0.96875, 0.90625, 0.9375, 0.9375, 0.9375, 0.90625, 0.90625, 0.96875, 0.875, 0.96875, 0.90625, 0.9375, 0.9375, 0.9375, 0.90625, 1.0, 0.9375, 0.9375, 0.9375, 0.8125, 0.90625, 0.9375, 0.84375, 0.90625, 0.84375, 0.96875, 0.96875, 0.875, 0.90625, 1.0, 0.96875, 0.875], [0.96875, 0.9375, 0.96875, 1.0, 0.90625, 1.0, 0.96875, 0.875, 0.9375, 0.875, 0.875, 0.8125, 0.90625, 0.90625, 0.9375, 0.96875, 1.0, 0.9375, 0.84375, 0.90625, 0.875, 0.9375, 1.0, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.90625, 0.96875, 0.9375, 0.84375, 0.78125, 0.9375, 0.9375, 0.90625, 0.9375, 0.96875, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 1.0, 0.96875, 0.96875, 0.9375, 0.9375, 0.84375, 0.90625, 0.875, 0.90625, 0.96875, 0.96875, 0.90625, 0.9375, 1.0, 0.96875, 0.9375, 0.9375, 0.96875, 0.9375, 0.96875, 0.96875, 0.96875, 0.9375, 0.96875, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.9375, 0.96875, 0.90625, 0.90625, 0.9375, 0.96875, 0.90625, 0.9375, 0.90625, 0.90625, 0.875, 0.90625, 0.96875, 1.0, 0.90625, 0.96875, 0.9375, 0.96875, 0.96875, 0.90625, 0.90625, 0.875, 0.9375, 0.875, 0.96875, 0.96875, 0.90625, 0.9375, 0.9375, 0.84375, 0.96875, 0.96875, 0.90625, 0.9375, 1.0, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.8125, 0.9375, 0.9375, 0.9375, 0.9375, 0.96875, 0.875, 0.875, 0.90625, 0.96875, 1.0, 0.96875, 0.9375, 0.96875, 0.96875, 0.9375, 0.9375, 0.9375, 0.96875, 0.875, 0.9375, 1.0, 0.84375, 0.96875, 0.90625, 0.90625, 0.9375, 0.96875, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.96875, 0.90625, 0.90625, 1.0, 0.90625, 1.0, 0.875, 0.875, 0.875, 0.90625, 0.96875, 0.9375, 0.96875, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.90625, 0.90625, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.9375, 0.96875, 1.0, 0.96875, 1.0, 0.9375, 0.96875, 0.9375, 0.90625, 0.96875, 1.0, 1.0, 0.96875, 1.0, 0.90625, 0.90625, 0.96875, 0.9375, 0.90625, 0.96875, 0.9375, 0.96875, 0.90625, 1.0, 0.84375, 0.9375, 0.8125, 0.9375, 0.96875, 0.9375, 0.96875, 1.0, 0.9375, 0.875, 1.0, 0.96875, 1.0, 0.9375, 1.0, 0.90625, 0.96875, 0.9375, 0.875, 0.96875, 0.96875, 0.90625, 1.0, 0.96875, 0.90625, 0.96875, 1.0, 0.875, 0.875, 0.96875, 0.84375, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.875, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 1.0, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.96875, 1.0, 0.875, 0.9375, 0.9375, 0.875, 0.96875, 1.0, 1.0, 0.9375, 0.875, 0.96875, 0.90625, 0.90625, 0.90625, 0.90625, 0.96875, 0.9375, 0.9375, 0.9375, 0.96875, 0.9375, 0.96875, 1.0, 1.0, 1.0, 1.0, 0.96875, 0.96875, 0.96875, 1.0, 0.9375, 0.96875, 0.9375, 0.96875, 0.875, 0.9375, 0.875, 0.96875, 0.96875, 0.96875, 0.9375, 1.0, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 0.9375, 1.0, 0.90625, 0.90625, 1.0, 0.9375, 0.96875, 0.90625, 0.9375], [0.875, 0.96875, 0.9375, 0.9375, 0.96875, 0.90625, 0.9375, 0.84375, 0.96875, 0.96875, 1.0, 0.9375, 0.9375, 0.9375, 0.9375, 0.875, 1.0, 0.875, 0.9375, 0.96875, 0.9375, 1.0, 0.9375, 0.84375, 0.9375, 0.90625, 0.9375, 0.9375, 0.9375, 0.9375, 0.90625, 0.9375, 0.90625, 0.875, 0.90625, 0.9375, 0.9375, 0.9375, 0.84375, 0.875, 0.90625, 1.0, 0.90625, 0.96875, 0.90625, 0.90625, 0.9375, 0.9375, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.875, 0.96875, 0.875, 0.90625, 0.96875, 1.0, 0.9375, 0.90625, 0.9375, 0.90625, 0.875, 0.90625, 0.96875, 0.96875, 0.9375, 0.96875, 0.96875, 0.9375, 0.9375, 0.96875, 0.875, 0.9375, 0.9375, 0.84375, 1.0, 0.96875, 0.875, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.96875, 0.96875, 0.96875, 0.9375, 0.90625, 0.90625, 0.90625, 0.9375, 0.875, 0.8125, 0.9375, 1.0, 0.90625, 0.96875, 0.96875, 0.90625, 0.9375, 0.875, 0.875, 1.0, 0.96875, 0.875, 1.0, 0.96875, 0.96875, 0.84375, 0.90625, 0.90625, 0.96875, 0.875, 0.90625, 0.9375, 0.96875, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 0.90625, 0.90625, 0.90625, 0.96875, 0.84375, 0.9375, 0.9375, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.96875, 0.90625, 0.875, 0.96875, 0.96875, 0.96875, 1.0, 0.96875, 0.9375, 0.90625, 0.875, 0.71875, 0.9375, 0.875, 1.0, 0.9375, 0.9375, 0.84375, 1.0, 0.9375, 0.96875, 0.875, 0.9375, 0.9375, 0.90625, 0.84375, 0.90625, 0.9375, 0.9375, 1.0, 0.9375, 0.78125, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.875, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.90625, 0.875, 0.90625, 0.875, 0.96875, 0.84375, 0.96875, 0.96875, 0.9375, 1.0, 0.96875, 0.9375, 0.84375, 0.875, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.90625, 0.9375, 0.9375, 0.9375, 0.9375, 0.96875, 0.875, 0.90625, 0.90625, 0.875, 0.9375, 0.9375, 0.90625, 0.9375, 0.90625, 0.9375, 0.875, 1.0, 0.90625, 0.96875, 0.875, 0.9375, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.875, 0.96875, 0.9375, 0.90625, 0.96875, 0.78125, 0.875, 0.9375, 0.90625, 0.90625, 0.96875, 0.90625, 0.9375, 0.96875, 0.9375, 0.9375, 0.90625, 0.8125, 1.0, 0.96875, 0.9375, 0.90625, 0.96875, 0.90625, 0.96875, 0.96875, 0.90625, 1.0, 1.0, 0.9375, 0.875, 0.90625, 1.0, 0.9375, 0.84375, 0.9375, 0.96875, 0.96875, 1.0, 0.96875, 0.96875, 0.84375, 0.875, 0.875, 0.96875, 0.90625, 0.84375, 0.875, 0.96875, 0.96875, 0.9375, 0.96875, 1.0, 0.90625, 0.96875, 0.90625, 0.90625, 0.90625, 0.9375, 0.9375, 0.84375, 0.9375, 0.9375, 0.84375, 0.96875, 0.84375, 0.90625, 0.9375, 0.875, 0.84375, 0.90625, 0.96875, 0.9375, 0.9375, 0.9375], [0.96875, 0.90625, 0.90625, 0.9375, 0.875, 0.8125, 0.96875, 0.9375, 0.84375, 0.84375, 0.875, 0.96875, 0.9375, 0.90625, 0.78125, 0.90625, 0.875, 0.90625, 0.875, 0.9375, 0.90625, 0.9375, 0.78125, 0.9375, 0.90625, 0.90625, 0.875, 0.90625, 0.9375, 0.84375, 0.9375, 0.78125, 0.84375, 0.96875, 0.875, 0.875, 0.9375, 0.8125, 0.90625, 0.90625, 0.90625, 0.9375, 0.96875, 0.875, 1.0, 0.90625, 0.96875, 0.9375, 0.875, 0.90625, 0.9375, 0.90625, 0.96875, 0.90625, 0.78125, 0.90625, 0.9375, 0.875, 0.90625, 0.9375, 0.9375, 1.0, 0.9375, 0.90625, 0.9375, 0.9375, 0.84375, 0.96875, 0.875, 0.96875, 0.9375, 0.90625, 0.84375, 0.90625, 0.75, 0.875, 0.96875, 0.96875, 0.90625, 0.875, 0.875, 0.875, 0.90625, 0.8125, 0.875, 0.96875, 0.8125, 0.90625, 0.9375, 0.90625, 0.84375, 0.90625, 0.9375, 0.9375, 0.875, 0.84375, 0.96875, 0.84375, 0.875, 0.9375, 0.84375, 0.96875, 0.875, 0.90625, 0.875, 0.90625, 0.90625, 1.0, 0.96875, 0.875, 0.75, 0.90625, 0.875, 0.90625, 0.9375, 0.84375, 0.90625, 1.0, 0.90625, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.9375, 0.84375, 0.96875, 0.9375, 0.875, 0.875, 1.0, 0.84375, 0.9375, 0.96875, 0.9375, 0.8125, 0.875, 0.9375, 0.9375, 0.84375, 0.84375, 0.90625, 0.84375, 0.96875, 0.96875, 1.0, 0.84375, 0.9375, 0.96875, 0.90625, 0.875, 0.9375, 0.90625, 0.96875, 0.84375, 0.96875, 0.96875, 0.96875, 0.9375, 0.875, 0.9375, 0.8125, 0.90625, 0.9375, 0.9375, 0.90625, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 0.875, 0.875, 0.90625, 0.84375, 1.0, 0.96875, 0.9375, 0.90625, 0.9375, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625, 0.8125, 0.875, 0.90625, 0.90625, 0.875, 0.875, 0.9375, 0.875, 0.96875, 0.90625, 0.90625, 0.875, 0.875, 0.9375, 1.0, 0.96875, 0.84375, 0.96875, 0.78125, 0.9375, 0.90625, 0.84375, 0.78125, 0.875, 0.875, 0.9375, 0.9375, 0.875, 0.90625, 0.875, 0.875, 0.90625, 0.96875, 0.96875, 0.78125, 0.96875, 0.96875, 0.84375, 0.875, 0.8125, 0.875, 0.875, 0.875, 0.875, 0.9375, 0.78125, 0.9375, 0.84375, 0.96875, 1.0, 0.90625, 0.90625, 1.0, 0.90625, 0.90625, 0.9375, 0.96875, 0.9375, 0.96875, 0.90625, 0.90625, 0.8125, 0.90625, 0.90625, 0.96875, 0.9375, 0.84375, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.84375, 0.90625, 0.9375, 0.875, 1.0, 0.9375, 1.0, 0.96875, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.90625, 0.8125, 0.875, 0.9375, 0.84375, 0.90625, 0.96875, 0.875, 0.90625, 0.90625, 0.90625, 0.875, 1.0, 0.9375, 0.9375, 0.9375, 0.84375, 0.90625, 0.9375, 0.9375, 0.875, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.90625, 0.9375, 0.9375, 0.9375, 0.96875, 0.96875, 0.9375, 0.9375, 0.875, 0.875, 0.96875, 0.9375, 0.90625, 0.875, 0.75]]
# # solo_correctness = [0.75, 0.875, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.9375, 0.9375, 1.0, 1.0, 0.9375, 0.90625, 0.9375, 0.90625, 0.96875, 0.96875, 1.0, 0.9375, 0.9375, 0.875, 0.9375, 0.90625, 0.875, 0.96875, 0.9375, 0.90625, 0.875, 0.96875, 0.96875, 0.9375, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.90625, 0.90625, 0.90625, 0.90625, 1.0, 0.78125, 0.90625, 1.0, 0.9375, 0.9375, 1.0, 0.90625, 0.90625, 0.9375, 0.96875, 0.84375, 0.96875, 0.96875, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 1.0, 1.0, 0.9375, 0.96875, 0.84375, 1.0, 1.0, 0.96875, 0.84375, 1.0, 0.90625, 0.9375, 0.96875, 1.0, 0.90625, 0.96875, 1.0, 0.96875, 0.96875, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 0.9375, 1.0, 0.96875, 1.0, 0.9375, 0.90625, 0.96875, 0.96875, 0.9375, 1.0, 0.9375, 1.0, 0.90625, 0.9375, 0.96875, 0.9375, 0.96875, 0.84375, 0.96875, 0.90625, 0.875, 0.90625, 0.9375, 0.96875, 1.0, 0.9375, 0.9375, 0.96875, 0.9375, 0.84375, 0.96875, 0.875, 0.90625, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 1.0, 0.90625, 0.90625, 0.96875, 0.9375, 1.0, 0.90625, 0.90625, 0.9375, 0.90625, 0.96875, 0.8125, 0.9375, 0.84375, 0.90625, 0.84375, 0.90625, 1.0, 0.96875, 1.0, 0.9375, 0.96875, 0.875, 1.0, 0.96875, 1.0, 0.875, 0.875, 0.9375, 0.96875, 1.0, 0.84375, 0.9375, 0.875, 0.96875, 1.0, 0.90625, 0.9375, 0.875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.90625, 0.90625, 0.96875, 0.9375, 0.8125, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.90625, 1.0, 0.96875, 1.0, 0.9375, 1.0, 0.9375, 0.9375, 0.9375, 0.875, 0.90625, 0.875, 0.96875, 1.0, 0.90625, 0.84375, 0.9375, 0.9375, 1.0, 1.0, 0.875, 1.0, 1.0, 0.9375, 0.90625, 0.96875, 0.90625, 1.0, 1.0, 0.90625, 0.96875, 0.96875, 1.0, 0.96875, 0.96875, 0.875, 0.90625, 0.96875, 0.90625, 1.0, 0.96875, 0.96875, 1.0, 0.875, 0.9375, 1.0, 0.875, 0.90625, 0.96875, 0.96875, 0.96875, 0.875, 0.96875, 0.9375, 0.875, 0.96875, 0.90625, 0.90625, 0.90625, 0.9375, 1.0, 0.84375, 1.0, 0.9375, 1.0, 0.96875, 0.9375, 0.875, 0.875, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 0.96875, 0.96875, 0.9375, 0.96875, 0.96875, 1.0, 0.96875, 0.9375, 0.90625, 0.96875, 0.84375, 1.0, 1.0, 0.96875, 0.90625, 0.96875, 0.90625, 1.0, 0.9375, 1.0, 0.9375, 0.90625, 0.96875, 0.96875, 1.0, 0.90625, 1.0, 0.90625, 1.0, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 1.0, 0.96875, 0.875, 0.90625, 0.9375, 0.9375, 0.96875, 0.875, 0.9375, 0.96875, 0.84375, 0.90625, 0.96875, 0.875, 1.0, 1.0]
# # acc_list = [solo_correctness] + [our_correctness[0]+our_correctness[1]] + [our_correctness[2]+our_correctness[3]]
# acc_list = [[0.9296875, 0.90625, 0.9375, 0.9453125, 0.9375, 0.9453125, 0.90625, 0.921875, 0.9453125, 0.953125, 0.9296875, 0.9453125, 0.9296875, 0.9609375, 0.9453125, 0.953125, 0.96875, 0.9375, 0.953125, 0.9296875, 0.96875, 0.9375, 0.8984375, 0.9765625, 0.9453125, 0.9609375, 0.90625, 0.9296875, 0.953125, 0.9375, 0.96875, 0.9609375, 0.9453125, 0.9453125, 0.921875, 0.9453125, 0.953125, 0.9609375, 0.9609375, 0.8984375, 0.921875, 0.9296875, 0.953125, 0.90625, 0.953125, 0.9296875, 0.9140625, 0.9375, 0.8984375, 0.96875, 0.9453125, 0.9296875, 0.90625, 0.9375, 0.921875, 0.9453125, 0.953125, 0.953125, 0.953125, 0.9453125, 0.9296875, 0.9375, 0.953125, 0.9296875, 0.9296875, 0.9453125, 0.984375, 0.953125, 0.9296875, 0.9609375, 0.9375, 0.953125, 0.9140625, 0.96875, 0.953125, 0.9453125, 0.96875, 0.9453125, 1.0],
# [0.90625, 0.9296875, 0.9375, 0.9375, 0.953125, 0.9453125, 0.90625, 0.890625, 0.921875, 0.9453125, 0.8984375, 0.9609375, 0.8828125, 0.9140625, 0.9296875, 0.90625, 0.9453125, 0.9765625, 0.921875, 0.8828125, 0.8984375, 0.9453125, 0.921875, 0.921875, 0.9375, 0.9609375, 0.9375, 0.9453125, 0.9609375, 0.9296875, 0.9140625, 0.8984375, 0.90625, 0.9140625, 0.953125, 0.890625, 0.9375, 0.890625, 0.8671875, 0.9140625, 0.9296875, 0.9609375, 0.9375, 0.90625, 0.953125, 0.9375, 0.9453125, 0.96875, 0.9375, 0.9140625, 0.953125, 0.9140625, 0.921875, 0.9140625, 0.921875, 0.921875, 0.9609375, 0.9296875, 0.8984375, 0.921875, 0.9453125, 0.9296875, 0.9375, 0.9296875, 0.9296875, 0.9453125, 0.953125, 0.9453125, 0.9375, 0.9140625, 0.90625, 0.9296875, 0.921875, 0.9140625, 0.921875, 0.9765625, 0.921875, 0.9140625, 1.0]+\
# [0.921875, 0.9453125, 0.9453125, 0.90625, 0.921875, 0.953125, 0.9453125, 0.9140625, 0.9375, 0.9609375, 0.9375, 0.9375, 0.921875, 0.9140625, 0.875, 0.890625, 0.9765625, 0.921875, 0.9140625, 0.9140625, 0.9375, 0.9140625, 0.921875, 0.9453125, 0.8984375, 0.90625, 0.9609375, 0.8984375, 0.90625, 0.9296875, 0.921875, 0.984375, 0.90625, 0.9140625, 0.9140625, 0.9296875, 0.9453125, 0.9453125, 0.921875, 0.9296875, 0.9296875, 0.9453125, 0.8984375, 0.8984375, 0.9453125, 0.90625, 0.9140625, 0.9375, 0.9453125, 0.9375, 0.9453125, 0.953125, 0.9609375, 0.953125, 0.9140625, 0.9140625, 0.9296875, 0.953125, 0.9296875, 0.9140625, 0.9453125, 0.9375, 0.90625, 0.9453125, 0.90625, 0.9453125, 0.921875, 0.953125, 0.8828125, 0.921875, 0.9453125, 0.90625, 0.8828125, 0.9140625, 0.9296875, 0.921875, 0.9140625, 0.9296875, 0.9375],
# [0.9140625, 0.8828125, 0.8359375, 0.8984375, 0.8984375, 0.8984375, 0.890625, 0.90625, 0.8828125, 0.9140625, 0.8671875, 0.875, 0.8828125, 0.8671875, 0.9140625, 0.8515625, 0.859375, 0.8984375, 0.875, 0.78125, 0.8515625, 0.84375, 0.8515625, 0.9453125, 0.8828125, 0.84375, 0.859375, 0.8984375, 0.8359375, 0.890625, 0.875, 0.8515625, 0.875, 0.8671875, 0.84375, 0.921875, 0.8515625, 0.921875, 0.8515625, 0.875, 0.828125, 0.875, 0.84375, 0.875, 0.875, 0.8984375, 0.8984375, 0.84375, 0.875, 0.8671875, 0.875, 0.8359375, 0.890625, 0.859375, 0.890625, 0.875, 0.84375, 0.875, 0.828125, 0.8828125, 0.8125, 0.8671875, 0.84375, 0.8828125, 0.84375, 0.8359375, 0.8984375, 0.828125, 0.8984375, 0.859375, 0.8515625, 0.8828125, 0.8671875, 0.796875, 0.875, 0.8046875, 0.84375, 0.921875, 0.9375]+\
# [0.9140625, 0.9609375, 0.953125, 0.890625, 0.921875, 0.9296875, 0.9453125, 0.9375, 0.9375, 0.953125, 0.9140625, 0.9296875, 0.9296875, 0.9296875, 0.9296875, 0.921875, 0.921875, 0.921875, 0.890625, 0.9375, 0.890625, 0.890625, 0.8828125, 0.953125, 0.890625, 0.9453125, 0.8671875, 0.8671875, 0.8984375, 0.8984375, 0.9140625, 0.890625, 0.875, 0.8671875, 0.8984375, 0.8828125, 0.9453125, 0.9296875, 0.921875, 0.921875, 0.859375, 0.90625, 0.859375, 0.890625, 0.9453125, 0.9453125, 0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.8515625, 0.921875, 0.90625, 0.921875, 0.890625, 0.921875, 0.890625, 0.9375, 0.90625, 0.90625, 0.9375, 0.8984375, 0.875, 0.8671875, 0.8828125, 0.9140625, 0.890625, 0.90625, 0.921875, 0.9453125, 0.9296875, 0.9375, 0.9375, 0.953125, 0.8515625, 0.9453125, 0.9296875, 0.9375, 1.0],
# ]

# labels = ['Original','Partition (100%)','Partition (50%)']
# colors_tmp = ['k'] + colors[:2]
# measurements_to_cdf(acc_list,'/home/bo/Dropbox/Research/SIGCOMM23/images/mot3.eps',labels,colors=colors_tmp,linestyles=linestyles,xlabel='Accuracy (%)',ratio=None,lbsize=20)
# # 4. via partitioning It is possible to maintain two-server performance while trading one-server performance for less computation overhead
# # with loss, the performance is close 
# methods_tmp = ['Accuracy','FLOPS']
# y = [[86.287141,100], [93.467252,200],[89.25,25.08*2]] 
# yerr = [[1.330113,0], [0.563777,0],[0.35401592039360327,0]]
# y,yerr = np.array(y),np.array(yerr)
# groupedbar(y,yerr,'%', 
# 	'/home/bo/Dropbox/Research/SIGCOMM23/images/mot4.eps',methods=methods_tmp,
# 	envs=['Standalone','Replication','Partition'],ncol=1,sep=.3,width=0.1,legloc='best',
# 	use_downarrow=True,labelsize=20)

colors_tmp = ['#e3342f','#f6993f','#ffed4a','#38c172','#4dc0b5','#3490dc','#6574cd','#9561e2','#f66d9b']
for name in ['resnet56-2','resnet56-3','resnet56-4']:
	numsn = 4
	if name == 'resnet56-3':
		numsn = 6
	elif name == 'resnet56-4':
		numsn = 8
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
	line_plot(all_epoch,all_acc,methods_tmp,colors_tmp,
			f'/home/bo/Dropbox/Research/SIGCOMM23/images/{name}.eps',
			'Epoch','Test Accuracy (%)',markersize=0,linewidth=2,xticks=xticks,legloc='best',
			use_resnet56_2arrow=(name == 'resnet56-2'),use_resnet56_3arrow=(name == 'resnet56-3'),
			use_resnet56_4arrow=(name == 'resnet56-4'),use_resnet50arrow=(name == 'resnet50'))

# a = [0.9345047923322684, 0.9362020766773163, 0.9336062300319489, 0.9326078274760383, 0.9308107028753994, 0.9269169329073482, 0.9281150159744409, 0.9285143769968051]
# a = [0.9396964856230032, 0.9386980830670927, 0.9401956869009584, 0.9352036741214057, 0.9362020766773163, 0.9343051118210862]


# different convs
re_vs_conv = [[0.008119009584664539, 0.003213208753747091,0.009724440894568698, 0.0011576304345555521],
[0.007833466453674132, 0.0034881853524135523,0.009375000000000033, 0.000749068110299625],
[0.008897763578274764, 0.003976683377131077,0.007068690095846664, 0.001153835480287017],
  [0.007140575079872202, 0.004132095691370403, 0.006259984025559107, 0.0013025614474167567],
[0.005862619808306713, 0.004521608104803331,0.006080271565495199, 0.000603104565117374],
# [0.008408546325878586, 0.003743638959291401, 0.00866613418530352, 0.0009942004220850314]
]
flops_vs_conv = [0.5298202593545005,0.558022230677192,0.5862242019998837,0.6003,0.642628144645267]#,0.6990320872906503]
bridge_size = [64,96,128,144,192]#,256]
re_vs_conv = np.array(re_vs_conv)
y = re_vs_conv[:,[0,2]]
yerr = re_vs_conv[:,[1,3]]
envs = [f'{bridge_size[i]}\n{int(flops_vs_conv[i]*100)}%' for i in range(5)]
groupedbar(y,yerr,'$R^{(2)}$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_conv.eps',methods=['Soft','Hard'],envs=envs,
	ncol=1,sep=1,bbox_to_anchor=(0.5, 1.2),width=0.4,xlabel='Bridge Size and FLOPS (%)',legloc='upper right')

# different sampling rates
re_vs_sr = [[0.0067911341853035155, 0.0034479736504274597,0.0051317891373801935, 0.0009054140614021413],
[0.004664536741214043, 0.004285370800509071,0.0042432108626198175, 0.0005616846317506139],
[0.004145367412140566, 0.003862424004132929,0.0048422523961661355, 0.0011472944346802055],
[0.004714456869009587, 0.004595007816971645,0.00409345047923323, 0.0013040528647967876],
[0.007140575079872202, 0.004132095691370403, 0.006259984025559107, 0.0013025614474167567],]

flops_vs_sr = [1.1925665854377538,0.8964458865494916,0.7483855371053606,0.6743553623832951,0.6003]
sample_interval = [1,2,3,4,9]
re_vs_sr = np.array(re_vs_sr)
y = re_vs_sr[:,[0,2]]
yerr = re_vs_sr[:,[1,3]]
envs = [f'{sample_interval[i]}\n{int(flops_vs_sr[i]*100)}%' for i in range(5)]
groupedbar(y,yerr,'$R^{(2)}$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_sr.eps',methods=['Soft','Hard'],envs=envs,
	ncol=2,sep=1,bbox_to_anchor=(0.5, 1.2),width=0.4,xlabel='Sampling Interval and FLOPS (%)',legloc='upper center')

# different partition ratios
diff_ratio_data = [[0.0007507987220447321, 0.003741558835808713, -0.000389376996805102, 0.0008822738262825615, 1.006022295959533],
[0.002939297124600637, 0.003882210251418619, 0.0010183706070287402, 0.0009027679357394152, 0.8929206401341552],
[0.004205271565495205, 0.003454625197475081, 0.003125, 0.001119858795400797, 0.7876039034759786],
[0.005842651757188488, 0.003802445462229222, 0.0056709265175718835, 0.0008597863475084146, 0.6900720859850035],
  [0.007140575079872202, 0.004132095691370403, 0.006259984025559107, 0.0013025614474167567,0.6003],
[0.008568290734824258, 0.004092656063333119, 0.009964057507987224, 0.0014678894528345747,0.518363208504657],
[0.01164337060702877, 0.003929390299692293, 0.013807907348242799, 0.0016216282858508531,0.44418614851528576],
[0.010545127795527154, 0.003529936353411632, 0.019968051118210893, 0.0032287171673031755,0.3777940076931159]]
diff_ratio_data = np.array(diff_ratio_data)
y = diff_ratio_data[:,[0,2]]
yerr = diff_ratio_data[:,[1,3]]
envs = [f'{0.0625*i}\n{int(diff_ratio_data[i-1,[4]]*100)}%' for i in range(1,9)]
groupedbar(y,yerr,'$R^{(2)}$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_ratio.eps',methods=['Soft','Hard'],envs=envs,
	ncol=2,sep=1.3,width=0.4,xlabel='Partition Ratio and FLOPS (%)',legloc='upper left')

# # baseline
# methods_tmp = ['Standalone','Ours']
# y = [[0.21911602480000006,0.006703817600000122],[0.17067866719999998,0.003996000000000066]]
# yerr = [[0.0019785233779630296,0.0009860426520709135],[0.0011066291237713699,0.0002033875583595318]]
# y,yerr = np.array(y),np.array(yerr)
# groupedbar(y,yerr,'$R^{(2)}$', 
# 	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_imagenet.eps',methods=methods_tmp,
# 	envs=['Soft','Hard'],ncol=1,sep=.3,width=0.1,legloc='best',use_barlabe_y=True)


# flop distribution
flops = [
	[100, 0],
	[100,100],
	[86.77293603320655]*2,
	[70.53025247728113]*2,
	[63.75213554749708]*2,
	[60.03251876612296]*2,
	[53.60070051652971]*2
	]
labels = ['Standalone','Replication','Ours\n(ResNet-20)','Ours\n(ResNet-32)','Ours\n(ResNet-44)','Ours\n(ResNet-56)','Ours\n(ResNet-110)']
filename = '/home/bo/Dropbox/Research/SIGCOMM23/images/flops_dist.eps'
plot_computation_dist(flops,labels,filename,horizontal=False)
# 77.09%: resnet-50



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
latency_types = ['N-Map','DCN','N-Reduce','WAN']	
labels = ['1','2', '4','8','16', '32','64']
plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
	'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_breakdown_vs_batch.eps',4,(0.5,0.85),title_posy=0.2)

# cifar breakdown
latency_types = ['N-Map','DCN','N-Reduce','Infer','WAN']	
labels = ['Ours','SE.', 'RN.']
latency_breakdown_mean = [[0.0048398783412604285, 0.014246439476907482, 0.00023509953349543075,0, 0.7486890470647757],
[0,0,0,0.00483251379701657, 1.0169146132483973],
[0,0,0,0.00483251379701657, 0.7486877294765364],]

latency_breakdown_std = [[0.00021182092754556427, 0.0011024832600207727, 4.049920186137311e-05,0, 0.34634288409670605],
[0,0,0,0.0002204459821230588, 0.5684324923094014],
[0,0,0,0.0002204459821230588, 0.34634275598435527],]

plot_latency_breakdown(latency_breakdown_mean,latency_breakdown_std,latency_types,labels,
	'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_breakdown_cifar.eps',3,(0.5,0.825),title_posy=0.25,ratio=0.6)


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
	'/home/bo/Dropbox/Research/SIGCOMM23/images/latency_breakdown_imagenet.eps',3,(0.5,0.85),title_posy=0.25,lim1=100,lim2=16000,ratio=0.6)


# ratio of latency
analyze_all_recorded_traces()

# multi-node analysis
methods = ['Soft (se.)','Hard (se.)','Soft (rn.)','Hard (rn.)']

re_vs_nodes = [[0.12740115814696484,0.13641316750342308,0.010894568690095847,0.004692492012779541],
[0.1817881389776358,0.20846265023581317,0.008861821086261991,0.002529286474973359],
[0.22135682907348236,0.24593412444850138,0.013224840255591045,0.006014186824889697]]

y = np.array(re_vs_nodes)
groupedbar(y,None,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_nodes.eps',methods=methods,
	envs=['Two-server','Three-server','Four-server'],width=0.25,bbox_to_anchor=(0.5, 1.25),use_barlabel_x=True,legloc='best',ncol=2,ylim=(0,0.3))


envs = ['ResNet-20','ResNet-32','ResNet-44','ResNet-56','ResNet-110']
methods_tmp = ['Standalone','Split','Ours']
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
	envs=envs,ncol=1,width=.25,sep=1,legloc='best',ylim=(85,95),rotation=30)

soft = [[0.12374700479233228,0.026222044728434514,0.006991813099041515],
[0.12570587060702873, 0.019972044728434522,0.009653554313099012],
[0.12668630191693292, 0.02085463258785944,0.012633785942492015],
[0.12740115814696484, 0.026772164536741177,0.010894568690095847],
[0.12879093450479234, 0.01743610223642173,0.015284544728434536]
]

y = np.array(soft)
groupedbar(y,None,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/soft_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,width=.25,sep=1,legloc='center left',use_barlabe_y=True,rotation=30)

hard = [[0.1316902479841777,0.02511220143009282,0.0019730336223946314],
[0.13636752624372434,0.01973033622394643,0.005381865206146362],
[0.13509242355089, 0.01987771945839038,0.007630648105887723],
[0.13641316750342308, 0.027132778031340316,0.004649703331811962],
[0.14010250266240679, 0.016777917237182412,0.00557203712155789]]

y = np.array(hard)
groupedbar(y,None,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,width=.25,sep=1,legloc='center left',use_barlabe_y=True,rotation=30)

# soft
y = [[0.09999999999999999, 0.10054512779552716, 0.10054512779552716, 0.11902755591054312, 0.24644369009584666, 0.4110902555910543, 0.5672503993610223, 0.6601238019169329, 0.7321505591054313, 0.793057108626198, 0.8280910543130989, 0.8515195686900958, 0.8795467252396165, 0.8962060702875398, 0.903905750798722, 0.9113977635782747, 0.915401357827476, 0.9167292332268369, 0.9250139776357826, 0.9292152555910542], [0.09999999999999999, 0.10026757188498403, 0.10026757188498403, 0.12053314696485622, 0.19403953674121407, 0.30107228434504796, 0.40742012779552716, 0.4781529552715654, 0.5490015974440895, 0.6168170926517572, 0.6582428115015976, 0.6940595047923324, 0.7466353833865815, 0.7764436900958467, 0.8043510383386583, 0.8349560702875399, 0.8493550319488816, 0.8560443290734824, 0.8767372204472844, 0.8892432108626197], [0.09999999999999999, 0.10055511182108626, 0.10055511182108626, 0.12967052715654953, 0.265960463258786, 0.43768769968051113, 0.5874161341853036, 0.6758865814696484, 0.7458226837060702, 0.8056110223642173, 0.838514376996805, 0.8573542332268371, 0.8893550319488819, 0.9001757188498403, 0.9084305111821086, 0.9165774760383385, 0.9192432108626198, 0.9202935303514378, 0.9301637380191693, 0.9328494408945687]]
y += [[0.09999999999999999, 0.10053514376996806, 0.10266573482428117, 0.14354432907348244, 0.3245746805111821, 0.5350858626198083, 0.7037440095846644, 0.7912959265175717, 0.8449900159744409, 0.885880591054313, 0.9049580670926517, 0.9186940894568689, 0.9284964057507986, 0.933292731629393, 0.9351956869009583, 0.9370287539936102, 0.937821485623003, 0.938356629392971, 0.9386242012779551, 0.9404373003194888], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12064297124600636, 0.19427915335463258, 0.3012919329073483, 0.40771964856230036, 0.478013178913738, 0.5486142172523961, 0.6166074281150159, 0.6581130191693292, 0.6936900958466452, 0.7458466453674121, 0.7759944089456868, 0.8038019169329074, 0.8345666932907347, 0.8490255591054312, 0.8558346645367412, 0.8764576677316294, 0.8892332268370605], [0.09999999999999999, 0.10055511182108626, 0.10348841853035144, 0.16188897763578275, 0.35286341853035147, 0.5664017571884984, 0.7198242811501598, 0.8020607028753993, 0.8510183706070287, 0.8899960063897764, 0.9071705271565496, 0.9196485623003194, 0.9303614217252397, 0.933849840255591, 0.9360303514376996, 0.9376257987220447, 0.938180910543131, 0.9387060702875398, 0.9400139776357828, 0.9408266773162939]]
y += [[0.09999999999999999, 0.10080271565495207, 0.1032308306709265, 0.16486222044728435, 0.39264776357827474, 0.6277076677316293, 0.7823861821086261, 0.8479812300319489, 0.8836880990415334, 0.9090674920127796, 0.9193051118210864, 0.9266773162939298, 0.931898961661342, 0.9332168530351439, 0.9335043929712461, 0.9340395367412141, 0.9340395367412141, 0.9343071086261983, 0.9343071086261983, 0.9345746805111823], [0.09999999999999999, 0.10027755591054313, 0.10027755591054313, 0.12060303514376995, 0.19419928115015977, 0.3015595047923323, 0.40726038338658144, 0.47754392971246007, 0.5476477635782747, 0.6165375399361024, 0.6580131789137379, 0.6937100638977636, 0.7460163738019169, 0.7760842651757188, 0.8038119009584663, 0.8344868210862619, 0.8490155750798722, 0.855804712460064, 0.8766074281150159, 0.8892831469648561], [0.09999999999999999, 0.10080271565495207, 0.10588658146964855, 0.18738019169329073, 0.42757787539936104, 0.6566253993610223, 0.7983146964856228, 0.8612539936102236, 0.8946825079872205, 0.9179932108626199, 0.9266553514376996, 0.9332747603833864, 0.9383785942492013, 0.9397464057507987, 0.9400139776357828, 0.9405690894568689, 0.9405690894568689, 0.940836661341853, 0.940836661341853, 0.9410942492012779]]

yerr = [[0.0, 0.0010904841391130044, 0.0010904841391130044, 0.012901423965101584, 0.05386465228666159, 0.06330897181706202, 0.061088301211427776, 0.042496575722181885, 0.03791339018992345, 0.0286516040794726, 0.024502583576222894, 0.025234458910697543, 0.025214803986917174, 0.02004981065222387, 0.01913403883501498, 0.0210146786784222, 0.020861977353891028, 0.020812112201925786, 0.012489402392714916, 0.009346223105772366], [0.0, 0.0008027156549520825, 0.0008027156549520825, 0.014877741975549278, 0.041380313427382555, 0.059803487736121326, 0.06442090733061269, 0.054010348189044745, 0.046256730645281255, 0.04990167454459851, 0.04680540980752168, 0.04270990091509234, 0.04092552622057239, 0.04201476212272375, 0.035258841989506476, 0.024606431299643745, 0.024426260082436837, 0.02257806012117962, 0.02338451126124495, 0.01813754932754337], [0.0, 0.0011111211234452062, 0.0011111211234452062, 0.020617885360086453, 0.058052933419270515, 0.06406899866962676, 0.06046125060499545, 0.04090110185496498, 0.03818100531319478, 0.022599839902736295, 0.024208484492046708, 0.02502698385338613, 0.021951503438406496, 0.0199137496482411, 0.019387677509606134, 0.020391848164711325, 0.020533313186606467, 0.020248210447870275, 0.01021375164308125, 0.008686921346931619]]
yerr += [[0.0, 0.0010740064615502457, 0.006303007931757033, 0.026021125914269503, 0.08105363023623796, 0.0740595744589342, 0.06041652501440283, 0.044844806014635245, 0.028175216533628814, 0.017087291384438583, 0.013970411329020677, 0.011378221653788169, 0.010262507752688744, 0.008127237448455074, 0.007248766702406512, 0.005366632081894418, 0.00540657578979667, 0.005105690103907452, 0.004519983947197429, 0.0021377652284912336], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.014884064755360492, 0.041080908076770895, 0.058943171934852476, 0.06350502758454735, 0.053835822922173385, 0.046569856977913715, 0.049974697051315166, 0.047266872346802026, 0.04321938603316445, 0.04093103410050893, 0.04192103684866083, 0.03560689885840395, 0.024849190569816677, 0.024534119292260462, 0.02273722235521622, 0.02336269319504597, 0.018269360880982136], [0.0, 0.0011102236421725309, 0.006982465449370012, 0.03891130443930724, 0.07933421510907138, 0.07312045790566846, 0.05709689397238947, 0.03976837160964229, 0.02660124404395353, 0.015912401995836385, 0.01412622823382116, 0.010690884125923806, 0.008377834943452267, 0.006884396773024983, 0.0056704638586413956, 0.004258329577667321, 0.004336842354278658, 0.0041518970931095625, 0.00178413056171682, 0.0008027156549520573]]
yerr += [[0.0, 0.0012269810926168812, 0.006283865420787238, 0.028908457380234936, 0.08261134500237968, 0.06686804182363132, 0.04642606964208399, 0.03309409330081189, 0.022616908893124406, 0.012616052522013305, 0.008800497129825013, 0.004936795253624663, 0.0031782620755874194, 0.0024480741091414157, 0.002122190531986667, 0.0021105776637082045, 0.0021105776637082045, 0.0020216199707253544, 0.0020216199707253544, 0.002237869831544598], [0.0, 0.0008326677316293981, 0.0008326677316293981, 0.015036159407656817, 0.04115502264840482, 0.0596680467179605, 0.06398301658478026, 0.05410036418702724, 0.04657518349967236, 0.049823079627498594, 0.04696471449231237, 0.04299295114084161, 0.04112233192725984, 0.042494933804224914, 0.03553520396804823, 0.024830376263057598, 0.024664085467536682, 0.022878465565610913, 0.0235558085394264, 0.01808878859085014], [0.0, 0.0012269810926168812, 0.00856012814586162, 0.04520614628870527, 0.08161380704950524, 0.06805698287006944, 0.04363282099972548, 0.02765925284340011, 0.022176663654976114, 0.013079873551852804, 0.009191775440817871, 0.004524296847070941, 0.002748419268439603, 0.0021768518736728507, 0.0021615809620174464, 0.0015754792332268283, 0.0015754792332268283, 0.0007727635782747377, 0.0007727635782747377, 0.0]]

y,yerr = np.array(y)*100,np.array(yerr)*100
y = y[[2,5,8,0,3,6,1]];yerr = yerr[[2,5,8,0,3,6,1]]
linestyles_tmp = ['solid','dashed','solid','dotted','solid','dashdot',(0, (3, 5, 1, 5))]
methods_tmp = [r'Rn.($\times$2)',r'Rn.($\times$3)',r'Rn.($\times$4)',r'Ours($\times$2)',r'Ours($\times$3)',r'Ours($\times$4)','Standalone']
colors_tmp = colors + ['k']
x = [[0.1*i for i in range(1,21)]for _ in range(7)]
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/soft_ea_multi_nodes.eps',
		'Deadline (s)','Delivered Accuracy (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,yticks=[0,25,50,75,100],
		yerr=yerr)	
# hard
y = [[0.9372004792332268, 0.9361701277955271, 0.9269009584664536, 0.9164237220447286, 0.9027735623003196, 0.8834584664536742, 0.8610463258785941, 0.8285802715654953, 0.8152276357827477, 0.759117412140575, 0.7229712460063897, 0.6756689297124601, 0.6312699680511182, 0.5680770766773162, 0.5175319488817891, 0.4741453674121406, 0.4014556709265175, 0.33900359424920135, 0.2573242811501598, 0.18030551118210864, 0.09999999999999999], [0.9411940894568691, 0.902811501597444, 0.8544488817891374, 0.8166713258785941, 0.7749001597444088, 0.7351876996805112, 0.6917911341853034, 0.6364616613418531, 0.6134464856230032, 0.5541433706070287, 0.5137739616613418, 0.46715055910543135, 0.43328873801916934, 0.39498402555910544, 0.3444468849840256, 0.32219249201277955, 0.2572484025559106, 0.23126397763578277, 0.18252396166134188, 0.14374400958466454, 0.09999999999999999], [0.9411940894568691, 0.9401437699680513, 0.9326417731629391, 0.9210263578274761, 0.9081449680511181, 0.8886900958466454, 0.8666873003194888, 0.8341613418530353, 0.8198901757188498, 0.7644788338658146, 0.7271046325878594, 0.6807507987220446, 0.6360423322683705, 0.5725099840255591, 0.5214756389776357, 0.4769908146964855, 0.4042711661341853, 0.34105031948881787, 0.25919129392971246, 0.18125399361022368, 0.09999999999999999]]
y += [[0.9400958466453673, 0.9401956869009584, 0.9383805910543129, 0.9373063099041534, 0.933089057507987, 0.9279732428115016, 0.9211841054313099, 0.9051557507987219, 0.8867132587859425, 0.857855431309904, 0.8277096645367411, 0.7944748402555909, 0.7633366613418531, 0.7049161341853036, 0.6524500798722045, 0.5657148562300318, 0.5106709265175718, 0.44044728434504793, 0.3210682907348243, 0.21947683706070292, 0.09999999999999999], [0.9410942492012779, 0.901283945686901, 0.8557068690095846, 0.8141493610223642, 0.7788997603833866, 0.7213238817891373, 0.6767871405750798, 0.6389976038338658, 0.5990415335463258, 0.574297124600639, 0.5195986421725239, 0.4946745207667732, 0.43301717252396166, 0.3993829872204473, 0.3646305910543131, 0.289576677316294, 0.2619848242811502, 0.23528554313099043, 0.18138378594249202, 0.14484424920127797, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9394388977635784, 0.9389736421725241, 0.9354852236421725, 0.9292012779552715, 0.9227515974440894, 0.9070926517571886, 0.8886102236421725, 0.8608406549520767, 0.8305351437699681, 0.7975898562300319, 0.7667212460063897, 0.7082308306709264, 0.6555151757188498, 0.5692392172523961, 0.5138258785942492, 0.44196485623003195, 0.322366214057508, 0.2194468849840256, 0.09999999999999999]]
y += [[0.9375, 0.9376497603833867, 0.9371445686900959, 0.9371026357827474, 0.9352535942492013, 0.9325499201277955, 0.9315035942492014, 0.9250638977635782, 0.9102176517571886, 0.9010862619808307, 0.8807787539936103, 0.8570706869009586, 0.8257128594249202, 0.7743091054313098, 0.7381749201277954, 0.6637919329073483, 0.5800619009584664, 0.4988019169329073, 0.38542731629392973, 0.2624101437699681, 0.09999999999999999], [0.9410942492012779, 0.8992911341853036, 0.8525379392971246, 0.8219109424920127, 0.7779173322683706, 0.7269888178913739, 0.6812340255591054, 0.6470347444089457, 0.6034464856230031, 0.5638059105431309, 0.5189456869009584, 0.47060103833865813, 0.44147164536741207, 0.39790934504792336, 0.3615874600638978, 0.3022923322683707, 0.2679213258785943, 0.22620607028753992, 0.17921325878594252, 0.14382388178913744, 0.09999999999999999], [0.9410942492012779, 0.9410942492012779, 0.9405890575079873, 0.9408166932907349, 0.9397563897763579, 0.9380810702875401, 0.9364756389776356, 0.9311142172523962, 0.915, 0.907645766773163, 0.8875079872204472, 0.8641693290734824, 0.8331110223642172, 0.7802096645367411, 0.7443949680511183, 0.6709804313099041, 0.5870706869009584, 0.502745607028754, 0.3899201277955272, 0.265145766773163, 0.09999999999999999]]

yerr = [[0.0, 0.002383087978836537, 0.003873038115141878, 0.0061453230457425975, 0.012967568025285573, 0.010063407266560125, 0.008824341552946188, 0.01611947452454557, 0.01890696132833872, 0.013174793019248588, 0.019867655361733523, 0.02406030887038924, 0.019344045784808497, 0.026841082314703498, 0.03579666570409922, 0.028536735524548405, 0.023743592117576182, 0.013090604293229717, 0.01902712161252409, 0.0103868882719809, 0.0], [1.1102230246251565e-16, 0.008922661919075328, 0.010855201934208752, 0.014047328679292339, 0.015535675044506014, 0.021138001603641914, 0.02019673453507974, 0.019017540855227846, 0.02697623136127924, 0.0165951113967507, 0.015869314630196252, 0.01722226165574532, 0.016248312393671693, 0.0225139485027758, 0.024553767381638156, 0.01572217242443465, 0.01882411807025787, 0.0094800814188465, 0.01617607815639621, 0.008617508759934122, 0.0], [1.1102230246251565e-16, 0.0021006389776357715, 0.004292221186753114, 0.005502080447956476, 0.012873009197567621, 0.009931106560055136, 0.008520796056967147, 0.016406406989810517, 0.019195156676365066, 0.014236333899824661, 0.019578295223714896, 0.025014428016724065, 0.02032100402565394, 0.026638863608938305, 0.03630411823945728, 0.02905046349658962, 0.0247702550563865, 0.013421182987465093, 0.019594189900178752, 0.009960898934023193, 0.0]]
yerr += [[1.1102230246251565e-16, 0.000995402841902101, 0.0019230062619173323, 0.0029350189218929797, 0.00398937924839249, 0.0056962200434122765, 0.006343453860877856, 0.008939742647962826, 0.012168240034661555, 0.013548031179396408, 0.010352070348634003, 0.01476602223042188, 0.015315176158104037, 0.013853354946653736, 0.014429536171800459, 0.024700795620692797, 0.02165818295974954, 0.026376494321830004, 0.017636004877208376, 0.013967911382259396, 0.0], [0.0, 0.008788792242237194, 0.014406748369610225, 0.01914729164577468, 0.016900213540757675, 0.015256297891266042, 0.017869286669125815, 0.01721277510219242, 0.030673250835451138, 0.02429898644176315, 0.019434009232524578, 0.018285541630780777, 0.017436328739240895, 0.020986287686181066, 0.01961241417004981, 0.02212697397046627, 0.01855550272008473, 0.024734784397244265, 0.011483376204111629, 0.010381818971123509, 0.0], [0.0, 0.0, 0.0018206975879296605, 0.0022187874482035596, 0.0034773826415680944, 0.005119523263411318, 0.006296798832352713, 0.008350527247477434, 0.011745044717527173, 0.013318481881632085, 0.010710799335342996, 0.014662990250481664, 0.015255625554048257, 0.012476912709232601, 0.015621043017821563, 0.02367807973759844, 0.022399031195351683, 0.025831532860168975, 0.017872497713262124, 0.014208441988889771, 0.0]]
yerr += [[0.0, 0.0007340126308192104, 0.0016790222193592268, 0.001307578062589785, 0.0031738430517763, 0.002771281353113958, 0.004821811372567936, 0.003766440262694593, 0.011413445446441009, 0.007024970391018178, 0.006641610340463003, 0.012514366348787647, 0.015057229462531088, 0.0270727367263636, 0.015020235103509156, 0.024594070767625157, 0.020907486396156177, 0.028864179469003905, 0.018842506916103334, 0.012339164246798697, 0.0], [0.0, 0.007235762980987184, 0.01883170894266918, 0.018729712734086437, 0.021647803322049578, 0.022337040981863596, 0.01886215930979941, 0.02078621611097996, 0.0263849748859761, 0.030812869391009003, 0.016535778834780458, 0.02421286892331994, 0.013720264138036297, 0.022343865629389774, 0.024706330212331377, 0.028134078150203846, 0.014843995233644925, 0.01338282025191434, 0.012181244413064132, 0.011104893901555347, 0.0], [0.0, 0.0, 0.0010165307096568136, 0.0008326677316293772, 0.002489656459515469, 0.00228894232457122, 0.0038375150073354097, 0.004086751317099877, 0.01060951544718555, 0.008054753685088496, 0.006235662698745973, 0.013869294865888601, 0.015110500730909011, 0.02580070555943828, 0.013983579435754644, 0.024901725834084645, 0.021823995121518922, 0.02967566715074053, 0.018866276695059758, 0.012773052866064803, 0.0]]

y,yerr = np.array(y)*100,np.array(yerr)*100
y = y[[2,5,8,0,3,6,1]];yerr = yerr[[2,5,8,0,3,6,1]]
x = [[5*i for i in range(21)]for _ in range(7)]
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_ea_multi_nodes.eps',
		'Loss Rate (%)','Delivered Accuracy (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,yticks=[25,50,75,100],
		yerr=yerr)	



# FCC 2-nodes no loss analysis
# CIFAR, IMAGENET EA and FR FCC
x = [[0.1*i for i in range(1,21)] for _ in range(4)]
y = [[0.09999999999999999, 0.10054512779552716, 0.10054512779552716, 0.11902755591054312, 0.24644369009584666, 0.4110902555910543, 0.5672503993610223, 0.6601238019169329, 0.7321505591054313, 0.793057108626198, 0.8280910543130989, 0.8515195686900958, 0.8795467252396165, 0.8962060702875398, 0.903905750798722, 0.9113977635782747, 0.915401357827476, 0.9167292332268369, 0.9250139776357826, 0.9292152555910542], [0.09999999999999999, 0.10026757188498403, 0.10026757188498403, 0.12053314696485622, 0.19403953674121407, 0.30107228434504796, 0.40742012779552716, 0.4781529552715654, 0.5490015974440895, 0.6168170926517572, 0.6582428115015976, 0.6940595047923324, 0.7466353833865815, 0.7764436900958467, 0.8043510383386583, 0.8349560702875399, 0.8493550319488816, 0.8560443290734824, 0.8767372204472844, 0.8892432108626197], [0.09999999999999999, 0.10055511182108626, 0.10055511182108626, 0.12967052715654953, 0.265960463258786, 0.43768769968051113, 0.5874161341853036, 0.6758865814696484, 0.7458226837060702, 0.8056110223642173, 0.838514376996805, 0.8573542332268371, 0.8893550319488819, 0.9001757188498403, 0.9084305111821086, 0.9165774760383385, 0.9192432108626198, 0.9202935303514378, 0.9301637380191693, 0.9328494408945687]]
y += [[0.09999999999999999, 0.10049520766773161, 0.10049520766773161, 0.12854233226837058, 0.2586421725239617, 0.4218510383386582, 0.5637360223642173, 0.6486861022364215, 0.7149021565495206, 0.7717452076677316, 0.8030311501597444, 0.8208426517571885, 0.8512160543130991, 0.8616873003194888, 0.8696825079872204, 0.8772404153354632, 0.8797963258785944, 0.8807268370607029, 0.8902276357827477, 0.8928234824281148]]
yerr = [[0.0, 0.0010904841391130044, 0.0010904841391130044, 0.012901423965101584, 0.05386465228666159, 0.06330897181706202, 0.061088301211427776, 0.042496575722181885, 0.03791339018992345, 0.0286516040794726, 0.024502583576222894, 0.025234458910697543, 0.025214803986917174, 0.02004981065222387, 0.01913403883501498, 0.0210146786784222, 0.020861977353891028, 0.020812112201925786, 0.012489402392714916, 0.009346223105772366], [0.0, 0.0008027156549520825, 0.0008027156549520825, 0.014877741975549278, 0.041380313427382555, 0.059803487736121326, 0.06442090733061269, 0.054010348189044745, 0.046256730645281255, 0.04990167454459851, 0.04680540980752168, 0.04270990091509234, 0.04092552622057239, 0.04201476212272375, 0.035258841989506476, 0.024606431299643745, 0.024426260082436837, 0.02257806012117962, 0.02338451126124495, 0.01813754932754337], [0.0, 0.0011111211234452062, 0.0011111211234452062, 0.020617885360086453, 0.058052933419270515, 0.06406899866962676, 0.06046125060499545, 0.04090110185496498, 0.03818100531319478, 0.022599839902736295, 0.024208484492046708, 0.02502698385338613, 0.021951503438406496, 0.0199137496482411, 0.019387677509606134, 0.020391848164711325, 0.020533313186606467, 0.020248210447870275, 0.01021375164308125, 0.008686921346931619]]
yerr += [[0.0, 0.00099443300328881, 0.00099443300328881, 0.019505029461655107, 0.0552931148145234, 0.06093023519257489, 0.05759666397695378, 0.038977482134838315, 0.03628601704797119, 0.022477917272181132, 0.023863743957173233, 0.024283259850336362, 0.021449119587870866, 0.01941129579656337, 0.018824164775408433, 0.019649505483998873, 0.019729050513594238, 0.019467567337130353, 0.009863530877285642, 0.008319727257849869]]
y = np.array(y)*100
yerr = np.array(yerr)*100
y = y[[2,1,3,0]];yerr = yerr[[2,1,3,0]]
start=2;y = y[:,start:];yerr = yerr[:,start:];x=[[0.1*i for i in range(1+start,21)] for _ in range(4)]
colors_tmp = colors#['k'] + colors[:2] + ['grey'] + colors[2:4]
linestyles_tmp = linestyles#['solid','dashed','dotted','solid','dashdot',(0, (3, 5, 1, 5)),]
methods_tmp = ['Replication','Standalone','Split','Ours']
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea.eps',
		'Deadline (s)','Delivered Accuracy (%)',lbsize=16,linewidth=2,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr)	

x = [[i for i in range(5,25)] for _ in range(3)]
y = [[0.8107558399999999, 3.4284124799999995, 10.97999784, 22.371772479999997, 35.817034719999995, 47.429199200000006, 56.09132407999999, 61.88740640000001, 67.27851023999999, 71.94365447999999, 76.23281984, 79.45724304000001, 81.85791112, 83.90299856, 85.73109911999998, 86.74704312, 87.64639344, 88.44354976000001, 89.07711472000001, 89.50449128], [0.41977728000000003, 1.9812919999999998, 6.079268319999999, 12.804502079999997, 20.932458399999994, 28.72923216, 35.286073599999995, 40.218603519999995, 45.33132447999999, 50.32745216, 54.93519968, 58.817387839999995, 62.51038559999999, 65.44322504, 69.0442284,71.36990184000001, 73.37839136, 75.29088663999998, 77.12638655999999, 78.55790816], [0.95474816, 3.8219919199999994, 11.801555999999996, 23.66390816, 37.112172560000005, 48.49955096, 56.82649495999999, 62.52638352, 67.98188552, 72.68282959999999, 76.84380232, 79.98943072, 82.30130424000001, 84.30679407999999, 86.19409135999999, 87.16523839999999, 88.0103916, 88.89054368000001, 89.46871208, 89.91028792000002]]
yerr = [[0.09959576198328121, 0.17668709286960838, 0.30307641874364033, 0.5804869653383868, 0.5990991196327445, 0.638665051047698, 0.8764608531915575, 0.7557494159787321, 0.6600721678757546, 0.5289821782620289, 0.39213258093077213, 0.3020933488802274, 0.3253069789276973, 0.30661995452472435, 0.26771376115544604, 0.24183179399757795, 0.2029552380300006, 0.1758850668591257, 0.16516863743796345, 0.15457537292594012], [0.22995632705842559, 0.1923929589916637, 0.5457179255272979, 0.7501392689421834, 0.8859903073181771, 0.9070374403595425, 0.9076571012136413, 0.8806301416161287, 0.8565438866454277, 0.7572900949915615, 0.8077192768442241, 1.0129573368256781, 0.872116772572183, 0.9473516123121386, 0.7400056693082095, 0.7599805637817056, 0.7489715147175358, 0.7053222141738541, 0.7006430845176033, 0.7804258055614768], [0.10951463583053361, 0.2185151974646195, 0.3305302662553735, 0.6610158226464041, 0.6853101128788982, 0.6623568954779163, 0.8428666893408425, 0.7834211506874251, 0.639088051442148, 0.5299022983794639, 0.3641402905760249, 0.28887398056230684, 0.33836100847137496, 0.2839030817756105, 0.25710027789149115, 0.21499646686019278, 0.19678684902766957, 0.15075224283835087, 0.16108527027541883, 0.1319121407883066]]
# y = \
# [[0.9527482399999998, 3.82659168, 11.816155199999999, 23.669707839999997, 37.10517296, 48.49395128, 56.82569496, 62.52358368, 67.98228552, 72.67742992000001, 76.85800160000001, 79.98463096, 82.29670448, 84.30679407999999, 86.19089152, 87.16683831999998, 88.01359144, 88.89074368000001, 89.470112, 89.90908800000003],
# [0.8113558399999998, 3.4240124800000005, 10.975397439999998, 22.363372079999998, 35.8026344, 47.4147988, 56.07372352, 61.84920631999999, 67.23531, 71.89945431999999, 76.18761984, 79.41204304, 81.81151128000002, 83.86039848, 85.69029895999999, 86.70704287999999, 87.60099344, 88.39814976000002, 89.03071472000002, 89.45729128000002], 
# [0.4191772799999999, 1.9860917599999997, 6.084867999999999, 12.797902399999998, 20.927858639999997, 28.729432160000005, 35.28787343999999, 40.21980352, 45.32772464, 50.32965208, 54.94639912, 58.810588239999994, 62.50858567999999, 65.44342504, 69.04382840000001, 71.36690200000001, 73.38259112, 75.29068663999999, 77.1213868, 78.560708], 
# ]
# yerr = \
# [[0.11535909628830483, 0.22316114788998928, 0.3319505427515128, 0.6684224420813447, 0.687086908807163, 0.6498107038480634, 0.8501591431050957, 0.7784922335867633, 0.647399544014218, 0.5299901446350953, 0.3632792759518867, 0.28872552536537005, 0.33768214084271714, 0.2839030817756105, 0.257687007288076, 0.20996273929474166, 0.19551776777777866, 0.1511714898035774, 0.16207031485376694, 0.132181114745732],
# [0.09860426520709135, 0.17826778208946675, 0.2995205596393919, 0.5802243520772641, 0.5986583623518704, 0.6518793665755521, 0.8780425083028963, 0.7598545508182218, 0.6529058731652871, 0.5372862371239481, 0.40029552923237954, 0.30537749546040943, 0.3215525493518599, 0.3091347386342381, 0.266972580217596, 0.23833641570403657, 0.20628548382233297, 0.183636992547902, 0.17350368482862988, 0.16410508007735], 
# [0.22742613175789975, 0.19785233779630296, 0.5503132230281367, 0.7436966758613143, 0.8840845002257762, 0.9044221319162175, 0.9104601466355116, 0.8776528903519115, 0.8549122796694266, 0.7596634984389306, 0.8046307635346153, 1.0117390262172763, 0.8708832045885845, 0.9446760371483859, 0.7392014174539016, 0.7620729000035643, 0.7402482943204561, 0.703640256478927, 0.6999962592360127, 0.7809538952671043], 
# ]
y = np.array(y);yerr = np.array(yerr)
y = y[[2,1,0]];yerr = yerr[[2,1,0]]
methods_tmp = ['Replication','Standalone','Ours']
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea2.eps',
		'Deadline (s)','Delivered Accuracy (%)',yticks=[0,25,50,75,100],lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr)
# FCC-2 nodes with loss
# cifar
y_ea_loss1 = [[0.9372004792332268, 0.9361701277955271, 0.9269009584664536, 0.9164237220447286, 0.9027735623003196, 0.8834584664536742, 0.8610463258785941, 0.8285802715654953, 0.8152276357827477, 0.759117412140575, 0.7229712460063897, 0.6756689297124601, 0.6312699680511182, 0.5680770766773162, 0.5175319488817891, 0.4741453674121406, 0.4014556709265175, 0.33900359424920135, 0.2573242811501598, 0.18030551118210864, 0.09999999999999999], [0.9411940894568691, 0.902811501597444, 0.8544488817891374, 0.8166713258785941, 0.7749001597444088, 0.7351876996805112, 0.6917911341853034, 0.6364616613418531, 0.6134464856230032, 0.5541433706070287, 0.5137739616613418, 0.46715055910543135, 0.43328873801916934, 0.39498402555910544, 0.3444468849840256, 0.32219249201277955, 0.2572484025559106, 0.23126397763578277, 0.18252396166134188, 0.14374400958466454, 0.09999999999999999], [0.9411940894568691, 0.9401437699680513, 0.9326417731629391, 0.9210263578274761, 0.9081449680511181, 0.8886900958466454, 0.8666873003194888, 0.8341613418530353, 0.8198901757188498, 0.7644788338658146, 0.7271046325878594, 0.6807507987220446, 0.6360423322683705, 0.5725099840255591, 0.5214756389776357, 0.4769908146964855, 0.4042711661341853, 0.34105031948881787, 0.25919129392971246, 0.18125399361022368, 0.09999999999999999]]
y_ea_loss1 += [[0.9008586261980831, 0.8992831469648562, 0.8935583067092653, 0.8811441693290734, 0.8696525559105431, 0.8490475239616613, 0.8290754792332269, 0.8009005591054313, 0.7716573482428115, 0.7384464856230031, 0.7078993610223642, 0.6664656549520767, 0.6096126198083066, 0.5554253194888179, 0.5133246805111822, 0.453202875399361, 0.38915734824281145, 0.3231869009584665, 0.24736222044728443, 0.17868011182108628, 0.09999999999999999]]
yerr_ea_loss1 = [[0.0, 0.002383087978836537, 0.003873038115141878, 0.0061453230457425975, 0.012967568025285573, 0.010063407266560125, 0.008824341552946188, 0.01611947452454557, 0.01890696132833872, 0.013174793019248588, 0.019867655361733523, 0.02406030887038924, 0.019344045784808497, 0.026841082314703498, 0.03579666570409922, 0.028536735524548405, 0.023743592117576182, 0.013090604293229717, 0.01902712161252409, 0.0103868882719809, 0.0], [1.1102230246251565e-16, 0.008922661919075328, 0.010855201934208752, 0.014047328679292339, 0.015535675044506014, 0.021138001603641914, 0.02019673453507974, 0.019017540855227846, 0.02697623136127924, 0.0165951113967507, 0.015869314630196252, 0.01722226165574532, 0.016248312393671693, 0.0225139485027758, 0.024553767381638156, 0.01572217242443465, 0.01882411807025787, 0.0094800814188465, 0.01617607815639621, 0.008617508759934122, 0.0], [1.1102230246251565e-16, 0.0021006389776357715, 0.004292221186753114, 0.005502080447956476, 0.012873009197567621, 0.009931106560055136, 0.008520796056967147, 0.016406406989810517, 0.019195156676365066, 0.014236333899824661, 0.019578295223714896, 0.025014428016724065, 0.02032100402565394, 0.026638863608938305, 0.03630411823945728, 0.02905046349658962, 0.0247702550563865, 0.013421182987465093, 0.019594189900178752, 0.009960898934023193, 0.0]]
yerr_ea_loss1 += [[0.0, 0.0017420439536039302, 0.0029415345396316193, 0.007147753771632083, 0.005778298588183379, 0.009994300024824121, 0.008848309404467163, 0.013715805473285277, 0.013925533885018195, 0.015674429694612985, 0.01324859410913445, 0.018769029529252034, 0.013349791966127953, 0.011702988574271404, 0.03150109653010247, 0.025379522111263788, 0.026080065096160693, 0.01758126474170829, 0.014596163094133774, 0.00866533841732982, 0.0]]
x = [[0.05*i for i in range(21)] for _ in range(4)]
y = np.array(y_ea_loss1);yerr = np.array(yerr_ea_loss1)
y = y[[2,1,3,0]];yerr = yerr[[2,1,3,0]]
methods_tmp = ['Replication','Standalone','Split','Ours']
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea.eps',
		'Loss Rate (%)','Delivered Accuracy (%)',lbsize=16,linewidth=2,markersize=0,linestyles=linestyles_tmp,yerr=yerr)

# imagenet 
y_ea_loss2 = [[91.202, 90.9202144, 90.3116468, 89.13271008000001, 87.39940504, 85.32331927999999, 82.99604520000001, 79.97001056, 76.32281008000001, 72.90759752, 68.06926231999998, 63.51791295999999, 58.560385199999985, 52.23993008, 46.47124616000001, 39.95400504, 33.0045872, 25.196417599999997, 17.173858799999998, 8.855113439999998, 0.004999999999999998], [91.52999999999999, 86.89205272000001, 82.34630159999999, 77.92174528000001, 73.00201320000001, 68.47845888, 64.19229320000002, 59.67814088, 54.96139704000001, 50.53783919999999, 45.76390135999999, 41.247146, 36.90458456, 31.92925479999999, 27.481100879999993, 23.128136319999996, 18.455192959999998, 13.857842719999997, 9.1880996, 4.77654032, 0.004999999999999998], [91.52999999999999, 91.26521440000002, 90.6738468, 89.52571007999998, 87.76840503999999, 85.69851928, 83.3802452, 80.34421056000001, 76.70561008000001, 73.25959752, 68.41686231999999, 63.82371296, 58.86998520000001, 52.52613008000001, 46.78404615999999, 40.17700504, 33.19478719999999, 25.3186176, 17.2604588, 8.91651344, 0.004999999999999998]]
yerr_ea_loss2 = [[0.0, 0.05798656128559632, 0.06900021804603022, 0.1228338706936923, 0.31604562206771536, 0.19983076434589445, 0.2140281106128623, 0.2771169737305441, 0.3270468516967591, 0.32448718252296105, 0.4462864029873493, 0.3885671789124424, 0.4244298189113099, 0.4852809003021273, 0.7224871398156378, 0.30742887213023684, 0.44779400478588055, 0.3540035702979286, 0.3387397810577788, 0.3706393955531091, 8.673617379884035e-19], [1.4210854715202004e-14, 0.2762135692915782, 0.3316258619645226, 0.20726475007837278, 0.3797153223580012, 0.4124907512014212, 0.25855713646514666, 0.7080915692230731, 0.5706039404536131, 0.5252931111981712, 0.493205716836166, 0.4429292890611953, 0.4431686400885038, 0.649461207386207, 0.449683869067782, 0.3142979423102246, 0.4108720301954161, 0.3847803022675996, 0.28580108537551757, 0.21610018183076485, 8.673617379884035e-19], [1.4210854715202004e-14, 0.058039641709164244, 0.05969244517022017, 0.11550548709734254, 0.31957228495000584, 0.2220576561775489, 0.2057506941240083, 0.31614598771070185, 0.3447872654794, 0.30477584225408944, 0.4337751146243707, 0.37902239653083, 0.4374639665859209, 0.456217583885193, 0.718906653455776, 0.3278329046012159, 0.4586741194379475, 0.3722928405753728, 0.34454717376735633, 0.3689043263286218, 8.673617379884035e-19]]
x = [[0.05*i for i in range(21)] for _ in range(3)]
y = np.array(y_ea_loss2);yerr = np.array(yerr_ea_loss2)
y = y[[2,1,0]];yerr = yerr[[2,1,0]]
methods_tmp = ['Replication','Standalone','Ours']
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea2.eps',
		'Loss Rate (%)','Delivered Accuracy (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr,ylim=(20,100))