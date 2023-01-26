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
		if use_barlabe_y and i==1:
			for k,xdx in enumerate(x_index):
				ax.text(xdx-0.02,data_mean[k,i]+.02,f'{data_mean[k,i]:.4f}',fontsize = 18, rotation='vertical',fontweight='bold')
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
exit(0)

# a = [0.9345047923322684, 0.9362020766773163, 0.9336062300319489, 0.9326078274760383, 0.9308107028753994, 0.9269169329073482, 0.9281150159744409, 0.9285143769968051]
# a = [0.9396964856230032, 0.9386980830670927, 0.9401956869009584, 0.9352036741214057, 0.9362020766773163, 0.9343051118210862]



# multi-node analysis
methods = ['Soft (se.)','Hard (se.)','Soft (rn.)','Hard (rn.)']

re_vs_nodes = [[0.10407947284345051, 0.03738313137189115,0.16654752396166136, 0.0137959207372693,0.007140575079872202, 0.004132095691370403,0.006259984025559107, 0.0013025614474167567,],
[0.1317492012779553, 0.03571484752205825,0.20654552715654956, 0.01022833202885187,0.0020946485623003054, 0.003389731212981527,0.0027156549520766736, 0.0006374157228239655],
[0.13577276357827478, 0.03598051353576017,0.21356230031948886, 0.008679869968650581,0.006569488817891345, 0.002178623274278881,0.005820686900958461, 0.0008257821278654765]]

re_vs_nodes = np.array(re_vs_nodes)
y = re_vs_nodes[:,0::2]
yerr = re_vs_nodes[:,1::2]
groupedbar(y,yerr,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_nodes.eps',methods=methods,
	envs=['Two-server','Three-server','Four-server'],width=0.25,bbox_to_anchor=(0.5, 1.25),use_barlabel_x=True,legloc='best',ncol=2,ylim=(0,0.3))

# soft
y = [[0.09999999999999999, 0.10055511182108626, 0.10055511182108626, 0.1275, 0.2615694888178913, 0.43260383386581475, 0.5827775559105431, 0.6732208466453673, 0.7437020766773162, 0.8037679712460063, 0.8377216453674119, 0.8562539936102237, 0.8874620607028753, 0.9001757188498403, 0.9079053514376996, 0.9165774760383385, 0.9192432108626198, 0.9202935303514378, 0.9301637380191693, 0.9328494408945687], [0.09999999999999999, 0.10054512779552716, 0.10054512779552716, 0.11802715654952076, 0.24427116613418529, 0.40908546325878586, 0.5655291533546325, 0.6578674121405751, 0.7310443290734823, 0.7912460063897764, 0.8270127795527156, 0.8501038338658147, 0.8780411341853034, 0.8949081469648563, 0.9026777156549523, 0.9101098242811501, 0.9141733226837061, 0.915491214057508, 0.9237559904153356, 0.9279872204472843]]
y += [[0.09999999999999999, 0.10055511182108626, 0.1032008785942492, 0.15934305111821087, 0.344866214057508, 0.5618730031948882, 0.7168909744408947, 0.7999500798722045, 0.8500079872204471, 0.8883706070287539, 0.9066353833865815, 0.9196485623003194, 0.9298462460063899, 0.933849840255591, 0.9357527955271566, 0.9376257987220447, 0.938180910543131, 0.9387060702875398, 0.9397364217252395, 0.9408266773162939], [0.09999999999999999, 0.10050519169329072, 0.10262579872204473, 0.1434644568690096, 0.322879392971246, 0.5332687699680511, 0.7021825079872204, 0.7901996805111822, 0.8436561501597446, 0.8847044728434504, 0.9040095846645366, 0.917755591054313, 0.9275079872204472, 0.9319768370607029, 0.9341373801916933, 0.9359704472843451, 0.9367631789137378, 0.937298322683706, 0.9375658945686901, 0.9393789936102236]]
y += [[0.09999999999999999, 0.10080271565495207, 0.10477635782747603, 0.18279153354632588, 0.4190055910543132, 0.6485982428115016, 0.79504392971246, 0.8599361022364217, 0.8927695686900957, 0.9171505591054313, 0.9266553514376996, 0.9332747603833864, 0.9383785942492013, 0.9397464057507987, 0.9400139776357828, 0.9405690894568689, 0.9405690894568689, 0.940836661341853, 0.940836661341853, 0.9410942492012779], [0.09999999999999999, 0.10080271565495207, 0.1032308306709265, 0.16350439297124603, 0.39160742811501603, 0.6263897763578274, 0.7821884984025559, 0.848101038338658, 0.8834904153354632, 0.9091373801916932, 0.9193750000000002, 0.9267372204472843, 0.9319588658146966, 0.9332767571884985, 0.9335642971246008, 0.9340994408945689, 0.9340994408945689, 0.9343670127795528, 0.9343670127795528, 0.9346345846645369]]
y += [[0.09999999999999999, 0.10026757188498403, 0.10026757188498403, 0.11975039936102234, 0.19217651757188503, 0.30001198083067093, 0.40552715654952076, 0.4770626996805111, 0.5468909744408946, 0.6152216453674121, 0.6571924920127795, 0.6927416134185304, 0.7455551118210864, 0.77620607028754, 0.804083466453674, 0.8346785143769967, 0.8493550319488816, 0.8557767571884984, 0.8749041533546326, 0.8889956070287539]]

yerr = [[0.0, 0.0011111211234452062, 0.0011111211234452062, 0.019736191416547883, 0.05781884086631109, 0.06330104623281653, 0.05911905270708192, 0.04226653417241109, 0.038654191520637676, 0.024175384800931236, 0.024436596247174074, 0.025130377087104926, 0.021527238078907263, 0.0199137496482411, 0.018989586062905652, 0.020391848164711325, 0.020533313186606467, 0.020248210447870275, 0.01021375164308125, 0.008686921346931619], [0.0, 0.0010904841391130044, 0.0010904841391130044, 0.012777161941904503, 0.055058518559993413, 0.06413397521289246, 0.060972396221046325, 0.04164513494875629, 0.03795789128381409, 0.02938632786258216, 0.024709733183780367, 0.025058686629537936, 0.025475035912879022, 0.020048597531399648, 0.01927089591910302, 0.02126664012592631, 0.021020402219154452, 0.021007606046009677, 0.012717516943128379, 0.009529858616401331]]
yerr += [[0.0, 0.0011102236421725309, 0.0070605574617650215, 0.037698284889912406, 0.07884444617397303, 0.07354891075030004, 0.057778843071216626, 0.04082228857233057, 0.026412660193391446, 0.016373359204643848, 0.01411313559028846, 0.010690884125923806, 0.008786786782335255, 0.006884396773024983, 0.006341296281823835, 0.004258329577667321, 0.004336842354278658, 0.0041518970931095625, 0.0021822257124341528, 0.0008027156549520573], [0.0, 0.001010629997433127, 0.006273078543585942, 0.02594276535785207, 0.08175422327877459, 0.07493148267312437, 0.06026493596123263, 0.045363954336851794, 0.02769775856276808, 0.01699831297060157, 0.01374522029207335, 0.011395568560453295, 0.010305836755633105, 0.008391278351580916, 0.007269106754491677, 0.005340841161023331, 0.005451654263246475, 0.005180287067751031, 0.00458431012492275, 0.0020666971492918082]]
yerr += [[0.0, 0.0012269810926168812, 0.007654537203337821, 0.04325089717935922, 0.08125000480921343, 0.06849467241494683, 0.04361631127857972, 0.028303426234784938, 0.022444976720748636, 0.013199795424242409, 0.009191775440817871, 0.004524296847070941, 0.002748419268439603, 0.0021768518736728507, 0.0021615809620174464, 0.0015754792332268283, 0.0015754792332268283, 0.0007727635782747377, 0.0007727635782747377, 0.0], [0.0, 0.0012269810926168812, 0.006283865420787238, 0.028471231332562876, 0.08278611290168393, 0.06763594722208092, 0.04702812049206951, 0.03237371355774275, 0.022910121879485077, 0.012771872692299555, 0.008964245292526185, 0.005057478091818651, 0.003270206571373641, 0.0024860544412008447, 0.0021176388874606447, 0.0021161245215707346, 0.0021161245215707346, 0.002006216836996953, 0.002006216836996953, 0.0022046618445279572]]
yerr += [[0.0, 0.0008027156549520825, 0.0008027156549520825, 0.014103284665866791, 0.04029656789614061, 0.060084307910632496, 0.06290831228461682, 0.053766013930545115, 0.04726965347279629, 0.050434924714486955, 0.04749330568151011, 0.044019340431446334, 0.040160865312640154, 0.04220955278933893, 0.035445069534178246, 0.02484295217493549, 0.024426260082436837, 0.02293679724493796, 0.022630677424858008, 0.018178147483001592]]
y,yerr = np.array(y)*100,np.array(yerr)*100

linestyles_tmp = ['solid','dashed','solid','dotted','solid','dashdot',(0, (3, 5, 1, 5))]
methods_tmp = ['Two-server (rn.)','Two-server (ours)','Three-server (rn.)','Three-server (ours)','Four-server (rn.)','Four-server (ours)','Standalone']
colors_tmp = colors + ['k']
x = [[0.1*i for i in range(1,21)]for _ in range(7)]
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/soft_ea_multi_nodes.eps',
		'Deadline (s)','Delivered Acc. (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,yticks=[0,25,50,75,100],
		yerr=yerr)	
# hard
y = [[0.9398362619808307, 0.9342771565495207, 0.9242951277955271, 0.9062500000000002, 0.8891833067092652, 0.8649161341853036, 0.8381170127795526, 0.793883785942492, 0.7700399361022364, 0.7238997603833865], [0.9339257188498402, 0.9286960862619807, 0.9183945686900958, 0.9002096645367412, 0.8840615015974441, 0.8586361821086262, 0.831986821086262, 0.7881829073482428, 0.7644888178913738, 0.7196365814696486]]
y += [[0.9410942492012779, 0.9405491214057508, 0.9391813099041533, 0.9362480031948882, 0.92560303514377, 0.9164776357827475, 0.9024221246006391, 0.8836461661341852, 0.8670527156549521, 0.8296505591054313], [0.9381888977635782, 0.9377635782747605, 0.9363957667731627, 0.9336022364217251, 0.921349840255591, 0.9129832268370608, 0.8984185303514376, 0.879153354632588, 0.8635982428115015, 0.826485623003195]]
y += [[0.9410942492012779, 0.9410942492012779, 0.9402715654952077, 0.9389536741214057, 0.9376058306709265, 0.936495607028754, 0.9284584664536741, 0.923604233226837, 0.9062999201277956, 0.883961661341853], [0.9374400958466454, 0.9362420127795528, 0.9357987220447284, 0.9336421725239619, 0.9325738817891374, 0.9322823482428115, 0.9237759584664536, 0.91676517571885, 0.8990015974440894, 0.8781309904153355]]
y += [[0.899530750798722, 0.864319089456869, 0.8097623801916931, 0.7743450479233227, 0.7402835463258784, 0.6900159744408946, 0.6552356230031948, 0.6029612619808307, 0.5585303514376998, 0.5106629392971247,]]

yerr = [[0.0013582678763340256, 0.0039556925979927545, 0.004574079076829142, 0.007590486345346805, 0.015893878949545526, 0.008896679967686099, 0.017192840928320387, 0.014187333482434289, 0.014443641571087571, 0.01663186781359779], [0.001475205237001879, 0.004508213305902468, 0.004439017678055025, 0.008275167299519711, 0.01683947507257339, 0.008939093454225104, 0.016890831389822156, 0.014066929523216762, 0.014858441919403578, 0.016256702072775]]
yerr += [[0.0, 0.0010923108020666826, 0.0024664512169118163, 0.004777525131303064, 0.008106767099446167, 0.007640653106418316, 0.010308598203723582, 0.013099504199730693, 0.017409532167523414, 0.011347709931139703], [0.0007396943467249182, 0.0015560942219585417, 0.003176126144845128, 0.005518208409388351, 0.008127942656767568, 0.007786525083377888, 0.009152282985960216, 0.01517180881190893, 0.015944338150294137, 0.012659946607470019]]
yerr += [[0.0, 0.0, 0.0012585195891636192, 0.0016091523802251612, 0.002666659027786383, 0.0030495607496831157, 0.005764723446053238, 0.00533855061296741, 0.006952284992081522, 0.005986219243912046], [0.0006862156660319099, 0.0007579973201318762, 0.0009281944036257917, 0.00183942062616284, 0.0017570534819607885, 0.002832889446300388, 0.005072404205391947, 0.006238002550570389, 0.007687061713705773, 0.006760096929571217]]
yerr += [[0.008043096537025946, 0.015663758082919074, 0.009781856671578172, 0.01189533459010569, 0.02179118313103497, 0.019705777563059995, 0.015619452745556574, 0.018995240954784595, 0.024058887877872404, 0.02215784373516351, ]]
y,yerr = np.array(y)*100,np.array(yerr)*100
x = [[5*i for i in range(1,11)]for _ in range(7)]
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_ea_multi_nodes.eps',
		'Loss Rate (%)','Delivered Acc. (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,yticks=[25,50,75,100],
		yerr=yerr)	

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


# baseline
flops_base = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
re_base = [[0.0007507987220447321, 0.003741558835808713, -0.000389376996805102, 0.0008822738262825615,],
[0.002939297124600637, 0.003882210251418619, 0.0010183706070287402, 0.0009027679357394152, ],
[0.004205271565495205, 0.003454625197475081, 0.003125, 0.001119858795400797, ],
[0.005842651757188488, 0.003802445462229222, 0.0056709265175718835, 0.0008597863475084146],
  [0.007140575079872202, 0.004132095691370403, 0.006259984025559107, 0.0013025614474167567,],
[0.008568290734824258, 0.004092656063333119, 0.009964057507987224, 0.0014678894528345747,],
[0.01164337060702877, 0.003929390299692293, 0.013807907348242799, 0.0016216282858508531,],
[0.010545127795527154, 0.003529936353411632, 0.019968051118210893, 0.0032287171673031755,],
]
re_base = np.array(re_base)

# no collaboration
flops_nolab = [0.8791134250074207,0.7660117691820428,0.6606950325238663,0.5631632150328911,0.4734163167091172,0.3914543375525446,0.3172772775631734,0.25088513674100354]
re_nolab = [[0.011182108626198084, 0.00021413349230757408,0.01144169329073481, 0.00020558366894942454],
[0.016533546325878567, 0.00030348810210801457,0.0167332268370607, 0.00014260040791818284],
[0.015113817891373792, 0.0008313738309228903,0.01519568690095845, 0.0003213553701963086],
[0.01707268370607029, 0.0003788671318092965,0.01736222044728435, 0.0002839748932373711],
[0.03075079872204469, 0.0002893645516411547,0.031030351437699666, 0.00017747992047354493],
[0.0427096645367412, 0.0008352972554826552,0.042601837060702884, 0.00018436686614036263],
[0.0324081469648562, 0.0003034881021080212,0.03258785942492006, 0.0001682537894005061],
[0.04006589456869007, 0.0004932755322298511,0.04030551118210859, 0.00010033821506712767]
]
re_nolab = np.array(re_nolab)

# no onn
flops_noonn = [1.006022295959533,0.8929206401341552,0.7876039034759786,0.6900720859850035,0.6003251876612296,0.518363208504657,0.44418614851528576,0.3777940076931159]
re_noonn = [[0.018488418530351415, 0.0008804733209712612,0.018510383386581453, 0.0002759639568906995],
[0.01887779552715656, 0.0007617857299742458,0.01879992012779551, 0.00016129686922325556],
[0.02206,0.00063,0.02223,0.00019],
[0.02424,0.0011,0.02418,0.00015],
[0.02258,0.0004,0.02277,0.00016],
[0.01949,0.00035,0.01976,0.00018],
[0.03415,0.000389,0.0344,0.000128],
[0.03685,0.0005458,0.03716,0.0001]
]

re_noonn = np.array(re_noonn)

# no bridge
flops_nobridge = [0.25088513674100354,0.3172, 0.3914543375525,0.4734,0.5632,0.6606950325238663,0.7660,0.8791134250074207]
re_nobridge = [
[0.019211261980830664, 0.004297730001922057,0.027935303514376987, 0.0015407780929600943],
[0.02037939297124599, 0.0033718607135043055,0.025748801916932917, 0.0013851913484271385],
[0.01936102236421724, 0.0031575948408859317,0.020257587859424907, 0.002005497255559385],
[0.014139376996805097, 0.0039418515955599925,0.012450079872204489, 0.0015128681623009163],
[0.013610223642172504, 0.003613138154618183,0.01082268370607029, 0.0009357327886469531],
[0.014778354632587842, 0.0030541648232636745,0.013658146964856233, 0.0009353065837564237],
[0.006561501597444075, 0.0039978531773773454,0.004203274760383358, 0.0016298443922219685],
[0.0032967252396166047, 0.003847802001265624,-0.0009384984025558984, 0.0009389231585869508],
]
re_nobridge = np.array(re_nobridge)

x = np.array([flops_base,flops_nobridge,flops_noonn,flops_nolab])*100
# Model Transform; Collaborative Training
methods_tmp = ['Ours (w/ C+T)','w/o Transform','w/o Collaboration','Standalone (w/o C+T)']
y = np.concatenate((re_base[:,0].reshape((1,8)),
					re_nobridge[:,0].reshape((1,8)),
					re_noonn[:,0].reshape((1,8)),
					re_nolab[:,0].reshape((1,8))))
yerr = np.concatenate((re_base[:,1].reshape((1,8)),
					re_nobridge[:,1].reshape((1,8)),
					re_noonn[:,1].reshape((1,8)),
					re_nolab[:,1].reshape((1,8))))
line_plot(x,y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/ablation_soft.eps',
		'FLOPS (%)','$R^{(2)}$',lbsize=16,linewidth=4,markersize=8,linestyles=linestyles,
		yerr=yerr)	
y = np.concatenate((re_base[:,2].reshape((1,8)),
					re_nobridge[:,2].reshape((1,8)),
					re_noonn[:,2].reshape((1,8)),
					re_nolab[:,2].reshape((1,8))))
yerr = np.concatenate((re_base[:,3].reshape((1,8)),
					re_nobridge[:,3].reshape((1,8)),
					re_noonn[:,3].reshape((1,8)),
					re_nolab[:,3].reshape((1,8))))
line_plot(x,y,methods_tmp,colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/ablation_hard.eps',
		'FLOPS (%)','$R^{(2)}$',lbsize=16,linewidth=4,markersize=8,linestyles=linestyles,
		yerr=yerr)	

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


# FCC 2-nodes no loss analysis
# CIFAR, IMAGENET EA and FR FCC
x = [[0.1*i for i in range(1,21)] for _ in range(3)]
y = \
[[0.09999999999999999, 0.10055511182108626, 0.10055511182108626, 0.12888777955271566, 0.2629872204472844, 0.43451677316293924, 0.5849580670926517, 0.6748063099041535, 0.7455650958466453, 0.8045607028753994, 0.8371765175718849, 0.8568091054313098, 0.8890674920127795, 0.9001757188498403, 0.9084305111821086, 0.9165774760383385, 0.9192432108626198, 0.9202935303514378, 0.9301637380191693, 0.9328494408945687],
[0.09999999999999999, 0.10054512779552716, 0.10054512779552716, 0.11828474440894568, 0.24496605431309906, 0.4097404153354631, 0.5658386581469649, 0.6580571086261983, 0.7310143769968052, 0.7913458466453673, 0.8267452076677315, 0.8503115015974441, 0.8783286741214058, 0.8949680511182109, 0.9026976837060703, 0.9101497603833867, 0.9141833067092652, 0.9155111821086261, 0.9237859424920126, 0.9279472843450478], 
[0.09999999999999999, 0.10026757188498403, 0.10026757188498403, 0.11975039936102234, 0.19217651757188503, 0.30001198083067093, 0.40552715654952076, 0.4770626996805111, 0.5468909744408946, 0.6152216453674121, 0.6571924920127795, 0.6927416134185304, 0.7455551118210864, 0.77620607028754, 0.804083466453674, 0.8346785143769967, 0.8493550319488816, 0.8557767571884984, 0.8749041533546326, 0.8889956070287539], 
]
yerr = \
[[0.0, 0.0011111211234452062, 0.0011111211234452062, 0.019928147128896512, 0.05780761020699797, 0.06396649670392815, 0.05917607804754459, 0.041502195261175594, 0.03821855193538518, 0.023126224413790246, 0.024887560511660814, 0.025361025458283948, 0.022115593809318868, 0.0199137496482411, 0.019387677509606134, 0.020391848164711325, 0.020533313186606467, 0.020248210447870275, 0.01021375164308125, 0.008686921346931619],
[0.0, 0.0010904841391130044, 0.0010904841391130044, 0.012357426669505291, 0.054528611589075536, 0.06381844043036748, 0.06071610397413788, 0.04267753235171702, 0.038135808543041716, 0.029169540946201715, 0.024594239616715332, 0.025323665071670966, 0.025300140210057387, 0.020082745682600975, 0.01908335362768818, 0.0208702609473867, 0.02075241049544623, 0.02068069913944312, 0.01250327165281592, 0.00917044655528695], 
[0.0, 0.0008027156549520825, 0.0008027156549520825, 0.014103284665866791, 0.04029656789614061, 0.060084307910632496, 0.06290831228461682, 0.053766013930545115, 0.04726965347279629, 0.050434924714486955, 0.04749330568151011, 0.044019340431446334, 0.040160865312640154, 0.04220955278933893, 0.035445069534178246, 0.02484295217493549, 0.024426260082436837, 0.02293679724493796, 0.022630677424858008, 0.018178147483001592], 
]
y = np.array(y)*100
yerr = np.array(yerr)*100
colors_tmp = colors#['k'] + colors[:2] + ['grey'] + colors[2:4]
linestyles_tmp = ['solid','dashed','dotted','solid','dashdot',(0, (3, 5, 1, 5)),]
methods_tmp = ['Replication','Partition (Ours)','Standalone']
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea.eps',
		'Deadline (s)','Delivered Acc. (%)',xticks=[0.5,1,1.5,2.0],yticks=[0,25,50,75,100],lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr)	

x = [[i for i in range(5,25)] for _ in range(3)]
y = \
[[0.9527482399999998, 3.82659168, 11.816155199999999, 23.669707839999997, 37.10517296, 48.49395128, 56.82569496, 62.52358368, 67.98228552, 72.67742992000001, 76.85800160000001, 79.98463096, 82.29670448, 84.30679407999999, 86.19089152, 87.16683831999998, 88.01359144, 88.89074368000001, 89.470112, 89.90908800000003],
[0.8113558399999998, 3.4240124800000005, 10.975397439999998, 22.363372079999998, 35.8026344, 47.4147988, 56.07372352, 61.84920631999999, 67.23531, 71.89945431999999, 76.18761984, 79.41204304, 81.81151128000002, 83.86039848, 85.69029895999999, 86.70704287999999, 87.60099344, 88.39814976000002, 89.03071472000002, 89.45729128000002], 
[0.4191772799999999, 1.9860917599999997, 6.084867999999999, 12.797902399999998, 20.927858639999997, 28.729432160000005, 35.28787343999999, 40.21980352, 45.32772464, 50.32965208, 54.94639912, 58.810588239999994, 62.50858567999999, 65.44342504, 69.04382840000001, 71.36690200000001, 73.38259112, 75.29068663999999, 77.1213868, 78.560708], 
]
yerr = \
[[0.11535909628830483, 0.22316114788998928, 0.3319505427515128, 0.6684224420813447, 0.687086908807163, 0.6498107038480634, 0.8501591431050957, 0.7784922335867633, 0.647399544014218, 0.5299901446350953, 0.3632792759518867, 0.28872552536537005, 0.33768214084271714, 0.2839030817756105, 0.257687007288076, 0.20996273929474166, 0.19551776777777866, 0.1511714898035774, 0.16207031485376694, 0.132181114745732],
[0.09860426520709135, 0.17826778208946675, 0.2995205596393919, 0.5802243520772641, 0.5986583623518704, 0.6518793665755521, 0.8780425083028963, 0.7598545508182218, 0.6529058731652871, 0.5372862371239481, 0.40029552923237954, 0.30537749546040943, 0.3215525493518599, 0.3091347386342381, 0.266972580217596, 0.23833641570403657, 0.20628548382233297, 0.183636992547902, 0.17350368482862988, 0.16410508007735], 
[0.22742613175789975, 0.19785233779630296, 0.5503132230281367, 0.7436966758613143, 0.8840845002257762, 0.9044221319162175, 0.9104601466355116, 0.8776528903519115, 0.8549122796694266, 0.7596634984389306, 0.8046307635346153, 1.0117390262172763, 0.8708832045885845, 0.9446760371483859, 0.7392014174539016, 0.7620729000035643, 0.7402482943204561, 0.703640256478927, 0.6999962592360127, 0.7809538952671043], 
]
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea2.eps',
		'Deadline (s)','Delivered Acc. (%)',yticks=[0,25,50,75,100],lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr)
# FCC-2 nodes with loss
# cifar
y_ea_loss1 = \
[[0.9395886581469648, 0.9355451277955271, 0.9231968849840255, 0.9064576677316294, 0.894061501597444, 0.8674420926517572, 0.8352396166134184, 0.8100998402555911, 0.7676218051118211, 0.7180930511182109, 0.684185303514377, 0.6422444089456868, 0.5829652555910543, 0.5285363418530351, 0.4677735623003195, 0.40289137380191686, 0.3329552715654952, 0.26033945686900967, 0.1833566293929713, 0.09999999999999999],
[0.9363238817891373, 0.9306429712460066, 0.9192432108626196, 0.9021845047923325, 0.8896485623003196, 0.8623801916932907, 0.8314756389776357, 0.8047384185303512, 0.7634784345047922, 0.7133206869009585, 0.6802715654952076, 0.6388797923322684, 0.5790415335463258, 0.525680910543131, 0.4648482428115016, 0.400245607028754, 0.3308885782747604, 0.2592911341853036, 0.18273761980830674, 0.09999999999999999], 
[0.899530750798722, 0.864319089456869, 0.8097623801916931, 0.7743450479233227, 0.7402835463258784, 0.6900159744408946, 0.6552356230031948, 0.6029612619808307, 0.5585303514376998, 0.5106629392971247, 0.4682667731629393, 0.44467252396166135, 0.39341054313099044, 0.3471585463258786, 0.30806709265175714, 0.2670806709265176, 0.22613218849840258, 0.1816713258785943, 0.13969249201277956, 0.09999999999999999], 
]
yerr_ea_loss1 = \
[[0.0021534818733086004, 0.004807122826269942, 0.004066546925011879, 0.00879812907368698, 0.011142956904382063, 0.011582885400136855, 0.012580350160807219, 0.009566383824699317, 0.020455907886160887, 0.014106625978565771, 0.01455718586106292, 0.017206874589478307, 0.014501868988785572, 0.027189121150484708, 0.021631105912234414, 0.020220537895996683, 0.020388440563463065, 0.012831811456275988, 0.01049326026015616, 0.0],
[0.0019371366494721324, 0.004588458288314903, 0.00414439823277979, 0.009328342602611927, 0.012013633873460319, 0.012058123944065857, 0.012289293382034258, 0.008732748359571248, 0.02004561015701889, 0.015600573185774222, 0.013672572394422981, 0.01708082417607047, 0.013641807178555848, 0.027454502496626478, 0.021524453610169147, 0.020367149911392154, 0.020200508844598793, 0.012975362073228098, 0.00988700564048156, 0.0], 
[0.008043096537025946, 0.015663758082919074, 0.009781856671578172, 0.01189533459010569, 0.02179118313103497, 0.019705777563059995, 0.015619452745556574, 0.018995240954784595, 0.024058887877872404, 0.02215784373516351, 0.02638469773508022, 0.024478237292861006, 0.011468919204652697, 0.02143756251131583, 0.015610426733831414, 0.011698637512988467, 0.01863280265939512, 0.016299024806537422, 0.010632914285249866, 0.0], 
]
# imagenet 
y_ea_loss2 = \
[[91.33101080000003, 90.60644991999999, 89.49291112, 87.95659496, 85.86170872, 83.21465423999999, 80.28001616, 76.86200176000001, 72.96281408000002, 69.01183128000001],
[90.9514108, 90.22024992, 89.09331112, 87.56459496, 85.46870872000002, 82.81345424000001, 79.88661616000002, 76.49360176, 72.60441408, 68.65863128], 
[87.02364624000002, 82.06531632, 77.92634296000001, 73.15500176000002, 68.793842, 63.92470848000001, 59.51935024000001, 55.212184560000004, 50.36724807999999, 45.92429256], 
]
yerr_ea_loss2 = \
[[0.05626240955778068, 0.09197665318038978, 0.17145129561051317, 0.24627422058304022, 0.29693415323216665, 0.2452078770150864, 0.31490016831645656, 0.40300899507274346, 0.4287114029382766, 0.3764830034422406],
[0.05532915478522786, 0.10402358606713132, 0.1756619652358583, 0.24289069550681458, 0.30163692611433485, 0.24904375295567502, 0.3149205898218357, 0.3826702392367903, 0.4426030997104579, 0.3638727618177673], 
[0.22001012256037583, 0.31734498384907517, 0.36730227485946276, 0.42945146053601585, 0.326857638718318, 0.34949152028904357, 0.30815160200411656, 0.4359560304021927, 0.3180484905611396, 0.438570039518222], 
]
methods_tmp = ['CIFAR-10 (rn.)','CIFAR-10 (ours)','CIFAR-10 (se.)','ImageNet (rn.)','ImageNet (ours)','ImageNet (se.)']
x = [[0.05*i for i in range(1,11)] for _ in range(6)]
y = np.concatenate((np.array(y_ea_loss1)[:,:10]*100,np.array(y_ea_loss2)),axis=0)
yerr = np.concatenate((np.array(yerr_ea_loss1)[:,:10]*100,np.array(yerr_ea_loss2)),axis=0)
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea2.eps',
		'Loss Rate (%)','Delivered Acc. (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr,ylim=(20,100))


envs = ['ResNet-20','ResNet-32','ResNet-44','ResNet-56','ResNet-110']
methods_tmp = ['Standalone','Ours']
# flops_res = [0.8677293603320655,0.7053025247728113,0.6375213554749708,0.6003251876612296,0.5360070051652971]
# acc_par = [[0.9265175718849841, 0.922923322683706, 0.9158346645367412, 0.9186301916932907],
# [0.9321086261980831, 0.9285143769968051, 0.9230231629392971, 0.9238218849840255],
# [0.9306110223642172, 0.9298123003194888, 0.9238218849840255, 0.9258186900958466],
# [0.9385982428115016, 0.9366014376996805, 0.9339057507987221, 0.9327076677316294],
# [0.9389976038338658, 0.9381988817891374, 0.9377995207667732, 0.9354033546325878]
# ]
# acc_base = [0.9243,0.9336,0.9379,0.9411,0.9446]
# acc_par = np.array(acc_par)
# acc_base = np.array(acc_base)
# acc_par_mean = acc_par.mean(axis=1)
# acc_par_std = acc_par.std(axis=1)
# y = np.concatenate((acc_base.reshape(5,1),acc_par_mean.reshape(5,1)),axis=1)*100
# yerr = np.concatenate((np.zeros((5,1)),acc_par_std.reshape(5,1)),axis=1)*100
# groupedbar(y,yerr,'Lossless Accuracy (%)', 
# 	'/home/bo/Dropbox/Research/SIGCOMM23/images/acc_res.eps',methods=methods_tmp,
# 	envs=envs,ncol=1,sep=.4,legloc='best',ylim=(90,95),rotation=30)

soft = [[0.002106629392971249, 0.003906847560424466,0.10148961661341853, 0.03568281929651537],
[0.006741214057507961, 0.003244131780051274,0.10220646964856232, 0.03583679685776256],
[0.011801118210862604, 0.00439929293904876,0.10365015974440901, 0.036999108780712796],
[0.007819488817891362, 0.003966352666503437,0.10384185303514379, 0.037132869618767474],
[0.011465654952076687, 0.002785178815481112,0.10562699680511187, 0.0382304333102761]
]

soft = np.array(soft)
y = soft[:,::2][:,::-1]
yerr = soft[:,1::2][:,::-1]
groupedbar(y,yerr,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/soft_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,sep=.4,legloc='best',use_barlabe_y=True,rotation=30)

hard = [[0.003025159744408923, 0.0021207771828151522,0.157468051118211, 0.009998619525461857],
[0.007108626198083057, 0.0010011941802493707,0.16112420127795526, 0.017046921714564307],
[0.011501597444089495, 0.0006899821303183288,0.16940495207667744, 0.01661585867739657],
[0.007807507987220452, 0.000714119598332561,0.1538378594249202, 0.013789771291098912],
[0.009834265175718882, 0.0015737117399115027,0.1666693290734825, 0.017363645480709185]]

hard = np.array(hard)
y = hard[:,::2][:,::-1]
yerr = hard[:,1::2][:,::-1]
groupedbar(y,yerr,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,sep=.4,legloc='best',use_barlabe_y=True,rotation=30)