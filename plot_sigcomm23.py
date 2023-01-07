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
markers = ['o','^','s','>','P','D']
hatches = ['/' ,'\\','--','x', '+', 'O','-',]
linestyles = ['solid','dashed','dotted','dashdot',(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1))]
methods3 = ['Replication (Optimal)','Partition (Ours)','Standalone']
methods6 = ['Ours','Baseline','Optimal$^{(2)}$','Ours*','Baseline*','Optimal$^{(2)}$*']


def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,legloc='best',linestyles=linestyles,
				xticks=None,yticks=None,ncol=None, yerr=None, xticklabel=None,yticklabel=None,xlim=None,ylim=None,ratio=None,
				use_arrow=False,arrow_coord=(0.4,30),markersize=8,bbox_to_anchor=None,get_ax=0,linewidth=2,logx=False,use_doublearrow=False,rotation=None):
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
				# linestyle = linestyles[i], 
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
			ax.text(x_index[1]-0.03,data_mean[1,i]+.01,f'{data_mean[1,i]:.4f}',fontsize = 14, rotation='vertical',fontweight='bold')
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

x = [[0.1*i for i in range(11)]for _ in range(4)]
y = [[1-0.1*i for i in range(11)],
[1-(0.1*i)**2 for i in range(11)],
[1-(0.1*i)**3 for i in range(11)],
[1-(0.1*i)**4 for i in range(11)]
]
line_plot(x, y,['1','2','3','4'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/test.eps',
		'Loss Rate (%)','Probability (%)',lbsize=20)
exit(0)
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

# 1. loss brings down accuracy
x = [[5*i for i in range(6)]for _ in range(4)]
y = [[94.12,89.984824,85.529952, 81.327276,75.978435,72.654353],
[94.12,93.84984,93.394968,92.176917,91.282947,89.128594],
[91.53,87.02364624000002, 82.06531632, 77.92634296000001, 73.15500176000002, 68.793842],
[91.53,91.33101080000003, 90.60644991999999, 89.49291112, 87.95659496, 85.86170872]] 
yerr = [[0,0.779019, 1.106963, 1.908686, 2.466583, 0.698803],
[0,0.212719,0.429682, 0.626182, 0.943877, 1.152293],
[0,0.22001012256037583, 0.31734498384907517, 0.36730227485946276, 0.42945146053601585,0.326857638718318],
[0,0.05626240955778068, 0.09197665318038978, 0.17145129561051317, 0.24627422058304022,0.29693415323216665]]
line_plot(x, y,['CIFAR-10 (se.)','CIFAR-10 (rn.)','ImageNet (se.)','ImageNet (rn.)'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/mot1.eps',
		'Loss Rate (%)','Accuracy (%)',yerr=yerr,lbsize=20)
# 2. replication helps but hurts flops
# replication makes sure consistent performance with one or two servers
# this is not necessary
# with even 10% loss, the chance is 4.5X
# simply with pruning does not change this fact, we want to exploit beyond replication: partitioning
# 
x = [[10*i for i in range(11)]for _ in range(3)]
y = [[(0.1*i)**2*100 for i in range(11)],
[(1-.1*i)*(2*0.1*i)*100 for i in range(11)],
[(1-.1*i)**2*100 for i in range(11)]
]
line_plot(x, y,['None alive','One alive','Two alive'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/mot2.eps',
		'Loss Rate (%)','Probability (%)',use_doublearrow=True,lbsize=20)
# 3. partition
# two capacity, different accuracy cdf
# our_correctness = [[1.0, 0.9375, 0.96875, 0.9375, 0.9375, 0.96875, 0.96875, 0.875, 0.9375, 0.90625, 0.90625, 0.96875, 0.90625, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.875, 0.9375, 1.0, 0.96875, 0.90625, 0.96875, 0.875, 0.9375, 0.96875, 0.9375, 0.9375, 1.0, 0.9375, 0.9375, 1.0, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.90625, 0.96875, 0.96875, 0.96875, 0.90625, 1.0, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.9375, 1.0, 0.9375, 0.9375, 0.96875, 0.90625, 0.90625, 0.96875, 0.84375, 0.90625, 0.96875, 1.0, 0.90625, 0.90625, 1.0, 0.9375, 1.0, 0.875, 0.96875, 0.96875, 0.9375, 1.0, 0.9375, 0.90625, 0.875, 0.9375, 0.96875, 0.875, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.96875, 0.9375, 0.96875, 0.96875, 0.875, 0.96875, 0.9375, 0.78125, 0.90625, 0.9375, 0.9375, 0.84375, 0.90625, 1.0, 1.0, 1.0, 0.9375, 0.84375, 0.96875, 0.90625, 1.0, 0.96875, 0.96875, 0.96875, 0.9375, 0.875, 0.96875, 0.96875, 0.90625, 0.96875, 0.9375, 0.96875, 0.9375, 1.0, 0.9375, 0.96875, 0.96875, 0.9375, 0.90625, 1.0, 0.9375, 0.90625, 1.0, 0.96875, 0.90625, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.9375, 0.90625, 1.0, 0.96875, 0.9375, 1.0, 1.0, 1.0, 0.9375, 0.875, 0.875, 0.90625, 0.9375, 0.96875, 0.90625, 0.9375, 0.9375, 0.90625, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.9375, 0.96875, 1.0, 0.90625, 0.90625, 0.90625, 0.90625, 0.875, 0.9375, 0.96875, 0.96875, 0.84375, 0.96875, 0.96875, 0.90625, 0.84375, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.90625, 0.90625, 0.9375, 0.9375, 0.96875, 0.875, 1.0, 0.96875, 0.96875, 0.9375, 0.96875, 0.90625, 1.0, 0.96875, 0.90625, 0.875, 0.9375, 0.875, 0.9375, 0.9375, 1.0, 0.9375, 1.0, 0.9375, 1.0, 0.96875, 1.0, 1.0, 0.96875, 0.96875, 0.9375, 0.96875, 0.875, 0.96875, 0.96875, 1.0, 0.9375, 0.90625, 0.9375, 0.96875, 0.84375, 0.90625, 0.9375, 0.96875, 0.90625, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 0.96875, 0.96875, 0.90625, 0.9375, 0.90625, 0.9375, 0.90625, 0.84375, 0.96875, 0.9375, 0.96875, 1.0, 0.9375, 0.96875, 0.96875, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625, 0.96875, 0.9375, 1.0, 0.9375, 0.90625, 0.9375, 1.0, 0.9375, 0.96875, 1.0, 0.96875, 0.9375, 0.9375, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625, 0.90625, 0.9375, 0.96875, 0.9375, 1.0, 0.90625, 1.0, 0.96875, 0.96875, 0.9375, 0.96875, 0.90625, 0.9375, 0.9375, 0.9375, 0.90625, 0.90625, 0.96875, 0.875, 0.96875, 0.90625, 0.9375, 0.9375, 0.9375, 0.90625, 1.0, 0.9375, 0.9375, 0.9375, 0.8125, 0.90625, 0.9375, 0.84375, 0.90625, 0.84375, 0.96875, 0.96875, 0.875, 0.90625, 1.0, 0.96875, 0.875], [0.96875, 0.9375, 0.96875, 1.0, 0.90625, 1.0, 0.96875, 0.875, 0.9375, 0.875, 0.875, 0.8125, 0.90625, 0.90625, 0.9375, 0.96875, 1.0, 0.9375, 0.84375, 0.90625, 0.875, 0.9375, 1.0, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.90625, 0.96875, 0.9375, 0.84375, 0.78125, 0.9375, 0.9375, 0.90625, 0.9375, 0.96875, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 1.0, 0.96875, 0.96875, 0.9375, 0.9375, 0.84375, 0.90625, 0.875, 0.90625, 0.96875, 0.96875, 0.90625, 0.9375, 1.0, 0.96875, 0.9375, 0.9375, 0.96875, 0.9375, 0.96875, 0.96875, 0.96875, 0.9375, 0.96875, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.9375, 0.96875, 0.90625, 0.90625, 0.9375, 0.96875, 0.90625, 0.9375, 0.90625, 0.90625, 0.875, 0.90625, 0.96875, 1.0, 0.90625, 0.96875, 0.9375, 0.96875, 0.96875, 0.90625, 0.90625, 0.875, 0.9375, 0.875, 0.96875, 0.96875, 0.90625, 0.9375, 0.9375, 0.84375, 0.96875, 0.96875, 0.90625, 0.9375, 1.0, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.8125, 0.9375, 0.9375, 0.9375, 0.9375, 0.96875, 0.875, 0.875, 0.90625, 0.96875, 1.0, 0.96875, 0.9375, 0.96875, 0.96875, 0.9375, 0.9375, 0.9375, 0.96875, 0.875, 0.9375, 1.0, 0.84375, 0.96875, 0.90625, 0.90625, 0.9375, 0.96875, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.96875, 0.90625, 0.90625, 1.0, 0.90625, 1.0, 0.875, 0.875, 0.875, 0.90625, 0.96875, 0.9375, 0.96875, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.90625, 0.90625, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.9375, 0.96875, 1.0, 0.96875, 1.0, 0.9375, 0.96875, 0.9375, 0.90625, 0.96875, 1.0, 1.0, 0.96875, 1.0, 0.90625, 0.90625, 0.96875, 0.9375, 0.90625, 0.96875, 0.9375, 0.96875, 0.90625, 1.0, 0.84375, 0.9375, 0.8125, 0.9375, 0.96875, 0.9375, 0.96875, 1.0, 0.9375, 0.875, 1.0, 0.96875, 1.0, 0.9375, 1.0, 0.90625, 0.96875, 0.9375, 0.875, 0.96875, 0.96875, 0.90625, 1.0, 0.96875, 0.90625, 0.96875, 1.0, 0.875, 0.875, 0.96875, 0.84375, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.875, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 1.0, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.96875, 1.0, 0.875, 0.9375, 0.9375, 0.875, 0.96875, 1.0, 1.0, 0.9375, 0.875, 0.96875, 0.90625, 0.90625, 0.90625, 0.90625, 0.96875, 0.9375, 0.9375, 0.9375, 0.96875, 0.9375, 0.96875, 1.0, 1.0, 1.0, 1.0, 0.96875, 0.96875, 0.96875, 1.0, 0.9375, 0.96875, 0.9375, 0.96875, 0.875, 0.9375, 0.875, 0.96875, 0.96875, 0.96875, 0.9375, 1.0, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 0.9375, 1.0, 0.90625, 0.90625, 1.0, 0.9375, 0.96875, 0.90625, 0.9375], [0.875, 0.96875, 0.9375, 0.9375, 0.96875, 0.90625, 0.9375, 0.84375, 0.96875, 0.96875, 1.0, 0.9375, 0.9375, 0.9375, 0.9375, 0.875, 1.0, 0.875, 0.9375, 0.96875, 0.9375, 1.0, 0.9375, 0.84375, 0.9375, 0.90625, 0.9375, 0.9375, 0.9375, 0.9375, 0.90625, 0.9375, 0.90625, 0.875, 0.90625, 0.9375, 0.9375, 0.9375, 0.84375, 0.875, 0.90625, 1.0, 0.90625, 0.96875, 0.90625, 0.90625, 0.9375, 0.9375, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.875, 0.96875, 0.875, 0.90625, 0.96875, 1.0, 0.9375, 0.90625, 0.9375, 0.90625, 0.875, 0.90625, 0.96875, 0.96875, 0.9375, 0.96875, 0.96875, 0.9375, 0.9375, 0.96875, 0.875, 0.9375, 0.9375, 0.84375, 1.0, 0.96875, 0.875, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.96875, 0.96875, 0.96875, 0.9375, 0.90625, 0.90625, 0.90625, 0.9375, 0.875, 0.8125, 0.9375, 1.0, 0.90625, 0.96875, 0.96875, 0.90625, 0.9375, 0.875, 0.875, 1.0, 0.96875, 0.875, 1.0, 0.96875, 0.96875, 0.84375, 0.90625, 0.90625, 0.96875, 0.875, 0.90625, 0.9375, 0.96875, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 0.90625, 0.90625, 0.90625, 0.96875, 0.84375, 0.9375, 0.9375, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.96875, 0.90625, 0.875, 0.96875, 0.96875, 0.96875, 1.0, 0.96875, 0.9375, 0.90625, 0.875, 0.71875, 0.9375, 0.875, 1.0, 0.9375, 0.9375, 0.84375, 1.0, 0.9375, 0.96875, 0.875, 0.9375, 0.9375, 0.90625, 0.84375, 0.90625, 0.9375, 0.9375, 1.0, 0.9375, 0.78125, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.875, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.90625, 0.875, 0.90625, 0.875, 0.96875, 0.84375, 0.96875, 0.96875, 0.9375, 1.0, 0.96875, 0.9375, 0.84375, 0.875, 0.9375, 0.90625, 0.90625, 0.96875, 0.9375, 0.90625, 0.9375, 0.9375, 0.9375, 0.9375, 0.96875, 0.875, 0.90625, 0.90625, 0.875, 0.9375, 0.9375, 0.90625, 0.9375, 0.90625, 0.9375, 0.875, 1.0, 0.90625, 0.96875, 0.875, 0.9375, 0.96875, 0.96875, 0.96875, 0.90625, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.90625, 0.96875, 0.9375, 0.875, 0.96875, 0.9375, 0.90625, 0.96875, 0.78125, 0.875, 0.9375, 0.90625, 0.90625, 0.96875, 0.90625, 0.9375, 0.96875, 0.9375, 0.9375, 0.90625, 0.8125, 1.0, 0.96875, 0.9375, 0.90625, 0.96875, 0.90625, 0.96875, 0.96875, 0.90625, 1.0, 1.0, 0.9375, 0.875, 0.90625, 1.0, 0.9375, 0.84375, 0.9375, 0.96875, 0.96875, 1.0, 0.96875, 0.96875, 0.84375, 0.875, 0.875, 0.96875, 0.90625, 0.84375, 0.875, 0.96875, 0.96875, 0.9375, 0.96875, 1.0, 0.90625, 0.96875, 0.90625, 0.90625, 0.90625, 0.9375, 0.9375, 0.84375, 0.9375, 0.9375, 0.84375, 0.96875, 0.84375, 0.90625, 0.9375, 0.875, 0.84375, 0.90625, 0.96875, 0.9375, 0.9375, 0.9375], [0.96875, 0.90625, 0.90625, 0.9375, 0.875, 0.8125, 0.96875, 0.9375, 0.84375, 0.84375, 0.875, 0.96875, 0.9375, 0.90625, 0.78125, 0.90625, 0.875, 0.90625, 0.875, 0.9375, 0.90625, 0.9375, 0.78125, 0.9375, 0.90625, 0.90625, 0.875, 0.90625, 0.9375, 0.84375, 0.9375, 0.78125, 0.84375, 0.96875, 0.875, 0.875, 0.9375, 0.8125, 0.90625, 0.90625, 0.90625, 0.9375, 0.96875, 0.875, 1.0, 0.90625, 0.96875, 0.9375, 0.875, 0.90625, 0.9375, 0.90625, 0.96875, 0.90625, 0.78125, 0.90625, 0.9375, 0.875, 0.90625, 0.9375, 0.9375, 1.0, 0.9375, 0.90625, 0.9375, 0.9375, 0.84375, 0.96875, 0.875, 0.96875, 0.9375, 0.90625, 0.84375, 0.90625, 0.75, 0.875, 0.96875, 0.96875, 0.90625, 0.875, 0.875, 0.875, 0.90625, 0.8125, 0.875, 0.96875, 0.8125, 0.90625, 0.9375, 0.90625, 0.84375, 0.90625, 0.9375, 0.9375, 0.875, 0.84375, 0.96875, 0.84375, 0.875, 0.9375, 0.84375, 0.96875, 0.875, 0.90625, 0.875, 0.90625, 0.90625, 1.0, 0.96875, 0.875, 0.75, 0.90625, 0.875, 0.90625, 0.9375, 0.84375, 0.90625, 1.0, 0.90625, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.9375, 0.84375, 0.96875, 0.9375, 0.875, 0.875, 1.0, 0.84375, 0.9375, 0.96875, 0.9375, 0.8125, 0.875, 0.9375, 0.9375, 0.84375, 0.84375, 0.90625, 0.84375, 0.96875, 0.96875, 1.0, 0.84375, 0.9375, 0.96875, 0.90625, 0.875, 0.9375, 0.90625, 0.96875, 0.84375, 0.96875, 0.96875, 0.96875, 0.9375, 0.875, 0.9375, 0.8125, 0.90625, 0.9375, 0.9375, 0.90625, 0.875, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 0.875, 0.875, 0.90625, 0.84375, 1.0, 0.96875, 0.9375, 0.90625, 0.9375, 0.90625, 0.96875, 0.90625, 0.96875, 0.90625, 0.8125, 0.875, 0.90625, 0.90625, 0.875, 0.875, 0.9375, 0.875, 0.96875, 0.90625, 0.90625, 0.875, 0.875, 0.9375, 1.0, 0.96875, 0.84375, 0.96875, 0.78125, 0.9375, 0.90625, 0.84375, 0.78125, 0.875, 0.875, 0.9375, 0.9375, 0.875, 0.90625, 0.875, 0.875, 0.90625, 0.96875, 0.96875, 0.78125, 0.96875, 0.96875, 0.84375, 0.875, 0.8125, 0.875, 0.875, 0.875, 0.875, 0.9375, 0.78125, 0.9375, 0.84375, 0.96875, 1.0, 0.90625, 0.90625, 1.0, 0.90625, 0.90625, 0.9375, 0.96875, 0.9375, 0.96875, 0.90625, 0.90625, 0.8125, 0.90625, 0.90625, 0.96875, 0.9375, 0.84375, 0.9375, 0.96875, 0.9375, 0.9375, 0.9375, 0.84375, 0.90625, 0.9375, 0.875, 1.0, 0.9375, 1.0, 0.96875, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.90625, 0.8125, 0.875, 0.9375, 0.84375, 0.90625, 0.96875, 0.875, 0.90625, 0.90625, 0.90625, 0.875, 1.0, 0.9375, 0.9375, 0.9375, 0.84375, 0.90625, 0.9375, 0.9375, 0.875, 0.875, 0.9375, 0.9375, 0.9375, 0.875, 0.90625, 0.9375, 0.9375, 0.9375, 0.96875, 0.96875, 0.9375, 0.9375, 0.875, 0.875, 0.96875, 0.9375, 0.90625, 0.875, 0.75]]
# solo_correctness = [0.75, 0.875, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.9375, 0.9375, 1.0, 1.0, 0.9375, 0.90625, 0.9375, 0.90625, 0.96875, 0.96875, 1.0, 0.9375, 0.9375, 0.875, 0.9375, 0.90625, 0.875, 0.96875, 0.9375, 0.90625, 0.875, 0.96875, 0.96875, 0.9375, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.90625, 0.90625, 0.90625, 0.90625, 1.0, 0.78125, 0.90625, 1.0, 0.9375, 0.9375, 1.0, 0.90625, 0.90625, 0.9375, 0.96875, 0.84375, 0.96875, 0.96875, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 1.0, 1.0, 0.9375, 0.96875, 0.84375, 1.0, 1.0, 0.96875, 0.84375, 1.0, 0.90625, 0.9375, 0.96875, 1.0, 0.90625, 0.96875, 1.0, 0.96875, 0.96875, 1.0, 0.96875, 0.96875, 0.90625, 0.96875, 0.9375, 1.0, 0.96875, 1.0, 0.9375, 0.90625, 0.96875, 0.96875, 0.9375, 1.0, 0.9375, 1.0, 0.90625, 0.9375, 0.96875, 0.9375, 0.96875, 0.84375, 0.96875, 0.90625, 0.875, 0.90625, 0.9375, 0.96875, 1.0, 0.9375, 0.9375, 0.96875, 0.9375, 0.84375, 0.96875, 0.875, 0.90625, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 1.0, 0.90625, 0.90625, 0.96875, 0.9375, 1.0, 0.90625, 0.90625, 0.9375, 0.90625, 0.96875, 0.8125, 0.9375, 0.84375, 0.90625, 0.84375, 0.90625, 1.0, 0.96875, 1.0, 0.9375, 0.96875, 0.875, 1.0, 0.96875, 1.0, 0.875, 0.875, 0.9375, 0.96875, 1.0, 0.84375, 0.9375, 0.875, 0.96875, 1.0, 0.90625, 0.9375, 0.875, 0.96875, 0.96875, 0.96875, 0.96875, 0.96875, 0.90625, 0.90625, 0.90625, 0.96875, 0.9375, 0.8125, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.90625, 1.0, 0.96875, 1.0, 0.9375, 1.0, 0.9375, 0.9375, 0.9375, 0.875, 0.90625, 0.875, 0.96875, 1.0, 0.90625, 0.84375, 0.9375, 0.9375, 1.0, 1.0, 0.875, 1.0, 1.0, 0.9375, 0.90625, 0.96875, 0.90625, 1.0, 1.0, 0.90625, 0.96875, 0.96875, 1.0, 0.96875, 0.96875, 0.875, 0.90625, 0.96875, 0.90625, 1.0, 0.96875, 0.96875, 1.0, 0.875, 0.9375, 1.0, 0.875, 0.90625, 0.96875, 0.96875, 0.96875, 0.875, 0.96875, 0.9375, 0.875, 0.96875, 0.90625, 0.90625, 0.90625, 0.9375, 1.0, 0.84375, 1.0, 0.9375, 1.0, 0.96875, 0.9375, 0.875, 0.875, 0.9375, 0.9375, 0.96875, 0.90625, 0.96875, 0.9375, 0.96875, 0.96875, 0.9375, 0.96875, 0.96875, 1.0, 0.96875, 0.9375, 0.90625, 0.96875, 0.84375, 1.0, 1.0, 0.96875, 0.90625, 0.96875, 0.90625, 1.0, 0.9375, 1.0, 0.9375, 0.90625, 0.96875, 0.96875, 1.0, 0.90625, 1.0, 0.90625, 1.0, 0.96875, 0.90625, 0.96875, 0.96875, 0.96875, 0.9375, 0.9375, 0.96875, 0.9375, 0.90625, 0.9375, 0.96875, 1.0, 0.96875, 0.875, 0.90625, 0.9375, 0.9375, 0.96875, 0.875, 0.9375, 0.96875, 0.84375, 0.90625, 0.96875, 0.875, 1.0, 1.0]
# acc_list = [solo_correctness] + [our_correctness[0]+our_correctness[1]] + [our_correctness[2]+our_correctness[3]]
acc_list = [[0.9296875, 0.90625, 0.9375, 0.9453125, 0.9375, 0.9453125, 0.90625, 0.921875, 0.9453125, 0.953125, 0.9296875, 0.9453125, 0.9296875, 0.9609375, 0.9453125, 0.953125, 0.96875, 0.9375, 0.953125, 0.9296875, 0.96875, 0.9375, 0.8984375, 0.9765625, 0.9453125, 0.9609375, 0.90625, 0.9296875, 0.953125, 0.9375, 0.96875, 0.9609375, 0.9453125, 0.9453125, 0.921875, 0.9453125, 0.953125, 0.9609375, 0.9609375, 0.8984375, 0.921875, 0.9296875, 0.953125, 0.90625, 0.953125, 0.9296875, 0.9140625, 0.9375, 0.8984375, 0.96875, 0.9453125, 0.9296875, 0.90625, 0.9375, 0.921875, 0.9453125, 0.953125, 0.953125, 0.953125, 0.9453125, 0.9296875, 0.9375, 0.953125, 0.9296875, 0.9296875, 0.9453125, 0.984375, 0.953125, 0.9296875, 0.9609375, 0.9375, 0.953125, 0.9140625, 0.96875, 0.953125, 0.9453125, 0.96875, 0.9453125, 1.0],
[0.90625, 0.9296875, 0.9375, 0.9375, 0.953125, 0.9453125, 0.90625, 0.890625, 0.921875, 0.9453125, 0.8984375, 0.9609375, 0.8828125, 0.9140625, 0.9296875, 0.90625, 0.9453125, 0.9765625, 0.921875, 0.8828125, 0.8984375, 0.9453125, 0.921875, 0.921875, 0.9375, 0.9609375, 0.9375, 0.9453125, 0.9609375, 0.9296875, 0.9140625, 0.8984375, 0.90625, 0.9140625, 0.953125, 0.890625, 0.9375, 0.890625, 0.8671875, 0.9140625, 0.9296875, 0.9609375, 0.9375, 0.90625, 0.953125, 0.9375, 0.9453125, 0.96875, 0.9375, 0.9140625, 0.953125, 0.9140625, 0.921875, 0.9140625, 0.921875, 0.921875, 0.9609375, 0.9296875, 0.8984375, 0.921875, 0.9453125, 0.9296875, 0.9375, 0.9296875, 0.9296875, 0.9453125, 0.953125, 0.9453125, 0.9375, 0.9140625, 0.90625, 0.9296875, 0.921875, 0.9140625, 0.921875, 0.9765625, 0.921875, 0.9140625, 1.0]+\
[0.921875, 0.9453125, 0.9453125, 0.90625, 0.921875, 0.953125, 0.9453125, 0.9140625, 0.9375, 0.9609375, 0.9375, 0.9375, 0.921875, 0.9140625, 0.875, 0.890625, 0.9765625, 0.921875, 0.9140625, 0.9140625, 0.9375, 0.9140625, 0.921875, 0.9453125, 0.8984375, 0.90625, 0.9609375, 0.8984375, 0.90625, 0.9296875, 0.921875, 0.984375, 0.90625, 0.9140625, 0.9140625, 0.9296875, 0.9453125, 0.9453125, 0.921875, 0.9296875, 0.9296875, 0.9453125, 0.8984375, 0.8984375, 0.9453125, 0.90625, 0.9140625, 0.9375, 0.9453125, 0.9375, 0.9453125, 0.953125, 0.9609375, 0.953125, 0.9140625, 0.9140625, 0.9296875, 0.953125, 0.9296875, 0.9140625, 0.9453125, 0.9375, 0.90625, 0.9453125, 0.90625, 0.9453125, 0.921875, 0.953125, 0.8828125, 0.921875, 0.9453125, 0.90625, 0.8828125, 0.9140625, 0.9296875, 0.921875, 0.9140625, 0.9296875, 0.9375],
[0.9140625, 0.8828125, 0.8359375, 0.8984375, 0.8984375, 0.8984375, 0.890625, 0.90625, 0.8828125, 0.9140625, 0.8671875, 0.875, 0.8828125, 0.8671875, 0.9140625, 0.8515625, 0.859375, 0.8984375, 0.875, 0.78125, 0.8515625, 0.84375, 0.8515625, 0.9453125, 0.8828125, 0.84375, 0.859375, 0.8984375, 0.8359375, 0.890625, 0.875, 0.8515625, 0.875, 0.8671875, 0.84375, 0.921875, 0.8515625, 0.921875, 0.8515625, 0.875, 0.828125, 0.875, 0.84375, 0.875, 0.875, 0.8984375, 0.8984375, 0.84375, 0.875, 0.8671875, 0.875, 0.8359375, 0.890625, 0.859375, 0.890625, 0.875, 0.84375, 0.875, 0.828125, 0.8828125, 0.8125, 0.8671875, 0.84375, 0.8828125, 0.84375, 0.8359375, 0.8984375, 0.828125, 0.8984375, 0.859375, 0.8515625, 0.8828125, 0.8671875, 0.796875, 0.875, 0.8046875, 0.84375, 0.921875, 0.9375]+\
[0.9140625, 0.9609375, 0.953125, 0.890625, 0.921875, 0.9296875, 0.9453125, 0.9375, 0.9375, 0.953125, 0.9140625, 0.9296875, 0.9296875, 0.9296875, 0.9296875, 0.921875, 0.921875, 0.921875, 0.890625, 0.9375, 0.890625, 0.890625, 0.8828125, 0.953125, 0.890625, 0.9453125, 0.8671875, 0.8671875, 0.8984375, 0.8984375, 0.9140625, 0.890625, 0.875, 0.8671875, 0.8984375, 0.8828125, 0.9453125, 0.9296875, 0.921875, 0.921875, 0.859375, 0.90625, 0.859375, 0.890625, 0.9453125, 0.9453125, 0.9140625, 0.9140625, 0.9140625, 0.9140625, 0.8515625, 0.921875, 0.90625, 0.921875, 0.890625, 0.921875, 0.890625, 0.9375, 0.90625, 0.90625, 0.9375, 0.8984375, 0.875, 0.8671875, 0.8828125, 0.9140625, 0.890625, 0.90625, 0.921875, 0.9453125, 0.9296875, 0.9375, 0.9375, 0.953125, 0.8515625, 0.9453125, 0.9296875, 0.9375, 1.0],
]

labels = ['Original','100% Neurons','50% Neurons']
colors_tmp = ['k'] + colors[:2]
measurements_to_cdf(acc_list,'/home/bo/Dropbox/Research/SIGCOMM23/images/mot3.eps',labels,colors=colors_tmp,linestyles=linestyles,xlabel='Accuracy (%)',ratio=None,lbsize=20)
# 4. via partitioning It is possible to maintain two-server performance while trading one-server performance for less computation overhead
# with loss, the performance is close 
methods_tmp = ['Accuracy','FLOPS']
y = [[86.287141,100], [93.467252,200],[93.031949,60.03*2]] 
yerr = [[1.330113,0], [0.563777,0],[0.57833,0]]
y,yerr = np.array(y),np.array(yerr)
groupedbar(y,yerr,'%', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/mot4.eps',methods=methods_tmp,
	envs=['Standalone','Replication','Partition (Ours)'],ncol=1,sep=.3,width=0.1,legloc='best',
	use_downarrow=True,labelsize=20)

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

envs = ['ResNet-20','ResNet-32','ResNet-44','ResNet-56','ResNet-110']
methods_tmp = ['Standalone','Ours']
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
y = np.concatenate((acc_base.reshape(5,1),acc_par_mean.reshape(5,1)),axis=1)*100
yerr = np.concatenate((np.zeros((5,1)),acc_par_std.reshape(5,1)),axis=1)*100
groupedbar(y,yerr,'Lossless Accuracy (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/acc_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,sep=.4,legloc='best',ylim=(90,95),rotation=30)

soft = [[0.002106629392971249, 0.003906847560424466,0.10148961661341853, 0.03568281929651537],
[0.006741214057507961, 0.003244131780051274,0.10220646964856232, 0.03583679685776256],
[0.011801118210862604, 0.00439929293904876,0.10365015974440901, 0.036999108780712796],
[0.007819488817891362, 0.003966352666503437,0.10384185303514379, 0.037132869618767474],
[0.011465654952076687, 0.002785178815481112,0.10562699680511187, 0.0382304333102761]
]
soft = np.array(soft)
y = soft[:,::2][:,::-1]
yerr = soft[:,1::2][:,::-1]
groupedbar(y,yerr,'$R^{(2)}$', 
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
groupedbar(y,yerr,'$R^{(2)}$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/hard_re_res.eps',methods=methods_tmp,
	envs=envs,ncol=1,sep=.4,legloc='best',use_barlabe_y=True,rotation=30)

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

# multi-node analysis
methods = ['$R_s^{(2)}$','$R_s^{(3)}$','$R_s^{(4)}$','$R_h^{(2)}$','$R_h^{(3)}$','$R_h^{(4)}$']

re_vs_nodes = [[0.10407947284345051, 0.03738313137189115, 0.1317492012779553, 0.03571484752205825, 0.13577276357827478, 0.03598051353576017,
0.16654752396166136, 0.0137959207372693, 0.20654552715654956, 0.01022833202885187, 0.21356230031948886, 0.008679869968650581],
[0.007140575079872202, 0.004132095691370403, 0.0020946485623003054, 0.003389731212981527, 0.006569488817891345, 0.002178623274278881,
0.006259984025559107, 0.0013025614474167567, 0.0027156549520766736, 0.0006374157228239655,0.005820686900958461, 0.0008257821278654765],
]
re_vs_nodes = np.array(re_vs_nodes)
y = re_vs_nodes[:,0::2]
yerr = re_vs_nodes[:,1::2]
groupedbar(y,yerr,'$R$', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/re_vs_nodes.eps',methods=methods,
	envs=['Standalone','Ours'],bbox_to_anchor=(0.5, 1.25),use_barlabel_x=True,xlabel='#Servers',legloc='upper right',ncol=2)

flops_nodes = [[100,60.03,64.26,67.08]]
flops_nodes = np.array(flops_nodes).transpose((1,0))
groupedbar(flops_nodes,None,'FLOPS (%)', 
	'/home/bo/Dropbox/Research/SIGCOMM23/images/flops_vs_nodes.eps',methods=['FLOPS'],envs=[1,2,3,4],
	ncol=0,sep=1,bbox_to_anchor=(0.5, 1.2),width=0.4,xlabel='#Servers')

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
colors_tmp = ['k'] + colors[:2] + ['grey'] + colors[2:4]
linestyles_tmp = ['solid','dashed','dotted','solid','dashdot',(0, (3, 5, 1, 5)),]
methods_tmp = ['Replication','Partition (Ours)','Standalone','Replication*','Partition (Ours)*','Standalone*']
line_plot(x,y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea.eps',
		'Deadline (s)','Delivered Acc. (%)',yticks=[0,25,50,75,100],lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
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
x = [[0.05*i for i in range(1,11)] for _ in range(6)]
y = np.concatenate((np.array(y_ea_loss1)[:,:10]*100,np.array(y_ea_loss2)),axis=0)
yerr = np.concatenate((np.array(yerr_ea_loss1)[:,:10]*100,np.array(yerr_ea_loss2)),axis=0)
line_plot(x, y,methods_tmp,colors_tmp,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea2.eps',
		'Loss Rate (%)','Delivered Acc. (%)',lbsize=16,linewidth=4,markersize=0,linestyles=linestyles_tmp,
		yerr=yerr,ylim=(20,100))	