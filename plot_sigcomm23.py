#!/usr/bin/python

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
labelsize_b = 13
linewidth = 2
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams["font.family"] = "Times New Roman"
colors = [
'#3288bd',
'#abdda4',
'#66c2a5',
'#9D5FFB',
'#fee08b',
'#fdae61',
'#f46d43',
'#d53e4f',
]
markers = ['o','P','s','D','v','^','<','>']


def line_plot(XX,YY,label,color,path,xlabel,ylabel,lbsize=labelsize_b,lfsize=labelsize_b,legloc='best',
				xticks=None,yticks=None,ncol=None, yerr=None,
				use_arrow=False,arrow_coord=(0.4,30)):
	fig, ax = plt.subplots()
	ax.grid(zorder=0)
	for i in range(len(XX)):
		xx,yy = XX[i],YY[i]
		if yerr is None:
			plt.plot(xx, yy, color = color[i], marker = markers[i], 
				# linestyle = linestyles[i], 
				label = label[i], 
				linewidth=2, markersize=8)
		else:
			plt.errorbar(xx, yy, yerr=yerr[i], color = color[i], 
				marker = markers[i], label = label[i], 
				linewidth=2, markersize=8)
	plt.xlabel(xlabel, fontsize = lbsize)
	plt.ylabel(ylabel, fontsize = lbsize)
	if xticks is not None:
		plt.xticks(xticks,fontsize=lfsize)
	if yticks is not None:
		plt.yticks(yticks,fontsize=lfsize)
	if use_arrow:
		ax.text(
		    arrow_coord[0], arrow_coord[1], "Better", ha="center", va="center", rotation=-45, size=lbsize-8,
		    bbox=dict(boxstyle="larrow,pad=0.3", fc="white", ec="black", lw=2))
	plt.tight_layout()
	if ncol!=0:
		if ncol is None:
			plt.legend(loc=legloc,fontsize = lfsize)
		else:
			plt.legend(loc=legloc,fontsize = lfsize,ncol=ncol)
	# plt.xlim((0.8,3.2))
	# plt.ylim((-40,90))
	plt.tight_layout()
	fig.savefig(path,bbox_inches='tight')
	plt.close()

x = []
y = []
# original
x += [[92.52,87.53,80.16,71.11,61.96,51.42,40.52,30.26,20.15,0.00]]
y += [[0.1040,0.1216,0.1160,0.1655,0.1820,0.2294,0.4102,0.8530,0.9135,0.9411]]
line_plot(x, y,['Ours','No rep','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/test.eps',
		'Deadline','Effective Top1 Accuracy (%)')	
