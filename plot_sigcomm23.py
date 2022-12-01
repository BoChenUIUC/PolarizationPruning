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
colors = ['#DB1F48','#FF9636','#1C4670','#9D5FFB','#21B6A8','#D65780']
markers = ['o','P','s','>','D','^']


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

x = [[0.1*i for i in range(1,6)] for _ in range(3)]
y = [[0.10300002, 0.54106605, 0.83518803, 0.91195405, 0.9342    ],
 [0.104596,   0.40781993, 0.66554797, 0.7922,     0.858016  ],
 [0.113564,   0.6185099,  0.857158,   0.91832,    0.93571204]]
yerr = [[0.00276906, 0.02527536, 0.01103008, 0.00990747, 0.00698827],
 [0.0052258,  0.04248973, 0.0337695,  0.0273513,  0.02132575],
 [0.00765002, 0.0223185,  0.0100436,  0.00890443, 0.00524915]]
line_plot(x, y,['Ours','No replica','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea.eps',
		'Deadline','Effective Top1 Accuracy (%)',
		yerr=yerr)	
y = [[0.9964,  0.47536, 0.12608, 0.03464, 0.0084 ],
 [0.99456, 0.634,   0.32688, 0.1768,  0.09856],
 [0.98384, 0.3836,  0.09968, 0.0272,  0.00632]] 
yerr = [[0.00334186, 0.02968162, 0.01238941, 0.01124021, 0.00779949],
 [0.00615324, 0.05072309, 0.04054698, 0.03244293, 0.02529519],
 [0.00916899, 0.02624012, 0.0118454,  0.01067933, 0.00627324]]
line_plot(x, y,['Ours','No replica','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_fr.eps',
		'Deadline','Failure Rate',
		yerr=yerr)	
