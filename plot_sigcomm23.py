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

x = [[0.1*i for i in range(1,10)] for _ in range(3)]
y = [[0.10286601632833481, 0.5390719175338745, 0.8344619870185852, 0.911393940448761, 0.9336099624633789, 0.9394559860229492, 0.9403420686721802, 0.9405380487442017, 0.9405380487442017], [0.10452401638031006, 0.4070579409599304, 0.6651359796524048, 0.7920659780502319, 0.8578199148178101, 0.8995599746704102, 0.9189940690994263, 0.9319100379943848, 0.9357859492301941], [0.11322401463985443, 0.6181179881095886, 0.8571680188179016, 0.9181960225105286, 0.9356400370597839, 0.940060019493103, 0.9410279989242554, 0.9410279989242554, 0.9410279989242554]]
yerr = [[0.002733248518779874, 0.02461308240890503, 0.01103442907333374, 0.009810970164835453, 0.00689014233648777, 0.002314268611371517, 0.001483639352954924, 0.00117053824942559, 0.00117053824942559], [0.0052833459340035915, 0.04242616519331932, 0.03385111317038536, 0.027212493121623993, 0.021590324118733406, 0.0148468017578125, 0.008772843517363071, 0.006255739834159613, 0.004351438954472542], [0.007508941926062107, 0.022326674312353134, 0.01010508555918932, 0.008913583122193813, 0.005452818237245083, 0.001438929233700037, 0.00021600723266601562, 0.00021600723266601562, 0.00021600723266601562]]
line_plot(x, y,['Ours','No replica','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_ea.eps',
		'Deadline','Effective Top1 Accuracy (%)',
		yerr=yerr)	
y = [[0.9965600000000002, 0.47752000000000006, 0.12632, 0.03463999999999999, 0.008400000000000008, 0.0013600000000000057, 0.0003199999999999981, 8.00000000000023e-05, 8.00000000000023e-05], [0.9946400000000001, 0.6348800000000001, 0.32736, 0.17695999999999998, 0.09879999999999999, 0.04919999999999998, 0.026240000000000006, 0.01080000000000001, 0.00616000000000001], [0.98424, 0.38408, 0.09968000000000002, 0.027359999999999995, 0.006400000000000017, 0.0012000000000000123, 8.00000000000023e-05, 8.00000000000023e-05, 8.00000000000023e-05]]
yerr = [[0.003299454500368193, 0.029329466411784594, 0.01254470406187406, 0.011240213521103576, 0.007799487162628069, 0.0017906423428479586, 0.0007332121111929237, 0.0002400000000000069, 0.0002400000000000069], [0.0062179096165833805, 0.05062451580015357, 0.04064094487090575, 0.03230019194989405, 0.025613121637160903, 0.01762997447530768, 0.010361775909563001, 0.0074382793709298215, 0.005034918072819033], [0.008994576143432215, 0.026260114241944948, 0.01192080534192216, 0.010690107576633634, 0.006499230723708775, 0.0016876018487783306, 0.0002400000000000069, 0.0002400000000000069, 0.0002400000000000069]]
line_plot(x, y,['Ours','No replica','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_fr.eps',
		'Deadline','Failure Rate',
		yerr=yerr)

# breakdown for different traces

# with loss
x = [[0.05*i for i in range(1,6)] for _ in range(3)]
y = [[0.9371520280838013, 0.9303600192070007, 0.9204179644584656, 0.904076099395752, 0.8894540071487427], [0.8965460658073425, 0.8547260165214539, 0.8151799440383911, 0.7671979665756226, 0.7338219881057739], [0.9391420483589172, 0.9315799474716187, 0.9210079908370972, 0.9062359929084778, 0.890514075756073]]
yerr = [[0.001424455433152616, 0.0031888154335319996, 0.004146410617977381, 0.003149099415168166, 0.0036634753923863173], [0.003918539732694626, 0.0049526747316122055, 0.009116784669458866, 0.009399198926985264, 0.009671228006482124], [0.001109693548642099, 0.002523125847801566, 0.003530264599248767, 0.003066291334107518, 0.003128807758912444]]
line_plot(x, y,['Ours','No replica','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_ea.eps',
		'Loss Rate','Effective Top1 Accuracy (%)',
		yerr=yerr)	
y = [[0.00232, 0.011199999999999998, 0.02368, 0.041759999999999985, 0.06024], [0.05296000000000001, 0.10255999999999998, 0.1492, 0.20688, 0.24672], [0.00232, 0.011199999999999998, 0.02368, 0.041759999999999985, 0.06024]]
yerr = [[0.001312097557348531, 0.0030357865537616576, 0.004082352263095388, 0.0031759093186046767, 0.0037016752964029527], [0.004305159695063582, 0.00551710068786135, 0.01101707765244487, 0.010376589034938224, 0.011269498657881807], [0.001312097557348531, 0.0030357865537616576, 0.004082352263095388, 0.0031759093186046767, 0.0037016752964029527]]
line_plot(x, y,['Ours','No replica','Optimal'],colors,
		'/home/bo/Dropbox/Research/SIGCOMM23/images/FCC_loss_fr.eps',
		'Loss Rate','Failure Rate',
		yerr=yerr)	