from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import pandas as pd
from etaprogress.progress import ProgressBar

np.set_printoptions(suppress=True)

i = 0
nh = 1080
nw = 1440

parser = argparse.ArgumentParser(description='Annotation kmeans')
parser.add_argument("-a","--annotations", help="file path to annotations", type=str, default = r'/mnt/Z/ShippingLab/mmdet/2019_04_12.csv' )
parser.add_argument("--sizes", help="calculate sizes", action='store_true')
args = parser.parse_args()

df_csv = pd.read_csv(args.annotations)
print(df_csv.columns.values)

maxRatio = 0

bar = ProgressBar(len(df_csv['Unnamed: 1'].notnull()), max_width=100)

sizes = np.zeros(len(df_csv['Unnamed: 1'].notnull()))
ratios = np.zeros(len(df_csv['Unnamed: 1'].notnull()))

old_img = ''
cnt = -1
for i, row in df_csv[df_csv['Unnamed: 1'].notnull()].iterrows():
	cnt += 1
	if args.sizes:
		if i % 1000 == 0:
			bar.numerator = i
			print(bar, end='\r')
		if row[0] != old_img:
			pp = (r'/mnt/Z/ShippingLab/mmdet/' + row[0][18:])
			im = mpimg.imread(pp)
			iw = im.shape[0]
			ih = im.shape[1]
			old_img = row[0]

		w = np.abs(row[1] - row[3]) * nw/iw
		h = np.abs(row[2] - row[4]) * nh/ih
		size = np.sqrt(float(w * iw * h * ih))
		sizes.append(size)
		sizes[cnt] = size
	w = np.abs(row[1] - row[3])
	h = np.abs(row[2] - row[4])
	if w != 0:
		ratio = float(h/w)
		# ratios.append(ratio)
		ratios[cnt] = ratio

if args.sizes:
	x = sizes[np.where(sizes != 0)]
	k_means = KMeans(n_clusters=5).fit(x.reshape(-1, 1))
	print('\n sizes: ')
	size = np.sort(k_means.cluster_centers_.squeeze())
	print(size)
	m = 0
	for i, s in enumerate(size):
		m = s/(2**(i+3))
		print("scale: ", m)

x = ratios[np.where(ratios != 0)]
k_means = KMeans(n_clusters=5).fit(x.reshape(-1, 1))
print('\n ratios: ')
print(np.sort(k_means.cluster_centers_.squeeze()))
