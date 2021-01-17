import numpy as np
import matplotlib.pyplot as plt
import random
import math


N = 50
data = []
K = 6
centroids = []
colors = ['b','g','r','c','m','y','k']
dists = []
clusters = [[] for i in range(K)]
x = np.random.rand(N)
y = np.random.rand(N)


# plot the data
def plot_data(clusters):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	for i in range(K):
		x,y = zip(*clusters[i])
		ax1.scatter(x, y, s=20, c=colors[i])

	plt.show()

def kmeans(centroids, dists):
	# init random centroids
	new_centroids = []
	for i in range(K):
		centroids.append((0,0))
		new_centroids.append(data[random.randint(0,N-1)])

	# calc distance from points to centroids
	while not (converged(centroids,new_centroids)):
		centroids = new_centroids[:]
		new_centroids = []
		for i in range(K):
			d = distances(centroids[i], data)
			dists.append(d)

		# reset clusters
		clusters = [[] for i in range(K)]
		# assign each point to the closest cluster
		asign_to_closest_cluster(dists, data, clusters)
		dists = []

		for i in range(K):
			mean = get_mean(clusters[i])
			new_centroids.append((mean))
		return clusters

			

def asign_to_closest_cluster(dists, data, clusters):
	for n in range(N):
		temp = []
		for k in range(K):
			temp.append(dists[k][n])
		clust = temp.index(min(temp))
		clusters[clust].append(data[n])

def get_mean(c):
	sumX = 0
	sumY = 0
	for i in range(len(c)):
		sumX += c[i][0]
		sumY += c[i][1]
	return (sumX/len(c), sumY/len(c))

# get distance for one centroid
def distances(c, data):
	d = []
	for x in range(N):
		dist = euclid_dist(c,data[x])
		d.append(dist)
	return d

# euclidian distance
def euclid_dist(c,p):
	return math.sqrt( math.pow( c[0] - p[0], 2) + math.pow( c[1] - p[1], 2) )

def init_data():
	for i in range(N):
		data.append((x[i],y[i]))

def converged(c1,c2):
	if len(c2) < 1:
		return False
	for k in range(K):
		if c1[k][0] != c2[k][0] or c1[k][1] != c2[k][1]:
			return False
	return True


init_data()

# displaying raw data
plt.scatter(x,y, color='k', s=25, marker="o")

clusters = kmeans(centroids, dists)

# displaying clustered data
plot_data(clusters)

