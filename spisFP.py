import spotipy
#import spotipy.util as util
#from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd
from pandas import DataFrame as df
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.distance import cdist, euclidean

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import random
import math 




### AUTHORIZATION: gives us access the Spotify web api
SPOTIFY_CLIENT_ID = "5d13c8e81ed443a1bbb0b82bf96b726c"
SPOTIFY_CLIENT_SECRET = "f1f41256d087422f934eff37df155217" 

token = spotipy.oauth2.SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
cache_token = token.get_access_token()
sp = spotipy.Spotify(cache_token)




### PLAYLISTS
niemaM = sp.playlist('6mev4XSDNtNE1ejOhtWJww?si=6rNyBhqsSKW6Mf2HoFTZZA')
lukeT = sp.playlist('0RTt4lJSZleqpatBo7vsbV?si=zSKVRhZCTGC_ZRl_Ci73hw')
matiasL = sp.playlist('0PELZmYcSulDClDVvnlUtb?si=u6sGwVeoRnqtWxh12K5NRw')
amitB = sp.playlist('1Kg8SdJX7qwUSE9fXKh6gF?si=3T2F-BTHQzeXak_IUUAs1A')
alexisC = sp.playlist('5jn7fDwNsM1MHw2hydTitg?si=T5msWOZkTcaHfoBzX8HtDg')
thomasG = sp.playlist('5M89yVJCIHsSjDLONsQkVX?si=zAWwxbtXTrKQrfWt1Yrq3Q')
davidC = sp.playlist('3l73bhpop3bOQkDA5KEC68?si=QryFCCsbQfCkEnlGWmRBdA')
armaanB = sp.playlist('44dqQ9QBGR8Vxx5qRp9GXL?si=m8-wcrbeTIyH0pdwjlxuXw')
ericX = sp.playlist('6VZUWvFaKEfsHEChkGpUCa?si=RoIWUS-0Qz-BBej-2HFm8w')
jenelleT = sp.playlist('5vtjTX3hkuUncqCJXiBFTs?si=HX9lVCv-RTCReLV7TIjDQw')
yukatiG = sp.playlist('4lsoy3nVvLdHT0RNOcmw9K?si=Y4vSiiK0TSukZPFGF_r11A')
alyssaS = sp.playlist('5VfyLSk3VUHbr7TOq3ryGS?si=aQe9qWCSTeC3-raT9zM3wg')
dhanviD = sp.playlist('0ReJZPwPH06ZBbB74Qbsvh?si=-Ia7-z5cRQmePf0YeodOjw')
faithL = sp.playlist('06OuGIVMOCjuUvfNsGuyAc?si=N8gTEPVxQDmnSgoCDVjd8Q')
lilyS = sp.playlist('0IDNTqC6Tiw3gbM2wUTvvx?si=yxss_VzOROSKxKYPqNIkOg')
                    #1200 songs --> 7ryCQmUj2kN0If31yJTETG?si=viRCXSH-SRqUCQ0ZQgpiqQ#
aliseB = sp.playlist('2GHp8Z5VHzlc0VUQSnZDO2?si=7CxsN0VXTRGU0DqYdYcpqg')
laurenC = sp.playlist('3RipGahx7PdhgA3CnxpFnE?si=u20QfqEkQKq5BMOySlWQ5Q')
colinL = sp.playlist('0plmXrWTZ2RmdSZaHP6SYY?si=fWCjpXO_Tcu82Gx6wWQAFw')
alexanderJ = sp.playlist('4dmrr99evD3aybKYKQBrs9?si=qVG344wbTKKHq0pVSfUOyA')
seanP = sp.playlist('4IkgHwIIyeTEYY7LSjC35s?si=EJkScZnMQlagVQW0QE4Wmw')
lailahG = sp.playlist('68o87Uqc2f2nWUCv1QNU1h?si=cUkIr10JRFWMPU_s6Edqtg')
jeffreyL = sp.playlist('0C44p8fy40fVHMj3Ib89ma?si=3x7zzo1lTQitFA5qfBLSqQ')
lindseyM = sp.playlist('5dRomcYUhlTY8xVlaI8LGQ?si=VCyd3HNWSq2q4UiKmlu_EA')
akshatL = sp.playlist('6UyGET0RPw1s3M8cFEwPjs?si=V5aD3p4GRcyD-KIWYnvymQ')
theoH = sp.playlist('4Ak2nxh3uKBGppMPPn4Gjm?si=r6K6y4qBRpapKmHLr-8wGQ')
nihalN = sp.playlist('7g4vagGaFI8EBclNv4rqX3?si=OWMnKdVYQSGhE79tUqMY7w')

playlist_list = [niemaM, lukeT, matiasL, amitB, alexisC, thomasG, davidC, armaanB, ericX, jenelleT, yukatiG, alyssaS, dhanviD, faithL, lilyS, aliseB, laurenC, colinL, alexanderJ, seanP, lailahG, jeffreyL, lindseyM, akshatL, theoH, nihalN]
playlist_list_names = ["niemaM", "lukeT", "matiasL", "amitB", "alexisC", "thomasG", "davidC", "armaanB", "ericX", "jenelleT", "yukatiG", "alyssaS", "dhanviD", "faithL", "lilyS", "aliseB", "laurenC", "colinL", "alexanderJ", "seanP", "lailahG", "jeffreyL", "lindseyM", "akshatL", "theoH", "nihalN"]





'''________________________________________________________________________________________________________________________________________'''

#creates clusters with all the data and algorithms that come with it
class Spotify_Clustering:

    #sets up global variables with class is instantiated
    def __init__(self):
        self.center_points = [] #used to hold geometric medians
        self.km = []




    # calculates 3 Principal Component values from the data from each song in the playlist
    # returns a list of PCA values
    def pca_calculations(self, playlist):

        list_dicts = []
        cur_num = 1
        cur_track = playlist["tracks"]
        #print("\n\n\n")

        #puts the data from each song into a list of dictionaries
        while cur_track:
            for i in range(len(cur_track["items"])):
                #print(cur_num, cur_track["items"][i]["track"]["name"])
                aud_feats = sp.audio_features(playlist["tracks"]["items"][i]["track"]["id"])
                aud_feats[0].pop('type')
                aud_feats[0].pop('id')
                aud_feats[0].pop('uri')
                aud_feats[0].pop('track_href')
                aud_feats[0].pop('analysis_url')
                list_dicts.append(aud_feats[0])
                cur_num += 1
            cur_track = sp.next(cur_track)

        #table of data for list of dictionaries
        song_stats_df = pd.DataFrame.from_dict(list_dicts)

        #calculates PCA values
        scaler = StandardScaler()
        scaler.fit(song_stats_df)
        scaled_data = scaler.transform(song_stats_df)
        pca = PCA(n_components=3)
        pca.fit(scaled_data)
        return pca.transform(scaled_data)

        #prints pca values
        #for i in range(len(x_pca)):
            #print("{}\t{}\t{}".format(x_pca[i][0], x_pca[i][1], x_pca[i][2]))
            #print("\n\n\n")



    # finds the geometric median using a list of 3D PCA points and returns the coordinates
    def find_geomedian(self, pca):
        self.x_vals = []
        self.y_vals = []
        self.z_vals = []
 
        for i in range(len(pca)):
            self.x_vals.append(pca[i][0])
            self.y_vals.append(pca[i][1])
            self.z_vals.append(pca[i][2])
 
        list_points = []
        for i in range(len(self.x_vals)):
            coor = []
            coor.append(self.x_vals[i])
            coor.append(self.y_vals[i])
            coor.append(self.z_vals[i])
            list_points.append(coor)

        list_points = np.array(list_points)
        return self.geometric_median(list_points)



    # helper method of find_geomedian() that calculates the geometric median of a cloud of points and returns the coordinates
    def geometric_median(self, X, eps=1e-5):
        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)

            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros/r
                y1 = max(0, 1-rinv)*T + min(1, rinv)*y

            if euclidean(y, y1) < eps:
                return y1

            y = y1



    # graphs geometric median point (black) on the figure
    def graph_geomedian(self, x, y, z):
        self.ax.plot(x, y, z, c='black', marker='o')
        #name = str(name)
        #self.ax.text(x, y, z, '%s' % name, fontdict=font, size=10, zorder=1, color='r')



    # using the elbow method, finds the best, optimal number of clusters
    def find_num_clusters(self):
        sse = []
        k_rng = range(1, len(playlist_list))

        for i in range(len(k_rng)):
            self.km.append(KMeans(n_clusters=k_rng[i]))
            self.km[i].fit(self.df_geomeds[['x_coor','y_coor', 'z_coor']])
            sse.append(self.km[i].inertia_)
        #plt.xlabel('K')
        #plt.ylabel('Sum of squared error')
        #plt.plot(k_rng,sse)
        areas = []
        for i in range(len(k_rng)):
            T = (1/2)* abs((k_rng[0] - k_rng[i]) * (sse[len(sse)-1]-sse[0]) - (k_rng[0] - k_rng[len(k_rng)-1])* (sse[i] - sse[0]))
            areas.append(T)
        return areas.index(max(areas))+1



    # using the center_points list, groups all geometric median points into a number of clusters and graphs them
    def graph_clusters(self, num_clusters=3):
        fig = plt.figure("SPIS-y Music")
        ax = fig.add_subplot(111, projection='3d')
 
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_zlabel('Principle Component 3')

        font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 10,
        }
 
        x_geomeds = []
        y_geomeds = []
        z_geomeds = []
 
        i = 0
        for coor in self.center_points:
            x_geomeds.append(coor[0])
            y_geomeds.append(coor[1])
            z_geomeds.append(coor[2])
            ax.text(coor[0], coor[1], coor[2], playlist_list_names[i], fontdict=font, size=5, zorder=1, color='black')
            i += 1

        y_predicted = self.km[num_clusters-1].fit_predict(self.df_geomeds)
        self.df_geomeds['cluster']= y_predicted


        for i in range(num_clusters):
            rand_color = '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cluster = self.df_geomeds[self.df_geomeds.cluster==i]
            ax.scatter3D(cluster['x_coor'], cluster['y_coor'], cluster['z_coor'], color=rand_color, alpha=1.0)
            #ax.scatter(self.km[num_clusters-1].cluster_centers_[:,0],self.km[num_clusters-1].cluster_centers_[:,1], self.km[num_clusters-1].cluster_centers_[:,2], color='purple',marker='x',label='centroid')
   
    

    # finds distance between two 3D points
    def distance(self, x1, y1, z1, x2, y2, z2):  
        return math.sqrt(math.pow(abs(x2 - x1), 2) + math.pow(abs(y2 - y1), 2) + math.pow(abs(z2 - z1), 2)* 1.0) 




'''________________________________________________________________________________________________________________________________________'''





#this class sets up to find the best set of clusters by creating a variable number of clusters and graphing the cluster
class Spotify_Runner:
 
    #sets up the global variables that will hold the sets of cluster arrangements and their inertias
    def __init__(self, num_runs):
        self.spot_list = []
        self.inertias = []
 
        self.min_intertia_spot = -1
        self.best_num_cluster = -1
 
        self.num_runs = num_runs
        self.center_points = []
 
 
    #creates a num_runs number of cluster arrangements and sorts them to find the most efficient arrangement and graphing that arrangement
    def spot_creation(self):
        for i in range(self.num_runs):
            self.spot_list.append(Spotify_Clustering())
 
            if i == 0:
                for playlist in playlist_list:
                    pca = self.spot_list[i].pca_calculations(playlist)
                    x, y, z = self.spot_list[i].find_geomedian(pca)
                    self.center_points.append([x, y, z])
 
            self.spot_list[i].center_points = self.center_points
            self.spot_list[i].df_geomeds = df(self.center_points, columns=['x_coor','y_coor','z_coor'])
 
            num_cluster = self.spot_list[i].find_num_clusters()
            self.inertias.append(self.spot_list[i].km[num_cluster-1].inertia_)
 
            if min(self.inertias) == self.spot_list[i].km[num_cluster-1].inertia_:
                self.min_intertia_spot = i
                self.best_num_cluster = num_cluster
 
        self.spot_list[self.min_intertia_spot].graph_clusters(self.best_num_cluster)



'''________________________________________________________________________________________________________________________________________'''



### MAIN
runner = Spotify_Runner(25)
runner.spot_creation()


plt.show()
print("\n\n\n")








# #fills center_points with geometric medians from each playlist in playlist_list
# for playlist in playlist_list:
#     pca = spot.pca_calculations(playlist)
#     x, y, z = spot.find_geomedian(pca)
#     spot.center_points.append([x * 100, y * 100, z * 100]) #data scaling
# #print(spot.center_points)
# num_clusters = spot.find_num_clusters()
# spot.graph_clusters(num_clusters)

#
# num_clusters = spot.find_num_clusters()
# for i in (5):
#     spot.graph_clusters(num_clusters)


#GRAPHS ALL SONGS (RED POINTS)
#ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')

# #AVERAGE POINT (GREEN)
# avg_x = sum(x_vals) / len(x_vals)
# avg_y = sum(y_vals) / len(y_vals)
# avg_z = sum(z_vals) / len(z_vals)
# ax.scatter(avg_x, avg_y, avg_z, c='g', marker='o')

#MEDIAN POINT (BLUE)
# sorted_x = sorted(x_vals)
# sorted_y = sorted(y_vals)
# sorted_z = sorted(z_vals)
# x_med = sorted_x[len(x_vals)//2]
# y_med = sorted_y[len(y_vals)//2]
# z_med = sorted_z[len(z_vals)//2]
# ax.scatter(x_med, y_med, z_med, c='b', marker='o')















    
















