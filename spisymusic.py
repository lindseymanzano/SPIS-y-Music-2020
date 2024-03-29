import csv
import matplotlib.pyplot as plt
import numpy as np
import spotipy
import pandas as pd
from pandas import DataFrame as df
from scipy.spatial.distance import cdist, euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import statistics
import warnings
import random

# accesses the Spotify web api
_CLIENT_ID = "fd0aef9ef8e34f4b9529ac5ac058af22"
_CLIENT_SECRET = "2e9d658ed1754c909f1f436764650cd4"

warnings.filterwarnings("ignore", category=DeprecationWarning)

token = spotipy.oauth2.SpotifyClientCredentials(client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET)
cache_token = token.get_access_token()
sp = spotipy.Spotify(cache_token)

playlist_list = []
playlist_list_names = []
playlist_list_roles = []

# accesses csv file with form responses and adds these values to relevant lists
cur_row = 0
with open('playlist.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if cur_row == 0:
            cur_row += 1
            continue

        playlist_list.append(sp.playlist(row[3][row[3].rindex("/") + 1:]))
        playlist_list_names.append(row[1][:row[1].index(" ") + 2])
        playlist_list_roles.append(row[2])

        cur_row += 1


# creates clusters with all the data and algorithms that come with it
class Spotify_Clustering:

    # sets up global variables with class is instantiated
    def __init__(self):
        self.center_points = []
        self.km = []
        self.cluster_name = {}

    # calculates the pca values of each song in the playlist and returns the list of pca values
    def pca_calculation(self, playlist):
        cur_num = 1
        cur_track = playlist["tracks"]
        list_dicts = []
        unwanted_vars = ["type", "id", "uri", "analysis_url", "track_href"]

        print("\n\n\n")
        while cur_track:
            for i in range(len(cur_track["items"])):
                audio_features = sp.audio_features(playlist["tracks"]["items"][i]["track"]["id"])
                print(cur_num, cur_track["items"][i]["track"]["name"])
                cur_num += 1

                for j in range(len(unwanted_vars)):
                    audio_features[0].pop(unwanted_vars[j])

                list_dicts.append(audio_features[0])

            cur_track = sp.next(cur_track)

        df = pd.DataFrame.from_dict(list_dicts)

        scaler = StandardScaler()
        scaler.fit(df)
        scaled_data = scaler.transform(df)

        pca = PCA(n_components=3)
        pca.fit(scaled_data)

        return pca.transform(scaled_data)

    # using the 3 dimensional points in the list of pca values, find the geometric median of the set of points and return it
    def find_geo_med(self, pca):
        x_vals = []
        y_vals = []
        z_vals = []

        for i in range(len(pca)):
            x_vals.append(pca[i][0])
            y_vals.append(pca[i][1])
            z_vals.append(pca[i][2])

        list_points = []
        for i in range(len(x_vals)):
            coor = []
            coor.append(x_vals[i])
            coor.append(y_vals[i])
            coor.append(z_vals[i])
            list_points.append(coor)

        list_points = np.array(list_points)
        return self.geometric_median(list_points)

    # helper method to find_geo_med() that takes in a list of 3-D points and returns the geometric median of the set of points
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
                rinv = 0 if r == 0 else num_zeros / r
                y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

            if euclidean(y, y1) < eps:
                return y1

            y = y1

    # takes in the number of clusters and graphs the geometrical medians on the 3-D figure, coloring them based on the cluster they belong to
    def graph_clusters(self, num_cluster):
        fig = plt.figure("SPIS-y Music")

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Music Preference by Cluster")
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_zlabel('Principle Component 3')

        x_geomeds = []
        y_geomeds = []
        z_geomeds = []

        i = 0
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 10,
                }

        for coor in self.center_points:
            x_geomeds.append(coor[0])
            y_geomeds.append(coor[1])
            z_geomeds.append(coor[2])
            ax.text(coor[0], coor[1], coor[2], playlist_list_names[i], fontdict=font, size=5, zorder=1, color='black')
            i += 1

        y_predicted = self.km[num_cluster - 1].fit_predict(self.df_geomeds)
        self.df_geomeds['cluster'] = y_predicted

        for i in range(len(playlist_list_names)):
            self.cluster_name.update({playlist_list_names[i]: y_predicted[i]})

        for i in range(num_cluster):
            rand_color = '#%02x%02x%02x' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cluster = self.df_geomeds[self.df_geomeds.cluster == i]

            ax.scatter3D(cluster['x_coor'], cluster['y_coor'], cluster['z_coor'], color=rand_color, alpha=1.0)

    # graphs the geometric medians on a separate figure and color the points based on their role in SPIS.
    def graph_roles(self):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Music Preference by SPIS Role")
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_zlabel('Principle Component 3')

        x_geomeds = []
        y_geomeds = []
        z_geomeds = []

        i = 0
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 10,
                }

        for coor in self.center_points:
            x_geomeds.append(coor[0])
            y_geomeds.append(coor[1])
            z_geomeds.append(coor[2])
            ax.text(coor[0], coor[1], coor[2], playlist_list_names[i], fontdict=font, size=5, zorder=1, color='black')
            i += 1

        for i in range(len(playlist_list_roles)):
            if playlist_list_roles[i] == "Professor":
                ax.scatter3D(x_geomeds[i], y_geomeds[i], z_geomeds[i], color='yellow', alpha=1.0)
            elif playlist_list_roles[i] == "Mentor":
                ax.scatter3D(x_geomeds[i], y_geomeds[i], z_geomeds[i], color='blue', alpha=1.0)
            else:
                ax.scatter3D(x_geomeds[i], y_geomeds[i], z_geomeds[i], color='red', alpha=1.0)

    # graph thes geometric medians and the name of the point on a figure as black dots
    def graph_geo_med(self):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Music Preference of All SPIS Members")
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_zlabel('Principle Component 3')

        i = 0
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 10,
                }
        for coor in self.center_points:
            ax.text(coor[0], coor[1], coor[2], playlist_list_names[i], fontdict=font, size=5, zorder=1, color='black')
            ax.scatter3D(coor[0], coor[1], coor[2], color='black', alpha=1.0)
            i += 1

    # graphs the pca of each song in one playlist as red dots and the geometric median of the whole playlist as a black dot
    def graph_pca(self, pca):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Music Preference of One SPIS Member")
        ax.set_xlabel('Principle Component 1')
        ax.set_ylabel('Principle Component 2')
        ax.set_zlabel('Principle Component 3')

        for coor in pca:
            ax.scatter3D(coor[0], coor[1], coor[2], color='red', alpha=1.0)
            x, y, z = self.find_geo_med(pca)
            ax.scatter3D(x, y, z, color='black', alpha=1.0)

    # finds the optimal number of clusters for the data by calculating the elbow of the graph of clusters-variance
    def find_num_clusters(self):
        sse = []
        k_rng = range(1, len(playlist_list))

        for i in range(len(k_rng)):
            self.km.append(KMeans(n_clusters=k_rng[i]))
            self.km[i].fit(self.df_geomeds[['x_coor', 'y_coor', 'z_coor']])
            sse.append(self.km[i].inertia_)

        areas = []
        for j in range(len(k_rng)):
            T = (1 / 2) * abs(
                (k_rng[0] - k_rng[j]) * (sse[len(sse) - 1] - sse[0]) - (k_rng[0] - k_rng[len(k_rng) - 1]) * (
                            sse[j] - sse[0]))
            areas.append(T)

        return areas.index(max(areas)) + 1


# this class sets up to find the best set of clusters by creating a variable number of clusters and graphing the cluster
class Spotify_Runner:

    # sets up the global variables that will hold the sets of cluster arrangements and their inertias
    def __init__(self, num_runs):
        self.spot_list = []
        self.inertias = []

        self.min_inertia_spot = -1
        self.best_num_cluster = -1

        self.num_runs = num_runs
        self.center_points = []
        self.first_pca = None

    # creates a num_runs number of cluster arrangements and sorts them to find the most efficient arrangement and graphing that arrangement
    def spot_creation(self):
        for i in range(self.num_runs):
            self.spot_list.append(Spotify_Clustering())

            if i == 0:
                for playlist in playlist_list:
                    pca = self.spot_list[i].pca_calculation(playlist)
                    if playlist_list.index(playlist) == 0:
                        self.first_pca = pca
                    x, y, z = self.spot_list[i].find_geo_med(pca)
                    self.center_points.append([x, y, z])

            self.spot_list[i].center_points = self.center_points
            self.spot_list[i].df_geomeds = df(self.center_points, columns=['x_coor', 'y_coor', 'z_coor'])

            if i == 0:
                list_num_cluster = []
                for j in range(25):
                    list_num_cluster.append(self.spot_list[0].find_num_clusters())

                self.best_num_cluster = statistics.mode(list_num_cluster)

            self.spot_list[i].find_num_clusters()

            self.inertias.append(self.spot_list[i].km[self.best_num_cluster - 1].inertia_)
            if min(self.inertias) == self.spot_list[i].km[self.best_num_cluster - 1].inertia_:
                self.min_inertia_spot = i

        self.spot_list[self.min_inertia_spot].graph_pca(self.first_pca)
        self.spot_list[self.min_inertia_spot].graph_geo_med()
        self.spot_list[self.min_inertia_spot].graph_roles()
        self.spot_list[self.min_inertia_spot].graph_clusters(self.best_num_cluster)

        print("\n\n\n")
        for i in range(self.best_num_cluster):
            list_names = [key for key, value in self.spot_list[self.min_inertia_spot].cluster_name.items() if
                          value == i]
            print("Cluster " + str(i + 1) + ": ", end='')
            for j in range(len(list_names) - 1):
                print(str(list_names[j]), end=', ')
            print(list_names[len(list_names) - 1])


runner = Spotify_Runner(100)
runner.spot_creation()
print("\n\n\n")

plt.show()