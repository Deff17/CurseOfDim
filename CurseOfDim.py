import matplotlib
from scipy.spatial.distance import cdist
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 10]
plt.style.use('ggplot')


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


#
def angle_case():
    N_MAX = 50
    M = int(1000)
    dims = np.zeros(N_MAX, dtype=np.int32)
    angle_mean = np.zeros(N_MAX)
    angle_std = np.zeros(N_MAX)
    angles_for_N_dim = np.zeros(M)
    for N in range(1, N_MAX + 1):
        x = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
        y = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
        p = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
        k = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
        vec1 = x - y
        vec2 = p - k
        dims[N - 1] = N
        for j in range(0, M):
            angles_for_N_dim[j] = angle_between(vec1[j], vec2[j])
        angle_mean[N - 1] = np.mean(angles_for_N_dim)
        angle_std[N - 1] = np.std(angles_for_N_dim)
    print(angle_mean)
    print(angle_std)
    df = pd.DataFrame(data={'dims': dims, 'angle': angle_mean})
    plt.errorbar(df.dims, df.angle, angle_std, marker='|', markersize=10)
    plt.plot(df.dims, df.angle, 'o-')
    plt.title('Angle between two random vectors', size=18)
    plt.xlabel('Dimensions', size=14)
    plt.ylabel('Angle', size=14)
    plt.show()


def standard_deviation_case():
    N_MAX = 10
    M = int(10000)
    dims = np.zeros(N_MAX, dtype=np.int32)
    ratio = np.zeros(N_MAX)
    ratio_probes = np.zeros((20, 10))
    ratio_for_N_dim = np.zeros((10, 20))
    ratio_std = np.zeros(10)
    ratio_mean = np.zeros(10)
    for i in range(1, 21):
        for N in range(1, N_MAX + 1):
            x = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
            y = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
            dist = cdist(x, y, metric='euclidean')
            mean_dist = np.mean(dist)
            standard_deviation = np.std(dist)
            dims[N - 1] = N
            ratio[N - 1] = standard_deviation / mean_dist
        ratio_probes[i - 1] = ratio

    for i in range(1, 11):
        for j in range(1, 21):
            ratio_for_N_dim[i - 1][j - 1] = ratio_probes[j - 1][i - 1]

    for i in range(0, 10):
        ratio_mean[i] = np.mean(ratio_for_N_dim[i])
        ratio_std[i] = np.std(ratio_for_N_dim[i])
    print(ratio_mean)
    print(ratio_std)
    df = pd.DataFrame(data={'dims': dims, 'ratio': ratio_mean})
    # print(df)
    plt.errorbar(df.dims, df.ratio, ratio_std, marker='|', markersize=10)
    plt.plot(df.dims, df.ratio, 'o-')
    plt.title('Ratio of standard deviation of distance to mean distance ', size=18)
    plt.xlabel('Dimensions', size=14)
    plt.ylabel('Ratio', size=14)
    plt.show()


def volum_case():
    N_MAX = 10
    M = int(1e+5)
    dims = np.zeros(N_MAX, dtype=np.int32)
    volume = np.zeros(N_MAX)
    volumes_probes = np.zeros((100, 10))
    volumes_for_N_dim = np.zeros((10, 100))
    volume_std = np.zeros(10)
    volumes_mean = np.zeros(10)
    for i in range(1, 101):

        for N in range(1, N_MAX + 1):
            y = np.random.uniform(low=-0.5, high=0.5, size=(M, N))
            dist = cdist(y, np.expand_dims(np.zeros(N), 0), metric='euclidean')
            p = np.sum(dist < 0.5) / M
            dims[N - 1] = N
            volume[N - 1] = p
        # dims_probes[i - 1] = dims
        volumes_probes[i - 1] = volume
    for i in range(1, 11):
        for j in range(1, 101):
            volumes_for_N_dim[i - 1][j - 1] = volumes_probes[j - 1][i - 1]

    for i in range(0, 10):
        volumes_mean[i] = np.mean(volumes_for_N_dim[i])
        volume_std[i] = np.std(volumes_for_N_dim[i])
    print(volume_std)
    print(volumes_mean)
    df = pd.DataFrame(data={'dims': dims, 'volume': volumes_mean, 'error': volume_std})
    # print(df)
    plt.plot(df.dims, df.volume, 'o-')
    plt.title('Volume hypersphere in unit hypercube', size=18)
    plt.xlabel('Dimensions', size=14)
    plt.ylabel('Volume of hyperball', size=14)
    plt.errorbar(df.dims, df.volume, volume_std, marker='+')
    plt.show()


if __name__ == '__main__':
    #volum_case()
    #standard_deviation_case()
    angle_case()
