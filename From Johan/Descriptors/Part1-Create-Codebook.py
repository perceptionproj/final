from os.path import exists, isdir, basename, join, splitext
import cv2
from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import numpy as np
import timeit
import time
import matplotlib.pyplot as plt
import scipy.cluster.vq as vq
from pickle import dump, HIGHEST_PROTOCOL
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans


# code snippets is loaned from:
# https://github.com/shackenberg/Minimal-Bag-of-Visual-Words-Image-Classifier/blob/master/learn.py

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
PRE_ALLOCATION_BUFFER = 1000

datasetpath = "/home/jo/PycharmProjects/Perception/dataset_full"

CODEBOOK_FILE = 'codebook.file'



#desc = cv2.xfeatures2d.SURF_create(400) #TJEK THREDSHOLD !!!!
desc = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.005,edgeThreshold=80) #contrastThreshold=0.01,edgeThreshold=40


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('k')
    plt.ylabel('error')
    plt.savefig('elbow.png')


def get_categories(datasetpath):
    cat_paths = [files for files in glob(datasetpath + "/*") if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname)) for fname in glob(path + "/*.*", recursive=True)])

    return all_files


def extractSift(input_files):
    all_features_dict = {}
    for i in input_files:

        image = cv2.imread(i)
        #image = image_resize(image, height=200)
        # cv2.imshow('image', image)

        #image = cv2.GaussianBlur(image, (5, 5), 0)
        #image = cv2.addWeighted(image, 1.5, image, -0.5, 0)
        """
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])

        # applying the sharpening kernel to the input image & displaying it.
        #
        image = cv2.filter2D(image, -1, kernel_sharpening)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)


        locs, descriptors = desc.detectAndCompute(gray, None)

        #only_object = cv2.drawKeypoints(image, locs, image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow('image', only_object)
        # cv2.imshow('image2',img2)
        #cv2.waitKey(2500)
        #cv2.destroyAllWindows()

        print("filename", i, " S descriptors shape", descriptors.shape)
        all_features_dict[i] = descriptors

    return all_features_dict


def dict2numpy(dict):
    an_array = np.array(list(all_features.values()))
    print(an_array.shape)
    print("****************")
    A = an_array[0]
    for i in range(1, len(an_array)):
        A = np.concatenate((A, an_array[i]), axis=0)

    print(A.shape)

    return A


def findOptimalK(X):
    #global kmeans
    distortions = []
    centroids = []
    # start_k = 1000
    # end_k = 6000
    # jump = 1000
    #k_value_list = 2 ** np.arange(12)
    #k_value_list = k_value_list[2:]
    k_value_list = [1000]

    print(k_value_list)
    for k in k_value_list:

        start = timeit.default_timer()
        print("tester for k=" , k)

        #kmeans = MiniBatchKMeans(n_clusters=k, init_size=10 * k).fit(X)
        kmeans = KMeans(n_clusters=k, precompute_distances=True, n_jobs=-1).fit(X)
        stop = timeit.default_timer()

        print('Time for k=', k, stop - start)
        print("error: ", kmeans.inertia_)
        print("")
        #centroids.append(kmeans.cluster_centers_)
        #print(len(kmeans.cluster_centers_))
        #distortions.append(kmeans.inertia_)
    #print(k_value_list)
    #print(distortions)
    #print("Best k value", k_value_list[np.argmin(distortions)])
    #np.savetxt('k_value_list.out', k_value_list, fmt='%d')
    #np.savetxt('distortions.out', distortions, fmt='%d')
    #np.savetxt('centroids.out', centroids[np.argmin(distortions)], fmt='%d')
    #plot(k_value_list, distortions)



    return kmeans.cluster_centers_, kmeans.inertia_ ,k


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,bins=range(codebook.shape[0] + 1))
    return histogram_of_words


def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print('nclusters have been reduced to ' + str(nwords))
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)


cats = get_categories(datasetpath)
print(cats)
ncats = len(cats)
print("searching for folders at " + datasetpath)
if ncats < 1:
    raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
print("found following folders / categories:")
print(cats)
print("---------------------")

all_files = []
all_files_labels = {}
all_features = {}
cat_label = {}
for cat, label in zip(cats, range(ncats)):
    cat_path = join(datasetpath, cat)
    cat_files = get_imgfiles(cat_path)
    cat_features = extractSift(cat_files)


    all_files = all_files + cat_files
    all_features.update(cat_features)
    cat_label[cat] = label
    for i in cat_files:
        all_files_labels[i] = label

# print("amount of cat features",len(all_features))
print("---------------------")
print("## computing the visual words via k-means")

all_features_array = dict2numpy(all_features)
codebook, distortion, nclusters = findOptimalK(all_features_array)

print("len codebook",codebook.shape)
print(type(codebook))

with open("/home/jo/PycharmProjects/Perception/desc/Codebooks/Codebook.out", 'wb') as f:
    dump(codebook, f, protocol=HIGHEST_PROTOCOL)

print("---------------------")
print("## compute the visual words histograms for each image")
all_word_histgrams = {}
for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram



histogram_array = np.array(list(all_word_histgrams.values()))
print(histogram_array.shape)
histogram_classes =np.array(list(all_files_labels.values()))
print(histogram_classes.shape)


#From dataset to learn SVM algorithm
np.savetxt('./Codebooks/histogram_array.out', histogram_array, fmt='%d')
np.savetxt('./Codebooks/histogram_classes.out', histogram_classes, fmt='%d')

print('Codebook and Histograms from dataset created')
