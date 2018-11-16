import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def readHDF5(filename, setfig=None, dstack=True, normalize=True, norm_max=1024):
    resolve = dict()
    f = h5py.File(filename, 'r')
    if 'ir1' in setfig:
        resolve['ir1'] = f['HDFEOS/GRIDS/IR Image Pixel Value/Data Fields/IR1 band Image Pixel Values']
    if 'ir2' in setfig:
        resolve['ir2'] = f['HDFEOS/GRIDS/IR Image Pixel Value/Data Fields/IR2 band Image Pixel Values']
    if 'swir' in setfig:
        resolve['swir'] = f['HDFEOS/GRIDS/IR Image Pixel Value/Data Fields/SWIR band Image Pixel Values']
    if 'wv' in setfig:
        resolve['wv'] = f['HDFEOS/GRIDS/IR Image Pixel Value/Data Fields/WV band Image Pixel Values']
    if 'vis' in setfig:
        resolve['vis'] = f['HDFEOS/GRIDS/Visible Image Pixel Value/Data Fields/Visible Image Pixel Values']

    if dstack:
        cat_list = list()

        for _, v in resolve.items():
            cat_list.append(np.array(v))

        resolve = np.dstack(cat_list)
    else:
        pass

    if normalize:
        larger_than = resolve > norm_max
        resolve[larger_than] = norm_max - 1
        resolve = np.true_divide(resolve, norm_max)
    else:
        pass

    return resolve

def image_hist_eq(image, number_bins=512):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

def showBinary(filename=None, data=None, shape=None, grayscale=True, normalize=False):
    _cmap = plt.cm.Greys_r if grayscale else None
    _coolwarm = plt.cm.coolwarm
    if filename is not None:
        arr = np.fromfile(filename, dtype='u2')
    elif data is not None:
        arr = data
    if normalize:
        arr, _ = image_hist_eq(arr)
    if shape is not None:
        arr = arr.reshape(shape)
    fig, ax = plt.subplots(1, 1)
    cax = ax.imshow(arr, aspect='auto', cmap=_cmap)

    ax.xaxis.set_ticks_position("top")
    fig.set_size_inches(24, 15)
    fig.colorbar(cax)

    plt.show()

def resolve_inv_y(height, ypos, switch=True):

    if switch:
        resolved = height - ypos
    else:
        resolved = ypos
    return resolved

def plotmtrx2D(data=None, shapedict=None, showText=False, backImg=None, backImgCm=None, plotshape=None, figsize=None,
               inv_y=True):
    _shape = shapedict if shapedict != None else {'height': data.shape[0], 'width': data.shape[1]}

    # matplotlib-based method

    # fig = plt.figure()
    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_subplot(111,projection=proj)

    fig.set_size_inches(figsize[0], figsize[1])

    if backImg is not None:
        backImg2plot = np.array(backImg, dtype=np.uint16)
        ax.imshow(backImg2plot, extent=plotshape, cmap=backImgCm)

    for i in range(_shape['height']):
        for j in range(_shape['width']):
            if showText:
                plt.text(j + 0.5, i + 0.5, '%.2f' % data[i, j],
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
            else:
                continue

    data2plot = data
    if inv_y:
        data2plot = np.flip(data, 0)
    else:
        pass
    # plt.colorbar(plt.pcolor(data, cmap='bwr', alpha=0.3))
    plt.colorbar(plt.pcolor(data2plot, cmap='RdGy', alpha=0.3))
    plt.clim(0, 1)
    # plt.gca().invert_yaxis()

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_ticks_position("top")

    plt.show()

def plotmtrx3D(data=None):
    fig = plt.figure()

    fig.set_size_inches(20, 10)
    ax = fig.gca(projection='3d')
    ds_shape = data.shape
    X = range(ds_shape[0])
    Y = range(ds_shape[1])
    X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    Z = data
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # ax.set_zlim(-1.01, 1.01)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plotmtrx(data, act=True, ptype='2d', shapedict=None, showText=False, backImg=None, backImgCm=None,
             plotshape=None, figsize=[10, 10]):
    if act == True:

        if ptype == '2d':
            plotmtrx2D(data=data, shapedict=shapedict, showText=showText, backImg=backImg, backImgCm=backImgCm,
                       plotshape=plotshape, figsize=figsize)
        elif ptype == '3d':
            plotmtrx3D(data=data)
        else:
            pass

    else:
        pass

def plot_pred_class(data_label=None, data_pred=None, shapedict=None,
                    backimg=None, backimg_cmap='gray', pred_cmap='RdGy',
                    plotshape=None, figsize=[10, 10], inv_y=True):

    _shape = shapedict if shapedict != None else {'height': data_label.shape[0], 'width': data_label.shape[1]}

    # matplotlib-based method

    # fig = plt.figure()
    fig, ax = plt.subplots(1, 1)
    # ax = fig.add_subplot(111,projection=proj)

    fig.set_size_inches(figsize[0], figsize[1])

    if backimg is not None:
        backimg2plot = np.array(backimg, dtype=np.uint16)
        ax.imshow(backimg2plot, extent=plotshape, cmap=backimg_cmap)
    else:
        pass

    if inv_y:
        data2plot_label = np.flip(data_label, 0)
        data2plot_pred = np.flip(data_pred, 0)
    else:
        data2plot_label = data_label
        data2plot_pred = data_pred

    for i in range(_shape['height']):
        for j in range(_shape['width']):

            label = data2plot_label[i, j]
            if label == 1:
                plt.text(j + 0.5, i + 0.5, '*', color='yellow', size='x-large', weight='bold',
                         horizontalalignment='center',
                         verticalalignment='center',
                         )
            else:
                continue

    plt.colorbar(plt.pcolor(data2plot_pred, cmap=pred_cmap, alpha=0.3))
    plt.clim(0, 1)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_ticks_position("top")

    plt.show()
