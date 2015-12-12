import os
import sys
import numpy as np
import stpyr
import scipy.integrate
import scipy.ndimage
import scipy.signal
import skimage.io
from time import clock, time
import pickle
from PIL import Image
from sklearn.externals import joblib
#from scipy.linalg.fblas import dger, dgemm
from scipy.linalg.blas import dgemm

from sklearn import svm, datasets, ensemble

def gauss_window(lw, sigma):
    sd = float(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights
avg_window = gauss_window(3, 7.0/6.0)

extend_mode = 'constant'#'nearest'#'wrap'
def SSIM(img1, img2):
    avg_window = gauss_window(5, 1.5)
    K_1 = 0.01;				
    K_2 = 0.03;
    L = 1;

    C1 = (K_1*L)**2;
    C2 = (K_2*L)**2;
#
    h, w = np.shape(img1)
    mu1 = np.zeros((h, w))
    mu2 = np.zeros((h, w))
    scipy.ndimage.correlate1d(img1, avg_window, 0, mu1, mode=extend_mode) 
    scipy.ndimage.correlate1d(mu1, avg_window, 1, mu1, mode=extend_mode) 
    scipy.ndimage.correlate1d(img2, avg_window, 0, mu2, mode=extend_mode) 
    scipy.ndimage.correlate1d(mu2, avg_window, 1, mu2, mode=extend_mode) 
    
    mu1_sq = mu1**2;
    mu2_sq = mu2**2;
    mu1_mu2 = mu1*mu2;

    var1 = np.zeros((h, w))
    var2 = np.zeros((h, w))
    var12 = np.zeros((h, w))

    scipy.ndimage.correlate1d(img1**2, avg_window, 0, var1, mode=extend_mode) 
    scipy.ndimage.correlate1d(var1, avg_window, 1, var1, mode=extend_mode) 
    scipy.ndimage.correlate1d(img2**2, avg_window, 0, var2, mode=extend_mode) 
    scipy.ndimage.correlate1d(var2, avg_window, 1, var2, mode=extend_mode) 
    scipy.ndimage.correlate1d(img1*img2, avg_window, 0, var12, mode=extend_mode) 
    scipy.ndimage.correlate1d(var12, avg_window, 1, var12, mode=extend_mode) 

    sigma1_sq = var1 - mu1_sq;
    sigma2_sq = var2 - mu2_sq;
    sigma12 = var12 - mu1_mu2;

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12+C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_map[5:-5, 5:-5]

    return np.mean(ssim_map)

#generate noise 
def noise_covariance(pyr_lvl, orientation, noise_level, cov_mt, t):
    tstr = ''
    if t == 1:
        tstr = '_h'
    power_sum = np.sum(np.abs(cov_mt))
    if pyr_lvl == 0:
        #compute power
        cov = np.zeros((15*15))
        coefs_lvl0 = joblib.load("pkls/poly_lvl0" + tstr + ".pkl")
        for i in xrange(15*15):
            func = np.poly1d(coefs_lvl0[i, :])
            cov[i] = func(noise_level)*power_sum
        cov = cov.reshape((15, 15))
        return cov
    if pyr_lvl == 1 or pyr_lvl == 2:
        cov = np.zeros((15*15))
#	power_sum = np.sum(np.abs(cov_mt[orientation, :, :]))
        coefs_lvl12 = joblib.load("pkls/poly_lvl" + str(pyr_lvl) + tstr + ".pkl")
        for i in xrange(15*15):
            func = np.poly1d(coefs_lvl12[orientation, i, :])
            cov[i] = func(noise_level)*power_sum
        cov = cov.reshape((15, 15))
        return cov
    if pyr_lvl == 3:
    #	power_sum = np.sum(np.abs(cov_mt[orientation, :, :]))
        cov = np.zeros((14*14))
        coefs_lvl3 = joblib.load("pkls/poly_lvl" + str(pyr_lvl) + tstr + ".pkl")
        for i in xrange(14*14):
            func = np.poly1d(coefs_lvl3[orientation, i, :])
            cov[i] = func(noise_level)*power_sum
        cov = cov.reshape((14, 14))
        return cov

def mean_cov(X):
    n,p = X.shape
    m = X.mean(axis=0)
    cx = X - m
    S = dgemm(1./(n-1), cx.T, cx.T, trans_a=0, trans_b=1)
    return cx,m,S.T

def rolling_window_lastaxis(a, window):
	"""Directly taken from Erik Rigtorp's post to numpy-discussion.
	<http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
	if window < 1:
		raise ValueError, "`window` must be at least 1."
	if window > a.shape[-1]:
		raise ValueError, "`window` is too long."
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
	if not hasattr(window, '__iter__'):
		return rolling_window_lastaxis(a, window)
	for i, win in enumerate(window):
		if win > 1:
			a = a.swapaxes(i, -1)
			a = rolling_window_lastaxis(a, win)
			a = a.swapaxes(-2, i)
	return a

def load_ir(name):
	im = Image.open(name)
	arr = np.array(im.getdata())
	arr = arr.astype('float')
#	if "halos" in name:
#		arr = np.log(arr + 1.0)
		#scale from 0 to 1
#		arr = 1.0/(np.log(65536.0)) * (arr)
	#	print name
#	else:
	maxval = np.max(arr)
	if maxval > 300:
		arr /= 65535.0
	else:
		arr /= 255.0

	#take log of arr if name mentions OSU
	


	#arr *= -1
	#hack because PIL reads as signed integer data
	#arr[arr<0] += 65535
	#arr += np.abs(np.min(arr))
	#print np.shape(arr)
	if len(np.shape(arr)) > 1:
		a, b = np.shape(arr)
		arr = arr.reshape(im.size[1], im.size[0], b)
		arr = arr[:, :, 0]
	else:
		arr = arr.reshape(im.size[1], im.size[0])

	#go ahead and dsX2
	#pick center 1/4
	w, h = np.shape(arr)
	#multiscale
	arr2 = scipy.ndimage.filters.gaussian_filter(arr, sigma=7.0/6.0, order=0, output=None, mode='nearest')
	arr2 = arr2[::2, ::2]
	arr4 = scipy.ndimage.filters.gaussian_filter(arr2, sigma=7.0/6.0, order=0, output=None, mode='nearest')
	arr4 = arr4[::2, ::2]


	#arr = arr[w/4:(w/4)*3, h/4:(h/4)*3]
	return arr, arr2, arr4

#estimate Z
def estimate_params(coef, height, order, noise_level):
	filtsize = (3, 3)
	z_bands = []
	cov_bands = []
	vec_bands = []
	#first estimate z and coef centered on parent 
	w, h = np.shape(coef[0])
	sublevel = coef[1]
	cov = np.array(np.hstack((
			#split image into overlapping blocks
			rolling_window(coef[0], filtsize).reshape(((w-2)*(h-2), 9)),
			#grab coefficients from neighboring orientations
			sublevel.transpose(1, 2, 0)[1:-1, 1:-1, :].reshape((w-2)*(h-2), order),
#			sublevel.transpose(1, 2, 0)[1:-1, 2:, :].reshape((w-2)*(h-2), order),
#			sublevel.transpose(1, 2, 0)[1:-1, :-2, :].reshape((w-2)*(h-2), order),
#			sublevel.transpose(1, 2, 0)[:-2, :-2, :].reshape((w-2)*(h-2), order),
#			sublevel.transpose(1, 2, 0)[2:, 2:, :].reshape((w-2)*(h-2), order),
#			sublevel.transpose(1, 2, 0)[:-2, 1:-1, :].reshape((w-2)*(h-2), order),
#			sublevel.transpose(1, 2, 0)[2:, 1:-1, :].reshape((w-2)*(h-2), order),
			#np.matrix(child[1:-1, 1:-1].reshape((w-2)*(h-2))).T,
	)))
	vec_bands.append(cov)
	_, _, cov_mat = mean_cov(cov)

	#cov_mat = noise_covariance(0, 0, noise_level) + nat_covariance(0, 0)
	#force positive semi-definite
	eigval, eigvec = np.linalg.eig(cov_mat)
	Q = np.matrix(eigvec)
	xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
	cov_mat = Q*xdiag*Q.T
	cov_bands.append(cov_mat)
	cov_inv = np.linalg.pinv(cov_mat)
	N = np.shape(cov_mat)[0]
	z = np.sqrt(np.einsum('ij,ij->i', np.dot(cov, cov_inv), cov)/N)
	result = z
	result = result.reshape(w-2, h-2)#[gb:-(gb), gb:-(gb)]
	z_bands.append(result)

	for pyr_h in xrange(height-2):
		inner_norm_bands = []
		inner_cov_bands = []
		inner_vec_bands = []
		sublevel = coef[pyr_h+1]
		for cband in xrange(order):
			child = coef[0]
			parent = []
			w, h = np.shape(sublevel[cband])
			if pyr_h > 0:
				child= scipy.misc.imresize(coef[pyr_h][cband], (w,h), interp='bilinear', mode='F')
			if pyr_h+3 < height:
				parent = scipy.misc.imresize(coef[pyr_h+2][cband], (w,h), interp='bilinear', mode='F')
				parent = parent[1:-1, 1:-1]
				wp, hp = np.shape(parent)
				#print np.shape(parent)
				if wp > w-2:
					parent = parent[:w-2, :]
				if hp > h-2:
					parent = parent[:, :h-2]
				#print np.shape(parent)
				#print np.shape(parent)
				#exit(0)


			idx = np.hstack((np.arange(0, cband), np.arange(cband+1, order)))

			#stick it all in a matrix
			if parent == []:
				cov = np.array(np.hstack((
						#split image into overlapping blocks
						rolling_window(sublevel[cband], filtsize).reshape(((w-2)*(h-2), 9)),
						#grab coefficients from neighboring orientations
						sublevel.transpose(1, 2, 0)[1:-1, 1:-1, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[1:-1, 2:, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[1:-1, :-2, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[:-2, :-2, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[2:, 2:, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[:-2, 1:-1, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[2:, 1:-1, idx].reshape((w-2)*(h-2), order-1),
						#np.matrix(child[1:-1, 1:-1].reshape((w-2)*(h-2))).T,
				)))
			else:
				#parent sometimes gets an extra pixel 
				#print np.shape(np.matrix(parent[1:-1, 1:-1]))#.reshape((w-2)*(h-2))).T,
				#print w-2, h-2
				cov = np.array(np.hstack((
						rolling_window(sublevel[cband], filtsize).reshape(((w-2)*(h-2), 9)),
						#grab coefficients from neighboring orientations
						sublevel.transpose(1, 2, 0)[1:-1, 1:-1, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[1:-1, 2:, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[1:-1, :-2, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[:-2, :-2, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[2:, 2:, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[:-2, 1:-1, idx].reshape((w-2)*(h-2), order-1),
						#sublevel.transpose(1, 2, 0)[2:, 1:-1, idx].reshape((w-2)*(h-2), order-1),
						np.matrix(parent.reshape((w-2)*(h-2))).T,
						#np.matrix(child[1:-1, 1:-1].reshape((w-2)*(h-2))).T,
				)))
			inner_vec_bands.append(cov)
			_, _, cov_mat = mean_cov(cov)

			N = np.shape(cov_mat)[0]

			#cov_mat = noise_covariance(pyr_h+1, cband, noise_level) + nat_covariance(pyr_h+1, cband)
			#force positive semi-definite
			eigval, eigvec = np.linalg.eig(cov_mat)
			Q = np.matrix(eigvec)
			xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
			cov_mat = Q*xdiag*Q.T


			#reference code claims to be correcting the cov matrix, by below basic computation
			#L = diag(diag(L).*(diag(L)>0))*sum(diag(L))/(sum(diag(L).*(diag(L)>0))+(sum(diag(L).*(diag(L)>0))==0));
			inner_cov_bands.append(cov_mat)
			cov_inv = np.linalg.pinv(cov_mat)

			# perform normalization by sqrt(Y^T * C_U * Y)
			z = np.sqrt(np.einsum('ij,ij->i', np.dot(cov, cov_inv), cov)/N)
			#inner = np.sum(np.multiply(np.dot(cov, cov_inv), cov), axis=1).T/N
			#print np.shape(z)
			#exit(0)

			cov[:, 4] -= np.average(cov[:, 4])

			result = z #cov[:, 4]/z
			#result = load_mat("g_c" + str(cband+1) + "_" + str(pyr_h + 1) + ".mat", "g_c")
			#result = cov[:, 4]/divisors
			gb = 16/(2**(pyr_h))

			result = result.reshape(w-2, h-2)#[gb:-(gb), gb:-(gb)]
			#result -= np.average(result)
			inner_norm_bands.append(result)

		z_bands.append(inner_norm_bands)
		cov_bands.append(inner_cov_bands)
		vec_bands.append(inner_vec_bands)
	z_bands = np.array(z_bands)
	return z_bands, cov_bands, vec_bands

def force_psd(ccc):
	eigval, eigvec = np.linalg.eig(ccc)
	Q = np.matrix(eigvec)
	xdiag = np.matrix(np.diag(np.maximum(eigval, 0)))
	return Q*xdiag*Q.T

# run with a file as input
input_file = sys.argv[1]
noise_level_v = np.float(sys.argv[2])

if (noise_level_v <= 0 or noise_level_v > 0.1):
    print "Try to set noise level between 0 and 0.1"
    exit(0)

bname = os.path.basename(input_file)

im, im2, im3 = load_ir(input_file)

#decompose into subbands
od=6
height=5
stobj = stpyr.Steerable(height=height, order=od)
coef_nois = stobj.buildSFpyr(im)

def save(name, data):
    data -= np.min(data)
    data /= np.max(data)
    data = np.log(data + 1)
    data -= np.min(data)
    data /= np.max(data)
    data *= 255
    data = data.astype(np.uint8)
    skimage.io.imsave(name, data)

# output 
h, w = np.shape(coef_nois[0])

#for gaussianization
z_gamma_noise, sigma_gamma_noise, vecs_gamma_noise = estimate_params(coef_nois, height, od, noise_level_v)

#make sure that sigma_gamma + noise is not actually subtracting energy from sigma_gamma

sigma_noise = noise_covariance(0, 0, noise_level_v, sigma_gamma_noise[0], 0)
sigma_gamma = force_psd(sigma_gamma_noise[0] - sigma_noise)
sigma_noise = force_psd(sigma_noise)

z1 = z_gamma_noise[0].reshape(h-2, w-2)
v1 = vecs_gamma_noise[0].reshape(h-2, w-2, 15).transpose((2, 0, 1))
for y in xrange(1, h-1):
    for x in xrange(1, w-1):
        m1 = z1[y-1, x-1]*np.linalg.inv(z1[y-1, x-1]*sigma_gamma + sigma_noise)
        rv1 = np.dot(np.dot(sigma_gamma, m1), v1[:, y-1, x-1])
        coef_nois[0][y, x] = rv1[0, 4]

#inverse
for angle in xrange(0, od):
    h, w = np.shape(coef_nois[1][angle])
    sigma_noise = noise_covariance(1, angle, noise_level_v, sigma_gamma_noise[1][angle], 0)
    sigma_gamma = force_psd(sigma_gamma_noise[1][angle] - sigma_noise)
    sigma_noise = force_psd(sigma_noise)

    z1 = z_gamma_noise[1][angle].reshape(h-2, w-2)
    v1 = vecs_gamma_noise[1][angle].reshape(h-2, w-2, 15).transpose((2, 0, 1))
    for y in xrange(1, h-1):
        for x in xrange(1, w-1):
            m1 = z1[y-1, x-1]*np.linalg.pinv(z1[y-1, x-1]*sigma_gamma + sigma_noise)
            rv1 = np.dot(np.dot(sigma_gamma, m1), v1[:, y-1, x-1])
            coef_nois[1][angle][y, x] = rv1[0, 4]

    h, w = np.shape(coef_nois[2][angle])

    sigma_noise = noise_covariance(2, angle, noise_level_v, sigma_gamma_noise[2][angle], 0)
    sigma_gamma = force_psd(sigma_gamma_noise[2][angle] - sigma_noise)
    sigma_noise = force_psd(sigma_noise)

    z1 = z_gamma_noise[2][angle].reshape(h-2, w-2)
    v1 = vecs_gamma_noise[2][angle].reshape(h-2, w-2, 15).transpose((2, 0, 1))
    for y in xrange(1, h-1):
        for x in xrange(1, w-1):
            m1 = z1[y-1, x-1]*np.linalg.pinv(z1[y-1, x-1]*sigma_gamma + sigma_noise)
            rv1 = np.dot(np.dot(sigma_gamma, m1), v1[:, y-1, x-1])
            coef_nois[2][angle][y, x] = rv1[0, 4]

    h, w = np.shape(coef_nois[3][angle])

    sigma_noise = noise_covariance(3, angle, noise_level_v, sigma_gamma_noise[3][angle], 0)
    sigma_gamma = force_psd(sigma_gamma_noise[3][angle] - sigma_noise)
    sigma_noise = force_psd(sigma_noise)

    z1 = z_gamma_noise[3][angle].reshape(h-2, w-2)
    v1 = vecs_gamma_noise[3][angle].reshape(h-2, w-2, 14).transpose((2, 0, 1))
    for y in xrange(1, h-1):
        for x in xrange(1, w-1):
            m1 = z1[y-1, x-1]*np.linalg.pinv(z1[y-1, x-1]*sigma_gamma + sigma_noise)#*z1[y-1,x-1])
            rv1 = np.dot(np.dot(sigma_gamma, m1), v1[:, y-1, x-1])
            coef_nois[3][angle][y, x] = rv1[0, 4]

cim = stobj.reconSFpyr(coef_nois)

# make sure the pixels are within bit depth
cim[cim>1] = 1
cim[cim<0] = 0

output_path = "corrected_" + os.path.basename(input_file)
skimage.io.imsave(output_path, cim)
