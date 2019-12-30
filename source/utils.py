import tensorflow as tf
import os
import numpy as np
import imageio
import scipy.stats as st
import scipy.misc as mc
import random

from scipy import signal

from config import config

# Paramters
patch_size = config.TRAIN.patch_size
eps = config.TRAIN.eps
lambda_ir = config.TRAIN.lambda_ir

def write_logs(filename, log, start=False):
  print(log)
  if start:
    f = open(filename, 'w')
    f.write(log + '\n')
    f.close()
  else:
    f = open(filename, 'a')
    f.write(log + '\n')
    f.close()

def get_filepath(path, suffix):
  filepath = []
  filename = []
  for f in os.listdir(path):
    if f.endswith(suffix):
      filepath.append(os.path.join(path, f))
      filename.append(f)
  filepath = sorted(filepath)
  filename = sorted(filename)
  return filepath, filename

def exposure(hdr_img):
  L = (hdr_img[:,:,0]+hdr_img[:,:,1]+hdr_img[:,:,2])/3
  Ls = np.sort(L, axis=None)
  th = np.minimum(np.maximum(random.gauss(0,1)*0.05+0.9, 0.85), 0.95)
  th = int(len(Ls)*th)
  Hth = Ls[th]
  return Hth

def sigma_n():
  sigma_mean = 0.6
  sigma_std = 0.1
  sigma = np.minimum(np.maximum(sigma_std*random.gauss(0,1)+sigma_mean, 0.0), 5.0)

  n_mean = 0.9
  n_std = 0.1
  n = np.minimum(np.maximum(n_std*random.gauss(0,1)+n_mean, 0.2), 2.5)

  sigma = sigma.astype('float32')
  n = n.astype('float32')

  return sigma, n

def norm_img(hdr):
  hdr = tf.maximum(1e-5, tf.minimum(hdr,100000.0))

  Hth = tf.py_func(exposure, [hdr], [tf.float32])
  Hth = tf.reshape(Hth, [1,1,1])
  hdr = tf.div(hdr, Hth)

  sigma, n = tf.py_func(sigma_n, [], [tf.float32, tf.float32])
  sigma = tf.reshape(sigma, [1,1,1])
  n = tf.reshape(n, [1,1,1])

  tmp = tf.pow(hdr, n)
  ldr = tf.multiply(1.0+sigma, tf.div(tmp, tmp+sigma))
  ldr = (255.0*tf.minimum(1.0,ldr)+0.5)/255.0
  
  ldr = tf.clip_by_value(ldr, 0, 1)
  hdr = tf.clip_by_value(hdr, 0, 10)

  return ldr, hdr, Hth

def img_write(img_dir, img_name, img, fmt):
  _, h, w, c = img.shape
  img = np.reshape(img, [h,w,c])
  path = os.path.join(img_dir, img_name)
  imageio.imwrite(path, img, fmt)

def alpha_msk(ldr_in):
  _, h, w, _ = ldr_in.shape
  tau = 0.95
  alpha = tf.reduce_max(ldr_in, reduction_indices=[3])
  alpha = tf.minimum(1.0, tf.maximum(0.0, alpha-tau)/(1-tau))
  alpha = tf.reshape(alpha, [-1,h,w,1])
  alpha = tf.tile(alpha, [1,1,1,3])
  return alpha

def loss_fn(ldr_in, hdr_out, hdr_tar):
  alpha = alpha_msk(ldr_in)

  hdr_tar_log = tf.log(hdr_tar + eps)

  # Luminance
  W = np.zeros((1,1,3,1))
  W[:,:,0,0] = 0.213
  W[:,:,1,0] = 0.715
  W[:,:,2,0] = 0.072
  hdr_tar_lum = tf.nn.conv2d(hdr_tar, W, [1,1,1,1], 'SAME')
  hdr_out_lum = tf.nn.conv2d(tf.exp(hdr_out)-eps, W, [1,1,1,1], 'SAME')
  hdr_tar_lum = tf.maximum(hdr_tar_lum, 0.0)
  hdr_out_lum = tf.maximum(hdr_out_lum, 0.0)

  # Log Luminance
  hdr_tar_lum = tf.log(hdr_tar_lum + eps)
  hdr_out_lum = tf.log(hdr_out_lum + eps)

  # Gaussian kernel
  nsig = 2
  filter_size = 13
  interval = (2 * nsig + 1.0) / filter_size
  l1 = np.linspace(-nsig-interval/2.0, nsig+interval/2.0, filter_size+1)
  kern1d = np.diff(st.norm.cdf(l1))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw / kernel_raw.sum()

  # Illumination, approximated by means of Gaussian filtering
  w_g = np.zeros((filter_size, filter_size, 1, 1))
  w_g[:,:,0,0] = kernel
  hdr_tar_ill = tf.nn.conv2d(hdr_tar_lum, w_g, [1,1,1,1], 'SAME')
  hdr_out_ill = tf.nn.conv2d(hdr_out_lum, w_g, [1,1,1,1], 'SAME')

  # Reflectance
  hdr_tar_refl = hdr_tar_log - tf.tile(hdr_tar_ill, [1,1,1,3])
  hdr_out_refl = hdr_out - tf.tile(hdr_out_ill, [1,1,1,3])

  irloss = tf.reduce_mean((lambda_ir*tf.square(hdr_out_ill-hdr_tar_ill)+(1-lambda_ir)*tf.square(hdr_out_refl-hdr_tar_refl))*alpha)
  dirloss = tf.reduce_mean(tf.square(tf.subtract(hdr_out,hdr_tar_log)*alpha))
  
  return irloss, dirloss

def inv_cam_curve(x):
  sigma = 0.6
  n = 0.9
  M = tf.scalar_mul(sigma, x)
  N = tf.maximum(0.0, 1.0+sigma-x)
  y = tf.div(M, N)
  y = tf.pow(y, 1.0/n)
  return y

def reinhard02(hdr_img, a):
  _, h, w, _ = np.shape(hdr_img)
  
  R = hdr_img[0][:,:,0]
  G = hdr_img[0][:,:,1]
  B = hdr_img[0][:,:,2]

  #L = 0.213*R + 0.715*G + 0.072*B
  L = (R + G + B)/3
  L_avg = np.prod(L)
  L_avg = np.power(L_avg, 1/(h*w))
  
  L_scaled = np.multiply(np.divide(a, L_avg+eps), L)
  L_white = np.amax(L_scaled)

  L_d = np.multiply(np.add(1.0, np.divide(L_scaled, np.power(L_white,2)+eps)), L_scaled)
  L_d = np.divide(L_d, np.add(1.0, L_scaled))
  M = np.divide(L_d, L+eps)
  
  Rnew = np.multiply(M, R)
  Gnew = np.multiply(M, G)
  Bnew = np.multiply(M, B)
  
  R = np.expand_dims(R, axis=2)
  G = np.expand_dims(G, axis=2)
  B = np.expand_dims(B, axis=2)
  
  ldr_img = np.concatenate((R, G, B), axis=2)
  ldr_img = np.expand_dims(ldr_img, axis=0)
  ldr_img = ldr_img * 255.0
  ldr_img = np.clip(ldr_img, 0, 255)
  ldr_img = ldr_img.astype('uint8')
  
  return ldr_img

