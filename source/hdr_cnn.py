import tensorflow as tf
import numpy as np

from config import config
from utils import inv_cam_curve, alpha_msk

# Parameters
eps = config.TRAIN.eps

def get_final_hdr(ldr_in, hdr_nn):
  alpha = alpha_msk(ldr_in)

  # Linearization
  ldr_lin = inv_cam_curve(ldr_in)

  # Alpha blending
  hdr_final = (1-alpha)*ldr_lin + alpha*(tf.exp(hdr_nn))
  hdr_final = hdr_final

  return hdr_final

def hdrcnn(x, is_training=False, reuse=False):
  encoder_out, skip = encoder(x, is_training, reuse)
  decoder_in = tf.layers.conv2d(encoder_out, 512, 3, 1, 'same', activation=tf.identity, name='hdrcnn/conv', reuse=reuse)
  decoder_in = tf.layers.batch_normalization(decoder_in, training=is_training, name='hdrcnn/batch_norm', reuse=reuse)
  decoder_in = tf.nn.relu(decoder_in, 'hdrcnn/relu')
  y = decoder(decoder_in, skip, is_training, reuse)
  return y

def encoder(x, is_training, reuse):
  conv1_1 = conv2d_layer(x, 64, 'encoder/conv1_1', reuse)
  conv1_2 = conv2d_layer(conv1_1, 64, 'encoder/conv1_2', reuse)
  p1 = maxpool2d_layer(conv1_2, 'encoder/max_pool_1')

  conv2_1 = conv2d_layer(p1, 128, 'encoder/conv2_1', reuse)
  conv2_2 = conv2d_layer(conv2_1, 128, 'encoder/conv2_2', reuse)
  p2 = maxpool2d_layer(conv2_2, 'encoder/max_pool_2')

  conv3_1 = conv2d_layer(p2, 256, 'encoder/conv3_1', reuse)
  conv3_2 = conv2d_layer(conv3_1, 256, 'encoder/conv3_2', reuse)
  conv3_3 = conv2d_layer(conv3_2, 256, 'encoder/conv3_3', reuse)
  p3 = maxpool2d_layer(conv3_3, 'encoder/max_pool_3')

  conv4_1 = conv2d_layer(p3, 512, 'encoder/conv4_1', reuse)
  conv4_2 = conv2d_layer(conv4_1, 512, 'encoder/conv4_2', reuse)
  conv4_3 = conv2d_layer(conv4_2, 512, 'encoder/conv4_3', reuse)
  p4 = maxpool2d_layer(conv4_3, 'encoder/max_pool_4')

  conv5_1 = conv2d_layer(p4, 512, 'encoder/conv5_1', reuse)
  conv5_2 = conv2d_layer(conv5_1, 512, 'encoder/conv5_2', reuse)
  conv5_3 = conv2d_layer(conv5_2, 512, 'encoder/conv5_3', reuse)
  p5 = maxpool2d_layer(conv5_3, 'encoder/max_pool_5')

  return p5, (x, conv1_2, conv2_2, conv3_3, conv4_3, conv5_3)

def decoder(x, sk, is_training, reuse):
  tconv5 = tranpose_conv2d_layer(x, 512, is_training, 'decoder/tconv5', reuse)
  sk5 = skip_layer(tconv5, sk[5], is_training, 'decoder/skip5', reuse)

  tconv4 = tranpose_conv2d_layer(sk5, 512, is_training, 'decoder/tconv4', reuse)
  sk4 = skip_layer(tconv4, sk[4], is_training, 'decoder/skip4', reuse)

  tconv3 = tranpose_conv2d_layer(sk4, 256, is_training, 'decoder/tconv3', reuse)
  sk3 = skip_layer(tconv3, sk[3], is_training, 'decoder/skip3', reuse)

  tconv2 = tranpose_conv2d_layer(sk3, 128, is_training, 'decoder/tconv2', reuse)
  sk2 = skip_layer(tconv2, sk[2], is_training, 'decoder/skip2', reuse)

  tconv1 = tranpose_conv2d_layer(sk2, 64, is_training, 'decoder/tconv1', reuse)
  sk1 = skip_layer(tconv1, sk[1], is_training, 'decoder/skip1', reuse)
  
  conv0 = tf.layers.conv2d(sk1, 3, 1, 1, 'same',
                          activation=tf.identity,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                          bias_initializer=tf.zeros_initializer(),
                          name='decoder/conv0',
                          reuse=reuse)
  conv0 = tf.layers.batch_normalization(conv0, training=is_training, name='decoder/batch_norm', reuse=reuse)
  conv0 = tf.nn.leaky_relu(conv0, 0.0, 'decoder/leaky_relu')

  sk0 = skip_layer(conv0, sk[0], is_training, 'decoder/skip0', reuse)

  return sk0

def conv2d_layer(x, fsize, name, reuse):
  y = tf.layers.conv2d(x, fsize, 3, 1, 'same', activation=tf.nn.relu, name=name, reuse=reuse)
  return y

def maxpool2d_layer(x, name):
  y = tf.layers.max_pooling2d(x, 2, 2, 'same', name=name)
  return y

def tranpose_conv2d_layer(x, fsize, is_training, name, reuse):
  BK = np.zeros((4,4), dtype=np.float32)
  for i in range(4):
    for j in range(4):
      BK[i,j] = (1-abs(i-1.5)/2)*(1-abs(j-1.5)/2)

  y = tf.layers.conv2d_transpose(x, fsize, 4, 2, 'same',
                                activation=tf.identity,
                                kernel_initializer=tf.constant_initializer(BK, dtype=tf.float32),
                                bias_initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                                name='%s/tconv'%name,
                                reuse=reuse)
  y = tf.layers.batch_normalization(y, training=is_training, name='%s/batch_norm'%name, reuse=reuse)
  y = tf.nn.leaky_relu(y, 0.0, name='%s/leaky_relu'%name)
  return y

def skip_layer(x, sk, is_training, name, reuse):
  bx, hx, wx, cx = x.shape
  bsk, hsk, wsk, csk = sk.shape

  sk = tf.pow(sk, 2.0)
  sk = tf.log(sk + eps)

  if hx > hsk:
    x = tf.slice(x, [0,0,0,0], [bx, hsk, wx, cx])
  elif hx < hsk:
    sk = tf.slice(sk, [0,0,0,0], [bsk, hx, wsk, csk])

  # Concatenate
  y = tf.concat([x, sk], axis=3)

  # Fuse concatenated layer
  fsize = cx + csk
  W_init = np.zeros((fsize, cx))
  for i in range(cx):
    W_init[i, i] = 1
    W_init[i+csk, i] = 1

  y = tf.layers.conv2d(y, cx, 1, 1, 'same',
                      activation=tf.identity,
                      kernel_initializer=tf.constant_initializer(W_init, dtype=tf.float32),
                      bias_initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                      name='%s/fuse'%name,
                      reuse=reuse)
  return y