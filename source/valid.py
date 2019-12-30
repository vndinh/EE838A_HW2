import tensorflow as tf
import time
import numpy as np
import imageio
import random

from hdr_cnn import hdrcnn, get_final_hdr
from config import config
from utils import get_filepath, write_logs, loss_fn, img_write, alpha_msk, norm_img, reinhard02

# Directories
model_dir = config.TRAIN.model_dir

# Parameters
lambda_ir = config.TRAIN.lambda_ir
gamma = config.VALID.gamma

def read_valid_images(ldr_str, hdr_str):
  ldr_img = imageio.imread(ldr_str, 'PNG-FI')
  hdr_img = imageio.imread(hdr_str, 'HDR-FI')
  return ldr_img, hdr_img

def valid_parse(ldr_dir, hdr_dir):
  ldr_string = tf.read_file(ldr_dir)
  hdr_string = tf.read_file(hdr_dir)
  ldr_img, hdr_img = tf.py_func(read_valid_images, [ldr_string, hdr_string], [tf.uint8, tf.float32])
  _, hdr_img, Hth = norm_img(hdr_img)
  ldr_img = tf.image.convert_image_dtype(ldr_img, tf.float32)
  return ldr_img, hdr_img, Hth

def validate(ldr_dir, hdr_dir, gen_dir, logs_dir, img_height, img_width):
  X = tf.placeholder(tf.float32, [1, img_height , img_width, 3])
  Y = tf.placeholder(tf.float32, [1, img_height, img_width, 3])
  
  valid_ldr_path, _ = get_filepath(ldr_dir, '.png')
  valid_hdr_path, valid_hdr_name = get_filepath(hdr_dir, '.hdr')
  num_valid = len(valid_hdr_path)
  
  # Data loader
  dataset = tf.data.Dataset.from_tensor_slices((valid_ldr_path, valid_hdr_path))
  dataset = dataset.map(valid_parse, num_parallel_calls=4)
  dataset = dataset.batch(1)
  iter = dataset.make_one_shot_iterator()
  ldr_img, hdr_img, Hth = iter.get_next()

  alpha = alpha_msk(X)

  # Prediction
  with tf.name_scope('HDR_CNN'):
    hdr_nn = hdrcnn(X, is_training=False, reuse=False)
  hdr_final = get_final_hdr(X, hdr_nn)

  # Loss functions
  with tf.name_scope('Loss'):
    irloss, dirloss = loss_fn(X, hdr_nn, Y)

  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    # Restore weights of model
    saver.restore(sess, model_dir)
    
    # Validation
    log = "\n========== Validation Begin ==========\n"
    write_logs(logs_dir, log, True)
    valid_start = time.time()
    avg_irloss = 0
    avg_dirloss = 0
    for f in valid_hdr_name:
      valid_img_start = time.time()
      ldr_image, hdr_image, Hth_val = sess.run([ldr_img, hdr_img, Hth])
      alpha_val, hdr_pred, irloss_val, dirloss_val = sess.run([alpha, hdr_final, irloss, dirloss], feed_dict={X:ldr_image, Y:hdr_image})
      avg_irloss += irloss_val
      avg_dirloss += dirloss_val

      f1, _ = f.split("_")
      img_write(gen_dir, 'alpha_'+f1+'_HDR.png', alpha_val, 'PNG-FI')

      # Gamma correction
      hdr_pred_save = np.multiply(Hth_val, np.maximum(hdr_pred, 0.0))
      img_write(gen_dir, 'pred_'+f, hdr_pred_save, 'HDR-FI')

      # Tone mapping
      hdr_pred_gamma = np.power(np.maximum(hdr_pred, 0.0), gamma)
      ldr_tone = reinhard02(hdr_pred_gamma, a=0.18)
      img_write(gen_dir, 'tm_'+f1+'_HDR.png', ldr_tone, 'PNG-FI')
    
      log = "Image {}, Time {:2.5f}, Shape = {}, I/R Loss = {:2.5f}, Direct Loss = {:2.5f}".format(f, time.time()-valid_img_start, hdr_pred.shape, irloss_val, dirloss_val)
      write_logs(logs_dir, log, False)
    log = "\nAverage I/R Loss = {:2.5f}, Average Direct Loss = {:2.5f}".format(avg_irloss/num_valid, avg_dirloss/num_valid)
    write_logs(logs_dir, log, False)
    log = "\nValidation Time: {:2.5f}".format(time.time()-valid_start)
    write_logs(logs_dir, log, False)
    log = "\n========== Validation End ==========\n"
    write_logs(logs_dir, log, False)
    
    sess.close()
    
