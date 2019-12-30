import tensorflow as tf
import time
import numpy as np
import imageio
import random

from config import config
from hdr_cnn import hdrcnn
from utils import get_filepath, write_logs, loss_fn, norm_img

# Directories
model_dir = config.TRAIN.model_dir
logs_dir = config.TRAIN.logs_dir
logs_train = config.TRAIN.logs_train
train_hdr_dir = config.TRAIN.hdr_dir

# Hyper Parameters
num_epoches = config.TRAIN.num_epoches
patch_size = config.TRAIN.patch_size
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
lr_decay = config.TRAIN.lr_decay
lr_decay_period = config.TRAIN.lr_decay_period

# Parameters
eps = config.TRAIN.eps
lambda_ir = config.TRAIN.lambda_ir

def read_train_images(img_str):
  img = imageio.imread(img_str, 'HDR-FI')

  R = img[:,:,0]
  G = img[:,:,1]
  B = img[:,:,2]

  R = R[~np.all(R==0, axis=1)]
  G = G[~np.all(G==0, axis=1)]
  B = B[~np.all(B==0, axis=1)]

  R = np.expand_dims(R, axis=2)
  G = np.expand_dims(G, axis=2)
  B = np.expand_dims(B, axis=2)

  img = np.concatenate((R, G, B), axis=2)

  img_h, img_w, _ = img.shape

  return img, img_h, img_w

def train_parse(hdr_dir):
  hdr_string = tf.read_file(hdr_dir)

  hdr_img, hdr_height, hdr_width = tf.py_func(read_train_images, [hdr_string], [tf.float32, tf.int32, tf.int32])
  hdr_img = tf.image.random_flip_up_down(hdr_img)
  hdr_img = tf.image.random_flip_left_right(hdr_img)

  hdr_height = tf.cast(hdr_height, tf.float32)
  hdr_width = tf.cast(hdr_width, tf.float32)

  ratio = random.uniform(0.2, 0.6)
  crop_size = tf.minimum(tf.sqrt(hdr_height*hdr_width*ratio), hdr_height)
  crop_size = tf.cast(crop_size, tf.int32)

  hdr_crop = tf.random_crop(hdr_img, [crop_size, crop_size, 3])
  hdr_patch = tf.image.resize_images(hdr_crop, [patch_size, patch_size])

  ldr_patch, hdr_patch, _ = norm_img(hdr_patch)

  return ldr_patch, hdr_patch

def training():
  x = tf.placeholder(tf.float32, [None, patch_size, patch_size, 3], name='ldr_input')
  y = tf.placeholder(tf.float32, [None, patch_size, patch_size, 3], name='hdr_target')

  # Data loader
  train_hdr_path, _ = get_filepath(train_hdr_dir, '.hdr')
  num_train = len(train_hdr_path)
  dataset = tf.data.Dataset.from_tensor_slices(train_hdr_path)
  dataset = dataset.shuffle(num_train)
  dataset = dataset.map(train_parse, num_parallel_calls=8)
  dataset = dataset.batch(batch_size)
  iter = dataset.make_initializable_iterator()
  ldr_patch, hdr_patch = iter.get_next()

  # Model
  with tf.name_scope('HDR_CNN'):
    hdr_nn = hdrcnn(x, is_training=True, reuse=False)
  
  # Loss functions
  with tf.name_scope('Loss'):
    irloss, dirloss = loss_fn(x, hdr_nn, y)
  tf.summary.scalar("irloss", irloss)
  tf.summary.scalar("dirloss", dirloss)

  with tf.variable_scope('learning_rate'):
    lr_v = tf.Variable(lr_init, trainable=False)
  
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_ir = tf.train.AdamOptimizer(lr_v).minimize(irloss)
    #train_dir = tf.train.AdamOptimizer(lr_v).minimize(dirloss)
    #train_op = tf.group(train_ir, train_dir)

  saver = tf.train.Saver()

  merged_sum_op = tf.summary.merge_all()

  if num_train % batch_size != 0:
    num_batches = int(num_train/batch_size) + 1
  else:
    num_batches = int(num_train/batch_size)

  with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Op to write logs to Tensorboard
    train_sum_writer = tf.summary.FileWriter(logs_dir, tf.get_default_graph())

    # Training process
    log = "\n========== Training Begin ==========\n"
    write_logs(logs_train, log, True)
    train_start = time.time()
    for epoch in range(num_epoches):
      epoch_start = time.time()    
      if (epoch > 300) and (epoch % lr_decay_period == 0):
        new_lr = lr_v * lr_decay
        sess.run(tf.assign(lr_v, new_lr))
        log = "** New learning rate: %f **\n" % (lr_v.eval())
        write_logs(logs_train, log, False)
      elif epoch == 0:
        sess.run(tf.assign(lr_v, lr_init))
        log = "** Initial learning rate: %f **\n" % (lr_init)
        write_logs(logs_train, log, False)
      
      avg_irloss = 0
      avg_dirloss = 0
      sess.run(iter.initializer)

      for batch in range(num_batches):
        batch_start = time.time()
        ldr_patches, hdr_patches = sess.run([ldr_patch, hdr_patch])
        _, irloss_val, dirloss_val, summary = sess.run([train_ir, irloss, dirloss, merged_sum_op], feed_dict={x:ldr_patches, y:hdr_patches})
        avg_irloss += irloss_val
        avg_dirloss += dirloss_val
        train_sum_writer.add_summary(summary, epoch*num_batches+batch)

        log = "Epoch {}, Time {:2.5f}, Batch {}, Batch I/R Loss = {:2.5f}, Batch Direct Loss = {:2.5f}".format(epoch, time.time()-batch_start, batch, irloss_val, dirloss_val)
        write_logs(logs_train, log, False)
      log = "\nEpoch {}, Time {:2.5f}, Average I/R Loss = {:2.5f}, Average Direct Loss = {:2.5f}\n".format(epoch, time.time()-epoch_start, avg_irloss/num_batches, avg_dirloss/num_batches)
      write_logs(logs_train, log, False)

    log = "\nTraining Time: {}".format(time.time()-train_start)
    write_logs(logs_train, log, False)
    log = "\n========== Training End ==========\n"
    write_logs(logs_train, log, False)

    # Save model
    save_path = saver.save(sess, model_dir)
    log = "Model is saved in file: %s" % save_path
    write_logs(logs_train, log, False)
    log = "Run the command line:\n" \
          "--> tensorboard --logdir=../logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser"
    write_logs(logs_train, log, False)
    sess.close()

