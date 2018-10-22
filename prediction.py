import numpy as np
import os.path
import scipy.misc
import shutil
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# get the model and work with it
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model/bartlebooth-fcn.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./model/'))
    print('model restored')

    latest_ckp = tf.train.latest_checkpoint('./model/')
    print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
    input = tf.get_default_graph().get_tensor_by_name("image_input:0")
    output = tf.get_default_graph().get_tensor_by_name("final_layer/kernel/Adam_1:0")

#    pred_image = './data/'
 #   image_shape = (160, 576)
  #  img = scipy.misc.imresize(scipy.misc.imread(pred_img), image_shape)
   # img_norm = normalise(img)

#    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [img_norm]})
 #   im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
  #  segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
   # mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
#    mask = scipy.misc.toimage(mask, mode="RGBA")
 #   street_im = scipy.misc.toimage(img)
  #  street_im.paste(mask, box=None, mask=mask)

   # scipy.misc.imsave('prediction.png', street_im)

