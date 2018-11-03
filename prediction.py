import numpy as np
import os.path
import scipy.misc
import shutil
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from moviepy.editor import VideoFileClip
import helper


RUN_VIDEO = True

def pipeline(img):
    img = scipy.misc.imresize(img, image_shape)

    img_norm = apply_gaussian(img)

    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [img_norm]})
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(img)
    street_im.paste(mask, box=None, mask=mask)

    return scipy.misc.fromimage(street_im)

# get session and tensors loaded
sess = tf.Session()
saver = tf.train.import_meta_graph('model/bartlebooth-fcn.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model'))
print('model restored')

image_pl = tf.get_default_graph().get_tensor_by_name("image_input:0")
logits = tf.get_default_graph().get_tensor_by_name("logits:0")
keep_prob = tf.get_default_graph().get_tensor_by_name("keep_prob:0")
image_shape = (160, 576)

# this is single image pipeline

if RUN_VIDEO:
    output = 'windy_road_output.mp4'
    clip = VideoFileClip("windy_road.mp4").subclip(5,7)
    clip = clip.fl_image(pipeline)

    # write to file
    clip.write_videofile(output)
else:
    pred_img = "test.png"
    img = scipy.misc.imresize(scipy.misc.imread(pred_img), image_shape)
    scipy.misc.imsave('prediction.png', pipeline(img))
