import os
import smtplib

from email.message import EmailMessage
from email.mime.text import MIMEText
from email.utils import formataddr
from smtplib import SMTP_SSL
import imghdr

import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt


from static.configs.config import id2name, animalid2name
from static.configs.thermal_animal_config import category_index
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
# from backend.config import animalid2name

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def send_alert_email(email_receiver):
    EMAIL_ADDRESS = os.environ.get('EMAIL_USER')
    EMAIL_PASSWORD = os.environ.get('EMAIL_PW')

    msg = EmailMessage()
    msg['Subject'] = "Farmland intruder alert"
    msg['From'] = [EMAIL_ADDRESS]
    msg['Reply-To'] = formataddr(("Huang", "huang.peng@sjsu.edu"))
    msg['To'] = [email_receiver]

    main_content = "Suspicious human/animal found on your farmland!\n\n"

    html_content = "Suspicious human/animal found on your farmland!"

    msg.set_content(main_content)
    msg.add_alternative("""\
    <html>
    <head>{part1}</head>
    <body>
        <p></p>
    </body>
    </html>
    """.format(part1=html_content), subtype='html')

    # image = MIMEImage(img_data, name=os.path.basename('night_detection_result.jpg'))
    # msg.attach(image)

    with open('static/images/night_result.jpg', 'rb') as fp:
        img_data = fp.read()
    msg.add_attachment(img_data, maintype='image',
                                 subtype=imghdr.what(None, img_data))

    with SMTP_SSL('smtp.gmail.com', 465) as server:
        # server.set_debuglevel(1)
        # server.starttls(context=context())
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)


# Don't need this function anymore
def save_image(image, graph):
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'models/labelmap.pbtxt'

    NUM_CLASSES = 15

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'), # make color strange
        use_normalized_coordinates=True,
        line_thickness=8)
    
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/images/night_result.jpg', image)
    # plt.savefig('foo.png')
    # img = Image.fromarray(array)
    # img.save("filename.jpeg")
    
