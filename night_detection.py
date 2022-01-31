import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from static.configs.config import id2name, animalid2name
from static.configs.thermal_animal_config import category_index

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

from utils import save_image
# from object_detection.utils import visualization_utils as vis_util

PATH_TO_HUMAN_CKPT = 'models/frozen_inference_graph_human.pb'
THERMAL_ANIMAL_MODEL_PATH = 'models/night_animal'
PATH_TO_ANIMAL_CKPT = 'models/frozen_inference_graph_animal.pb'

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def load_model(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.compat.v1.Session(graph=detection_graph)
                return sess, detection_graph


def night_detect(img_arr, conf_thresh=0.55):
    # Human intruder detection
    print('human')
    sess_human, human_detection_graph = load_model(PATH_TO_HUMAN_CKPT)
    
    image_tensor = human_detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = human_detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = human_detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = human_detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = human_detection_graph.get_tensor_by_name('num_detections:0')
    image_np_expanded = np.expand_dims(img_arr, axis=0)
    (boxes, scores, classes, num) = sess_human.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

    height, width, _ = img_arr.shape
    results = []
    for idx, class_id in enumerate(classes[0]):
        conf = scores[0, idx]
        if conf > (conf_thresh+0.2):
            bbox = boxes[0, idx]
            ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
            results.append({"name": id2name[class_id],
                            "conf": str(conf),
                            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })
        
    # Thermal animal detection
    print('thermal animal')
    model = tf.saved_model.load(THERMAL_ANIMAL_MODEL_PATH)

    input_tensor = tf.convert_to_tensor(img_arr)
    input_tensor = input_tensor[tf.newaxis,...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    for idx, class_id in enumerate(output_dict['detection_classes']):
        conf = output_dict['detection_scores'][idx]
        if conf > conf_thresh:
            bbox = output_dict['detection_boxes'][idx]
            ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
            results.append({"name": category_index[class_id],
                            "conf": str(conf),
                            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })
    
    # Animal detection
    print('animal')
    sess_animal, animal_detection_graph = load_model(PATH_TO_ANIMAL_CKPT)
    
    image_tensor = animal_detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = animal_detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = animal_detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = animal_detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = animal_detection_graph.get_tensor_by_name('num_detections:0')
    image_np_expanded = np.expand_dims(img_arr, axis=0)
    (boxes, scores, classes, num) = sess_animal.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

    height, width, _ = img_arr.shape
    for idx, class_id in enumerate(classes[0]):
        conf = scores[0, idx]
        if conf > (conf_thresh+0.1):
            bbox = boxes[0, idx]
            ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
            results.append({"name": animalid2name[class_id],
                            "conf": str(conf),
                            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)]
            })

    # save_image(img_arr, animal_detection_graph)
    
    print(results)
    # save result
    for i in results:
        x1,y1,x2,y2 = i.get('bbox')[0], i.get('bbox')[1], i.get('bbox')[2], i.get('bbox')[3]
        # box text and bar
        label = i.get('name') + ": " + i.get('conf')[2:4] + "%"
        cv2.rectangle(img_arr, (x1, y1), (x2,y2), (255, 0, 0), 3)
        cv2.putText(img_arr, label, (x1,y1-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    img = Image.fromarray(img_arr)
    img.save('static/images/night_result.jpg')

    return {"results":results}