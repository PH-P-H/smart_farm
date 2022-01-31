# clone from https://github.com/AIZOOTech/flask-object-detection
from flask import Flask, request, render_template, url_for, flash, redirect, send_from_directory
from PIL import Image
import numpy as np
import base64
import io
import os
import time
from shutil import copyfile

import cv2
from wtforms import form

from forms import UploadImageForm, UploadImagesForm, UploadVideoForm
from night_detection import night_detect, load_image_into_numpy_array
from utils import send_alert_email
from day_detection import *
from werkzeug.utils import secure_filename
import object_tracker


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

app = Flask(__name__)

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


def save_picture(form_picture, name="", folder='static/images'):
    _, f_ext = os.path.splitext(form_picture.filename)
    picture_fn = name + '.jpg'
    picture_path = os.path.join(folder, picture_fn)

    # output_size = (600, 400)
    i = Image.open(form_picture)
    # i.thumbnail(output_size) # don't know why make the image unpredictable
    if f_ext == '.png':
        i = i.convert('RGB')
    i.save(picture_path)

    return picture_fn


@app.route("/night_detection", methods=['GET', 'POST'])
def night_detection():
    form = UploadImageForm()
    if form.validate_on_submit():
        if form.picture.data:
            save_picture(form.picture.data, 'night_input')
            image = Image.open('static/images/night_input.jpg')
            if(image.mode!='RGB'):
                image = image.convert("RGB")
            img_arr = np.array(image)
            night_detect(img_arr)
            send_alert_email("huang.peng@sjsu.edu") 
        return redirect(url_for('night_detection'))
    elif request.method == 'GET':
        summary = ""
        image_file = url_for('static', filename='images/night_input.jpg')
    image_file = url_for('static', filename='images/night_input.jpg')

    return render_template('night_detection.html', title='Night_detection',
                           image_file=image_file, form=form, summary=summary)


@app.route("/tracking")
def tracking():
    form = UploadVideoForm()
    if form.validate_on_submit():   
        filename = secure_filename(form.file.data.filename)
        input_file = 'input/' + filename
        output_file = 'output/' + filename
        print(f"Saving input/{filename} ...")    
        form.file.data.save('input/' + filename)
        print(f"Start object tracking ...")
        object_tracker.object_tracker(input_file, output_file)
        print(f"Finish object tracking ...")
        return redirect(url_for('tracking'))
    return render_template('tracking.html', form=form)


@app.route("/daytime")
def daytime():
    return render_template('daytime.html')


@app.route("/livestock_detection", methods=['GET', 'POST'])
def livestock_detection():
    form = UploadImageForm()
    if form.validate_on_submit():
        if form.picture.data: 
            dir = 'files/object_detection/image'
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))

            print('start livestock detection')

            target = os.path.join(APP_ROOT, 'files')
            if not os.path.isdir(target):
                os.mkdir(target)

            # load faster-rcnn models 
            livestock_path = os.path.join(APP_ROOT,'models/modeling_livestock/faster_rcnn_model/saved_model')
            saved_models = load_livestock_model(livestock_path)

            # load the class name mapping file 
            class_name_mapping = load_json_file(os.path.join(APP_ROOT,'ref_files/class_name_mapping.json'))
            label_path = os.path.join(APP_ROOT,'ref_files/labelmap.pbtxt') 

            filename = save_picture(form.picture.data, name='livestock_det', folder='files/object_detection/image') # ph     
            destination = os.path.join(target, 'object_detection', 'image', filename)

            # do object detection 
            upload_dir = os.path.join(target, 'object_detection','image')
            pred_dir = os.path.join(APP_ROOT,'darknet/data/test_after_training')

            # do inference faster_rcnn 
            faster_rcnn_results = get_model_results_faster_rcnn(destination,saved_models,filename)
        
            # save the visualization of the detectected results 
            out_file_location = os.path.join(APP_ROOT,'files/object_detection/result')

            # generate the visualized change detection results         
            visualize_detection_results(faster_rcnn_results,out_file_location,label_path,upload_dir,class_name_mapping,filename,det_module='livestock')

            # # get the object counts 
            file_location = os.path.join(APP_ROOT,'files/object_detection/result')
            out_file_location = os.path.join(APP_ROOT,'files/object_detection/result_counts')

            visualize_results_counts(faster_rcnn_results,file_location,out_file_location,object_det_module='livestock')  

            copyfile('files/object_detection/result_counts/livestock_det.jpg', 'static/object_detection/result_counts/livestock_det.jpg')
            copyfile('files/object_detection/image/livestock_det.jpg', 'static/object_detection/image/livestock_det.jpg')

        return redirect(url_for('livestock_detection'))
    elif request.method == 'GET':
        image_file = url_for('static', filename='object_detection/image/livestock_det.jpg')
    image_file = url_for('static', filename='object_detection/image/livestock_det.jpg')
    
    return render_template('livestock_detection.html', title='Livestock_detection',
                           image_file=image_file, form=form)


@app.route("/static_detection", methods=['GET', 'POST'])
def static_detection():
    form = UploadImageForm()
    if form.validate_on_submit():
        if form.picture.data: # ph
            dir = 'files/object_detection/image'
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
                
            print('start object detection')
            # load faster-rcnn models 
            vehicle_path = os.path.join(APP_ROOT,'models/modeling_vehicle/faster_rcnn_model/saved_model')
            tree_path = os.path.join(APP_ROOT,'models/modeling_tree/faster_rcnn_model/saved_model')
            building_path = os.path.join(APP_ROOT,'models/modeling_building/faster_rcnn_model/saved_model')
            saved_models = load_faster_rcnn_models(vehicle_path,tree_path,building_path)

            # load the class name mapping file 
            class_name_mapping = load_json_file(os.path.join(APP_ROOT,'ref_files/class_name_mapping.json'))
            label_path = os.path.join(APP_ROOT,'ref_files/labelmap.pbtxt') 
            filename = save_picture(form.picture.data, name='static_det', folder='files/object_detection/image') 
            destination = os.path.join(APP_ROOT, f'files/object_detection/image/{filename}')

            # do object detection 
            upload_dir = os.path.join(APP_ROOT, 'files/object_detection/image') # ph
            pred_dir = os.path.join(APP_ROOT, 'darknet/data/test_after_training')     

            # do inference yolov4 
            yolo_results = run_yolo_inference(upload_dir, pred_dir)
            # do inference faster_rcnn 
            faster_rcnn_results = get_model_results_faster_rcnn(destination,saved_models,filename)
            # combine yolov4 and faster_rcnn results together 
            combined_results = combine_results(yolo_results,faster_rcnn_results)
            # ensemble faster_rcnn and yolov4 results together 
            ensembled_df = ensemble_results(combined_results,class_name_mapping)

            # save the visualization of the detectected results 
            out_file_location = os.path.join(APP_ROOT,'files/object_detection/result')
            # generate the visualized change detection results      
            visualize_detection_results(ensembled_df,out_file_location,label_path,upload_dir,class_name_mapping,filename,det_module='combined')

            # get the object counts 
            file_location = os.path.join(APP_ROOT,'files/object_detection/result')
            out_file_location = os.path.join(APP_ROOT,'files/object_detection/result_counts') 
            visualize_results_counts(ensembled_df,file_location,out_file_location,object_det_module='combined')

            copyfile('files/object_detection/image/static_det.jpg', 'static/object_detection/image/static_det.jpg')
            copyfile('files/object_detection/result_counts/static_det.jpg', 'static/object_detection/result_counts/static_det.jpg')
        return redirect(url_for('static_detection'))
    elif request.method == 'GET':
        image_file = url_for('static', filename='object_detection/image/static_det.jpg')
        # image_file = os.path.join(APP_ROOT,'files/object_detection/image/static_det.jpg')
    image_file = url_for('static', filename='object_detection/image/static_det.jpg')
    # image_file = os.path.join(APP_ROOT,'files/object_detection/image/static_det.jpg')
    
    return render_template('static_detection.html', title='Static_detection',
                           image_file=image_file, form=form)

@app.route("/change")
def change():
    return render_template('change.html')


@app.route("/livestock_change", methods=['GET', 'POST'])
def livestock_change():
    form = UploadImagesForm()
    # form2 = UploadImage2Form()
    if form.validate_on_submit():
        if form.picture1.data and form.picture2.data:
            print('start livestock change detection')

            target = os.path.join(APP_ROOT, 'files')
            if not os.path.isdir(target):
                os.mkdir(target)

            # load faster-rcnn models 
            livestock_path = os.path.join(APP_ROOT,'models/modeling_livestock/faster_rcnn_model/saved_model')
            saved_models = load_livestock_model(livestock_path)

            # load the class name mapping file 
            class_name_mapping = load_json_file(os.path.join(APP_ROOT,'ref_files/class_name_mapping.json'))
            label_path = os.path.join(APP_ROOT,'ref_files/labelmap.pbtxt') 

            results_dfs = [] 
            image_locations = []
            filenames = []

            filenames.append(save_picture(form.picture1.data, name='image_1', folder='files/change_detection/image_1')) 
            filenames.append(save_picture(form.picture2.data, name='image_2', folder='files/change_detection/image_2')) 

            for filename in filenames:
                image_dir = filename[:-4]
                destination = os.path.join(target, 'change_detection', image_dir, filename)
                
                # do object detection 
                upload_dir = os.path.join(target, 'change_detection', image_dir)
                pred_dir = os.path.join(APP_ROOT, 'darknet/data/test_after_training')

                # do inference faster_rcnn 
                faster_rcnn_results = get_model_results_faster_rcnn(destination,saved_models,filename)
                results_dfs.append(faster_rcnn_results)

                # save the visualization of the detectected results 
                out_file_location = os.path.join(APP_ROOT,'files/change_detection/result',image_dir)
                image_locations.append(out_file_location)

                # add something to function to remove the file if there is something there already 
                visualize_detection_results(faster_rcnn_results,out_file_location,label_path,upload_dir,class_name_mapping,filename,det_module='livestock')

            # now do the change detection part 
            results_df_all = pd.concat(results_dfs)
            
            image_1 = filenames[0]
            image_2 = filenames[1]
            changes_detected = get_changes_detected(results_df_all,image_1,image_2,det_module='livestock')
            out_file_location = os.path.join(APP_ROOT,'files/change_detection/result')
        
            # generate the change detection results and save to file location 
            visualize_object_count_changes(changes_detected,image_1,image_2,out_file_location,det_module='livestock')

            copyfile(f'{out_file_location}/change_detection/result.png', 'static/change_detection/result/change_detection/result.png')

        return redirect(url_for('livestock_change'))
    elif request.method == 'GET':
        image_file = url_for('static', filename='change_detection/result/change_detection/result.png')
        # image_file = os.path.join(APP_ROOT,'files/object_detection/image/static_det.jpg')
    image_file = url_for('static', filename='change_detection/result/change_detection/result.png')
    # image_file = os.path.join(APP_ROOT,'files/object_detection/image/static_det.jpg')
    
    return render_template('livestock_change.html', title='Livestock_change',
                           image_file=image_file, form=form)


@app.route("/static_change", methods=['GET', 'POST'])
def static_change():
    form = UploadImagesForm()
    if form.validate_on_submit():
        if form.picture1.data and form.picture2.data:
            print('start static object change detection')

            target = os.path.join(APP_ROOT, 'files')
            if not os.path.isdir(target):
                os.mkdir(target)

            # load faster-rcnn models 
            vehicle_path = os.path.join(APP_ROOT,'models/modeling_vehicle/faster_rcnn_model/saved_model')
            tree_path = os.path.join(APP_ROOT,'models/modeling_tree/faster_rcnn_model/saved_model')
            building_path = os.path.join(APP_ROOT,'models/modeling_building/faster_rcnn_model/saved_model')
            saved_models = load_faster_rcnn_models(vehicle_path,tree_path,building_path)

            # load the class name mapping file 
            class_name_mapping = load_json_file(os.path.join(APP_ROOT,'ref_files/class_name_mapping.json'))
            label_path = os.path.join(APP_ROOT,'ref_files/labelmap.pbtxt') 

            ensembled_dfs = [] 
            image_locations = []
            filenames = []

            filenames.append(save_picture(form.picture1.data, name='image_1', folder='files/change_detection/image_1')) 
            filenames.append(save_picture(form.picture2.data, name='image_2', folder='files/change_detection/image_2')) 

            for filename in filenames:
                image_dir = filename[:-4]
                destination = os.path.join(target,'change_detection' ,image_dir ,filename)
                
                # do object detection 
                upload_dir = os.path.join(target,'change_detection' ,image_dir)
                pred_dir = os.path.join(APP_ROOT,'darknet/data/test_after_training')


                print('\n\n\n', upload_dir)


                # do inference yolov4 
                yolo_results = run_yolo_inference(upload_dir, pred_dir)

                print('\n\n\n', yolo_results)
                # do inference faster_rcnn 
                faster_rcnn_results = get_model_results_faster_rcnn(destination,saved_models,filename)
                
                print('\n\n\n', faster_rcnn_results)
                # combine yolov4 and faster_rcnn results together 
                combined_results = combine_results(yolo_results,faster_rcnn_results)
                # ensemble faster_rcnn and yolov4 results together 
                ensembled_df = ensemble_results(combined_results,class_name_mapping)
                ensembled_dfs.append(ensembled_df)

                # save the visualization of the detectected results 
                out_file_location = os.path.join(APP_ROOT,'files/change_detection/result',image_dir)
                image_locations.append(out_file_location)

                # add something to function to remove the file if there is something there already 
                visualize_detection_results(ensembled_df,out_file_location,label_path,upload_dir,class_name_mapping,filename,det_module='combined')

            # now do the change detection part 
            ensembled_data = pd.concat(ensembled_dfs)
            
            image_1 = filenames[0]
            image_2 = filenames[1]
            changes_detected = get_changes_detected(ensembled_data,image_1,image_2,det_module='combined')
            out_file_location = os.path.join(APP_ROOT,'files/change_detection/result')

            # generate the change detection results and save to file location 
            visualize_object_count_changes(changes_detected,image_1,image_2,out_file_location,det_module='combined')

            copyfile(f'{out_file_location}/change_detection/result.png', 'static/change_detection/result/change_detection/static_result.png')

        return redirect(url_for('static_change'))
    elif request.method == 'GET':
        image_file = url_for('static', filename='change_detection/result/change_detection/static_result.png')
        # image_file = os.path.join(APP_ROOT,'files/object_detection/image/static_det.jpg')
    image_file = url_for('static', filename='change_detection/result/change_detection/static_result.png')
    # image_file = os.path.join(APP_ROOT,'files/object_detection/image/static_det.jpg')
    
    return render_template('static_change.html', title='Static_Change',
                           image_file=image_file, form=form)

if __name__ == '__main__':
    app.run(debug=True)
