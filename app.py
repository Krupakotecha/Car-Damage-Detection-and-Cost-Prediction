import os
import json
import urllib
import h5py
import pickle as pk
import numpy as np
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators


from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from functools import wraps


import argparse

import cv2
from keras.preprocessing import image


from os.path import join, dirname, realpath
from flask import Flask, request, redirect, url_for, send_from_directory, render_template, flash, Response
from werkzeug.utils import secure_filename

#m> tag is marked with enctype=multipart/form-data and an <input type=file> is placed in that form.
# The application accesses the file from the files dictionary on the request object.
# use the save() method of the file to save the file permanently somewhere on the filesystem.

UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static/uploads/') # where uploaded files are stored
ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF']) # models support png and gif as well

app = Flask(__name__)
app.config['UPLOAD'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 # max upload - 10MB
app.secret_key = 'secret'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'myflaskapp'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)



# check if an extension is valid and that uploads the file and redirects the user to the URL for the uploaded file
def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
	return render_template('home.html', result=None)

@app.route('/index')
def index():
	return render_template('index.html', result=None)


class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    email = StringField('Email', [validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        password = form.password.data

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute("INSERT INTO user(name, email,password) VALUES(%s, %s, %s)", (name,email,password))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('You are now registered and can log in', 'success')

        return redirect(url_for('login'))
    return render_template('register.html', form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        email = request.form['email']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute("SELECT * FROM user WHERE email = %s", [email])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if password_candidate== password:
                # Passed
                
                flash('You are now logged in', 'success')
                return redirect(url_for('index'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))


@app.route('/<a>')
def available(a):
	flash('{} coming soon!'.format(a))
	return render_template('index.html', result=None, scroll='third')

@app.route('/assessment')
def assess():
	return render_template('index.html', result=None, scroll='third')


@app.route('/assessment', methods=['GET', 'POST'])
def upload_and_classify():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(url_for('assess'))
		
		file = request.files['file']

		# if user does not select file, browser also
		# submit a empty part without filename
		if file.filename == '':
			flash('No selected file')
			return redirect(url_for('assess'))

		if file and allowed_file(file.filename):
			filename = file.filename # used to secure a filename before storing it directly on the filesystem
			file.save(os.path.join(app.config['UPLOAD'], filename))
			# return redirect(url_for('uploaded_file',
			#                         filename=filename))
			filepath = os.path.join(app.config['UPLOAD'], filename)
			print(filename)
			model_results =model_predict(filepath) 

			return render_template('results.html', result=model_results, scroll='third', filename=filename)
	
	flash('Invalid file format - please try your upload again.')
	return redirect(url_for('assess'))

# @app.route('/show/<filename>')
# def uploaded_file(filename):
#     return render_template('template.html', filename=filename)
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", default=img_path, help="image for prediction")
    parser.add_argument("--config", default='custom_data/yolov3-tiny_custom_obj.cfg', help="YOLO config path")
    parser.add_argument("--weights", default='custom_data/yolov3-tiny_custom_obj_last.weights', help="YOLO weights path")
    parser.add_argument("--names", default='custom_data/custom.names', help="class names path")
    args = parser.parse_args()

    CONF_THRESH, NMS_THRESH = 0.5, 0.5

    # Load the network
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    img = cv2.imread(args.image)
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

    # Draw the filtered bounding boxes with their class to the image
    with open(args.names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for index in indices:
        x, y, w, h = b_boxes[index]
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[index], 2)
        cv2.putText(img, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors[index], 2)
        preds=classes[class_ids[index]]
    print(preds)    
       
    return preds


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Now one last thing is missing: the serving of the uploaded files. 
# In the upload_file() we redirect the user to url_for('uploaded_file', filename=filename), 
# that is, /uploads/filename. So we write the uploaded_file() function to return the file of that name. 

@app.route('/uploads/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD'],
							   filename)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False) # remember to set back to False	