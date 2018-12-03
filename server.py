# -*- encoding: utf-8 -*-
# pip install flask

import flask
from flask import request
from PIL import Image
import io
import numpy as np
import time
import re
import os


app = flask.Flask(__name__)


folder = ""



# Configure the flask server
app.config['JSON_SORT_KEYS'] = False


@app.route("/api/cv/create_folder/", methods=['POST'])
def api_create_folder():
    try:
        global folder
        folder = str(int(time.time()))
        if not os.path.exists(folder):
            os.makedirs(r"D:\Users\IMPACT\Documents\py_iReal-master\py_iReal-master\webserver/" + folder)

            print("created folder: " + folder)


            response_list = [("folder", folder)]

            response_dict = dict(response_list)

            return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"



@app.route("/api/cv/save_images/", methods=['POST'])
def api_save_images():
    try:
        for f in request.files:
            img = request.files[f]

            global folder

            start = time.time()
            pimg = Image.open(img.stream)
            #npimg = np.array(pimg)
            filename = r"D:\Users\IMPACT\Documents\py_iReal-master\py_iReal-master\webserver/" + folder + "/" + str(start).replace('.', '_') + "_" + str(f) + ".jpg"
            print("saving  as: " + filename)

            pimg.save(filename);

            response_list = [("file", filename), ("folder", folder)]

            response_dict = dict(response_list)

            #print(response_dict)
            end = time.time()
            print("Processing time: {} ".format(end - start))
            return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"



@app.route("/api/cv/run_reconstruction/", methods=['POST'])
def api_run_reconstruction():
    try:
        global folder
        path = "D:/Users/IMPACT/Documents/py_iReal-master/py_iReal-master/webserver/"
        os.system(path + "COLMAP.bat automatic_reconstructor \ --workspace_path "+ path + folder + " \ --image_path " + path + folder)

        response_list = [("folder", folder)]

        response_dict = dict(response_list)

        #print(response_dict)


        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


@app.route("/api/cv/get_transformation/", methods=['POST'])
def api_get_transformation():
    try:
        global folder
        path = "D:/Users/IMPACT/Documents/py_iReal-master/py_iReal-master/webserver/"
        numOfCams = request.form['numberOfCams']
        print(numOfCams)


        response_list = [("folder", folder)]

        response_dict = dict(response_list)
        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


@app.route("/api/cv/query_reconstruction/", methods=['GET'])
def api_query_reconstruction():
    try:
        global folder

        with open(r"D:\Users\IMPACT\Documents\py_iReal-master\py_iReal-master\webserver/" + folder + "/dense/0/fused.ply", 'rb') as myfile:
            data = myfile.read()

        return bytes(data)

    except Exception as err:

        print("ko:", err)

@app.route("/api/cv/query_cameras/", methods=['GET'])
def api_query_cameras():
    try:
        global folder
        path = "D:/Users/IMPACT/Documents/py_iReal-master/py_iReal-master/webserver/"
        os.system(path + 'COLMAP.bat model_converter --input_path ' + path + folder + '/dense/0/sparse --output_path '+ path + folder + '/dense/0/sparse --output_type TXT')

        with open(r"D:\Users\IMPACT\Documents\py_iReal-master\py_iReal-master\webserver/" + folder + "/dense/0/sparse/images.txt", 'r') as myfile:
            data = myfile.read()

        return data

    except Exception as err:

        print("ko:", err)



from flask import redirect, url_for
@app.route('/')
def index():
    return redirect(url_for('hello_html'))



from flask import abort
@app.route('/broken')
def broken_url():
    abort(401)


if __name__ == "__main__":
    # app.config.update(MAX_CONTENT_LENGTH=100*10**6)
    app.run(port=9099
            ,host='10.35.100.210')
    

