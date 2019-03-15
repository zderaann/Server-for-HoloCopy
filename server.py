# -*- encoding: utf-8 -*-
# pip install flask

import flask
from flask import request
from PIL import Image
import numpy as np
import time
import os


app = flask.Flask(__name__)


folder = ""



# Configure the flask server
app.config['JSON_SORT_KEYS'] = False


def calculate_transform(holCams, colCams):
    print("Calculating transform")
    transformation = {}

    numOfCams = len(holCams)
    mHol = np.array([0, 0, 0])
    mCol = np.array([0, 0, 0])
    for i in range(numOfCams):
        mHol = mHol + holCams[i]
        mCol = mCol + colCams[i]

    mHol = mHol / numOfCams  #a.mean(axis=1)
    mCol = mCol / numOfCams
    Q = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    allCol = np.zeros((3, numOfCams))
    allHol = np.zeros((3, numOfCams))


    for i in range(numOfCams):
        holCams[i] = holCams[i] - mHol
        colCams[i] = colCams[i] - mCol
        x = Q * colCams[i]
        colCams[i] = x.diagonal()
        allCol[0, i] = colCams[i][0]
        allCol[1, i] = colCams[i][1]
        allCol[2, i] = colCams[i][2]

        allHol[0, i] = holCams[i][0]
        allHol[1, i] = holCams[i][1]
        allHol[2, i] = holCams[i][2]

    holNorm  = np.linalg.norm(allHol)
    colNorm = np.linalg.norm(allCol)

    scale = holNorm / colNorm
    transformation['scale'] = scale

    for i in range(numOfCams):
        colCams[i] = scale * colCams[i]


    allCol = scale * allCol.transpose()

    allHol = allHol.transpose()

    (n,m) = allHol.shape
    (ny,my) = allCol.shape


    muX = allHol.transpose().mean(axis=1)
    muY = allCol.transpose().mean(axis=1)
    allHol0 = allHol - np.tile(muX, (n, 1))


    allCol0 = allCol - np.tile(muY, (n, 1))


    sqHol = np.zeros(allHol0.shape)
    np.square(allHol0, out = sqHol)

    ssqX = np.sum(sqHol.transpose(), 1)

    sqCol = np.zeros(allCol0.shape)
    np.square(allCol0, out = sqCol)


    ssqY = np.sum(sqCol.transpose(), 1)


    constX = np.all(ssqX <= abs((np.finfo(type(allHol[0, 0])).eps * n * muX)) ** 2)
    constY = np.all(ssqY <= abs((np.finfo(type(allCol[0, 0])).eps * n * muY)) ** 2)

    ssqX = np.sum(ssqX)

    ssqY = np.sum(ssqY)



    if not constX and not constY:
        normX = np.sqrt(ssqX)
        normY = np.sqrt(ssqY)

        allHol0 = allHol0 / normX
        allCol0 = allCol0 / normY


        A = np.matmul(allHol0.transpose(), allCol0)
        U, S, V = np.linalg.svd(A)
        R = np.matmul(V, np.transpose(U))

        print("----SHAPES----")
        print(U.shape)
        print(V.shape)
        print(R.shape)


        traceTA = np.sum(S)
        b = 1
        d = 1 + ssqY / ssqX - 2 * traceTA * normY / normX
        #Z = normY*Y0 * T + repmat(muX, n, 1);
        c = muX - b * np.matmul(muY,R);
        #transform = struct('T', R, 'b', b, 'c', repmat(c, n, 1));
        transformation['rotation1'] = R[0].tolist()
        transformation['rotation2'] = R[1].tolist()
        transformation['rotation3'] = R[2].tolist()
        transformation['translation'] = c.tolist()


        print("----TOLIST----")
        print(c)
        print(R)

    # The degenerate cases: X all the same, and Y all the same.
    elif constX:
        d = 0
        #Z = repmat(muX, n, 1);
        R = np.eye(my, m)
        transformation['rotation1'] = R[0].tolist()
        transformation['rotation2'] = R[1].tolist()
        transformation['rotation3'] = R[2].tolist()
        transformation['translation'] = muX.tolist()

        print("----TOLIST----")
        print(muX)
        print(R)


    #!constX & constY
    else:
        d = 1
        R = np.eye(my, m)
        transformation['rotation1'] = R[0].tolist()
        transformation['rotation2'] = R[1].tolist()
        transformation['rotation3'] = R[2].tolist()
        transformation['translation'] = muX.tolist()
        
        print("----TOLIST----")
        print(muX)
        print(R)


    return transformation


@app.route("/api/cv/create_folder/", methods=['POST'])
def api_create_folder():
    try:
        global folder
        folder = str(int(time.time()))
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        if not os.path.exists(folder):
            os.makedirs(path + folder)

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
            path = os.path.dirname(os.path.realpath(__file__)) + "/"
            #npimg = np.array(pimg)
            filename = path + folder + "/" + str(start).replace('.', '_') + "_" + str(f) + ".jpg"
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
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        os.system(path + "COLMAP.bat automatic_reconstructor \ --workspace_path " + path + folder + " \ --image_path " + path + folder + " --quality medium")

        response_list = [("folder", folder)]

        response_dict = dict(response_list)


        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


@app.route("/api/cv/get_transformation/", methods=['POST'])
def api_get_transformation():
    try:
        global folder
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        os.system(path + 'COLMAP.bat model_converter --input_path ' + path + folder + '/dense/0/sparse --output_path ' + path + folder + '/dense/0/sparse --output_type TXT')
        print()
        numOfCams = request.json['numberOfCams']
        #print()
        #print(request.json)

        holCams = {}
        #cams = request.json['cameras']

        for i in range(0,numOfCams):
            holCams[i] = np.array([float(request.json['cameras'][i]['position']['x']), float(request.json['cameras'][i]['position']['y']), float(request.json['cameras'][i]['position']['z'])])
            #print(holCams[i])

        with open(path + folder + "/dense/0/sparse/images.txt", 'r') as file:
            line = file.readline()
            while not line.startswith('# Number of images:'):
                line = file.readline()

            #print(line)
            numOfColCams = int(line.split(' ')[4].split(',')[0])
            if numOfColCams != numOfCams:
                #print('Number of cameras does not match, can\'t calculate transformation\n')
                raise NameError('Number of cameras does not match, can\'t calculate transformation')

            colCams = {}
            for i in range(numOfColCams):
                info = file.readline()
                file.readline()
                parsed = info.split(' ')
                # print(parsed)
                colCams[int(parsed[0]) - 1] = np.array([float(parsed[5]), float(parsed[6]), float(parsed[7])])

        transform = calculate_transform(holCams, colCams)


        response_list = [("transformation", transform)]

        response_dict = dict(response_list)
        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


@app.route("/api/cv/query_reconstruction/", methods=['GET'])
def api_query_reconstruction():
    try:
        global folder
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        with open(path + folder + "/dense/0/fused.ply", 'rb') as myfile:
            data = myfile.read()

        return bytes(data)

    except Exception as err:

        print("ko:", err)

@app.route("/api/cv/query_cameras/", methods=['GET'])
def api_query_cameras():
    try:
        global folder
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        os.system(path + 'COLMAP.bat model_converter --input_path ' + path + folder + '/dense/0/sparse --output_path '+ path + folder + '/dense/0/sparse --output_type TXT')

        with open(path + folder + "/dense/0/sparse/images.txt", 'r') as myfile:
            data = myfile.read()

        return data

    except Exception as err:

        print("ko:", err)


if __name__ == "__main__":
    # app.config.update(MAX_CONTENT_LENGTH=100*10**6)
    app.run(port=9099
            ,host='10.35.100.210')
    

