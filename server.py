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
       # holCams[i] = holCams[i] - mHol
       # colCams[i] = colCams[i] - mCol
        colCams[i] = np.matmul(Q, np.array(colCams[i]))
        #print(colCams[i])
        #print(x)
        # = x
        allCol[0, i] = colCams[i][0]
        allCol[1, i] = colCams[i][1]
        allCol[2, i] = colCams[i][2]

        allHol[0, i] = holCams[i][0]
        allHol[1, i] = holCams[i][1]
        allHol[2, i] = holCams[i][2]

    holNorm  = np.linalg.norm(allHol)
    colNorm = np.linalg.norm(allCol)

    print("---------HolNorm---------")
    print(holNorm)
    print("")

    print("---------ColNom---------")
    print(colNorm)
    print("")

    scale = holNorm / colNorm


    print("---------SCALE---------")
    print(str(scale))
    print("")

    #for i in range(numOfCams):
    #    colCams[i] = scale * colCams[i]


    #allCol =  allCol.transpose() #ne

    #allHol = allHol.transpose() #ne

    print("---------HOLOLENS---------")
    print(allHol)
    print("")
    print("---------COLMAP---------")
    print(allCol)
    print("")

    #begin of procrustes
    (n,m) = allHol.shape
    (ny,my) = allCol.shape

    print("------------------")
    print("n = " + str(n) + ", m = " + str(m) + ", ny = " + str(ny) + ", my = " + str(my))
    print("")

    #muX = allHol.transpose().mean(axis=1)
    #muY = allCol.transpose().mean(axis=1)

    muX = allHol.mean(axis=0)
    muY = allCol.mean(axis=0)

    print("---------muX---------")
    print(muX)
    print("")
    print("---------muY---------")
    print(muY)
    print("")

    allHol0 = allHol - np.tile(muX, (n, 1))


    allCol0 = allCol - np.tile(muY, (n, 1))

    print("---------HOLOLENS0---------")
    print(allHol0)
    print("")
    print("---------COLMAP0---------")
    print(allCol0)
    print("")


    ssqX = np.square(allHol0)
    ssqX = sum(ssqX)

    ssqY = np.square(allCol0)
    ssqY = sum(ssqY)


    constX = np.all(ssqX <= abs((np.finfo(type(allHol[0, 0])).eps * n * muX)) ** 2)
    constY = np.all(ssqY <= abs((np.finfo(type(allCol[0, 0])).eps * n * muY)) ** 2)

    ssqX = np.sum(ssqX)

    ssqY = np.sum(ssqY)

    print("------------------")
    print("ssqX = " + str(ssqX) + ", ssqY = " + str(ssqY) + ", constX = " + str(constX) + ", constY = " + str(constY))
    print("")






    if not constX and not constY:
        normX = np.sqrt(ssqX)
        normY = np.sqrt(ssqY)

        print("------------------")
        print("normX = " + str(normX) + ", normY = " + str(normY))
        print("")

        allHol0 = allHol0 / normX
        allCol0 = allCol0 / normY


        A = np.matmul(allHol0.transpose(), allCol0)
        U, S, V = np.linalg.svd(A) # S je 16x1 misto 16x16 (jen diagonala), nevadi
                                   # V je transpozovane oproti matlabu
                                   # vraci trochu jiny vysledky nez matlab
        V = V.transpose()
        R = np.matmul(V, np.transpose(U))

        print("---------A---------")
        print(A)
        print("")

        print("---------U---------")
        print(U)
        print("")
        print("---------S---------")
        print(S)
        print("")
        print("---------V---------")
        print(V)
        print("")
        print("---------R---------")
        print(R)
        print("")



        traceTA = np.sum(S)
        #b = 1
        b = traceTA * normX / normY
       # d = 1 + ssqY / ssqX - 2 * traceTA * normY / normX
        d = 1 - traceTA * traceTA
        #Z = normY*Y0 * T + repmat(muX, n, 1);
        Z = normX * traceTA * np.matmul(allCol0, R) + np.tile(muX, (n, 1))
        c = muX - b * np.matmul(muY, R)


        print("---------traceTA---------")
        print(traceTA)
        print("")

        print("---------c---------")
        print(c)
        print("")
        print("*****************************************************")

        #transform = struct('T', R, 'b', b, 'c', repmat(c, n, 1));
        transformation['scale'] = traceTA * scale
        transformation['rotation1'] = R[0].tolist()
        transformation['rotation2'] = R[1].tolist()
        transformation['rotation3'] = R[2].tolist()
        transformation['translation'] = c.tolist()




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
        print(request.json)

        quality = request.json['folder']
        print(quality)

        os.system(path + "COLMAP.bat automatic_reconstructor \ --workspace_path " + path + folder + " \ --image_path " + path + folder + " --quality " + quality)

        response_list = [("folder", folder)]

        response_dict = dict(response_list)


        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


def quaternion_to_matrix(q):
    R = np.zeros((3,3))
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]

    R[0][0] = 1 - 2 * qy * qy - 2 * qz * qz
    R[0][1] = 2 * qx * qy - 2 * qz * qw
    R[0][2] = 2 * qx * qz + 2 * qy * qw

    R[1][0] = 2 * qx * qy + 2 * qz * qw
    R[1][1] = 1 - 2 * qx * qx - 2 * qz * qz
    R[1][2] = 2 * qy * qz - 2 * qx * qw

    R[2][0] = 2 * qx * qz - 2 * qy * qw
    R[2][1] = 2 * qy * qz + 2 * qx * qw
    R[2][2] = 1 - 2 * qx * qx - 2 * qy * qy

    return R

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

        for i in range(0, numOfCams):
            t = np.array([float(request.json['cameras'][i]['position']['x']), float(request.json['cameras'][i]['position']['y']), float(request.json['cameras'][i]['position']['z'])])
            q = np.array([float(request.json['cameras'][i]['rotation']['w']), float(request.json['cameras'][i]['rotation']['x']), float(request.json['cameras'][i]['rotation']['y']), float(request.json['cameras'][i]['rotation']['z'])])
            R = quaternion_to_matrix(q)
            holCams[i] = np.matmul(-R.transpose(), t)
            #C = R' * t
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
                #QW, QX, QY, QZ
                q = np.array([float(parsed[1]), float(parsed[2]), float(parsed[3]), float(parsed[4])])
                R = quaternion_to_matrix(q)
                t = np.array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                colCams[int(parsed[0]) - 1] = np.matmul(-R.transpose(), t)
                # C = R' * t

        transform = calculate_transform(holCams, colCams)


        response_list = transform



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
    

