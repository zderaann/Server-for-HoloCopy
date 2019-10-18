# -*- encoding: utf-8 -*-
# pip install flask

import flask
from flask import request
from PIL import Image
import numpy as np
import time
import os
import socket
import math

app = flask.Flask(__name__)


folder = ""

rotation = np.array([])
scale = 0.0
translation = np.array([])
globmuY = np.array([])
globnormY = 0.0
i = 0

mean = np.array([])



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

    #scale = holNorm / colNorm


    #print("---------SCALE---------")
    #print(str(scale))
    #print("")

    #for i in range(numOfCams):
    #    colCams[i] = scale * colCams[i]


    allCol =  allCol.transpose()

    allHol = allHol.transpose()

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

    global mean
    mean = muY

    print("---------muX---------")
    print(muX)
    print("")
    print("---------muY---------")
    print(muY)
    print("")

    allHol0 = allHol - np.tile(muX, (n, 1))


    allCol0 = allCol - np.tile(muY, (n, 1))
    global globmuY
    globmuY = muY

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
        global globnormY
        globnormY = normY

        A = np.matmul(allHol0.transpose(), allCol0)
        U, S, V = np.linalg.svd(A)
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
        global scale
        scale = b
        global rotation
        rotation = R
        global translation
        translation = c

        print("---------traceTA---------")
        print(traceTA)
        print("")

        print("---------c---------")
        print(c)
        print("")


        #transform = struct('T', R, 'b', b, 'c', repmat(c, n, 1));
        transformation['scale'] = b
        transformation['rotation1'] = R[0].tolist()
        transformation['rotation2'] = R[1].tolist()
        transformation['rotation3'] = R[2].tolist()
        transformation['translation'] = c.tolist()

        print("---------Z---------")
        print(Z)
        print("*****************************************************")

        cams = []
        for i in range(0, numOfCams):
            for j in range(0, 3):
                cams.append(Z[i][j])


        transformation['cameras'] = cams

        '''print("----TOLIST----")
        print("----b----")
        print(b)
        print("----R----")
        print(R.tolist())
        print("----c----")
        print(c.tolist())
        print("----Z----")
        print(Z.transpose().tolist())'''

        print("----Z----")
        print(Z.transpose().tolist())


        # The degenerate cases: X all the same, and Y all the same.
        ''' elif constX:
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
            print(R)'''


    return transformation


@app.route("/api/cv/create_folder/", methods=['POST'])
def api_create_folder():
    try:
        global folder
        folder = str(int(time.time()))
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        if not os.path.exists(folder):
            workdir = path + folder
            os.makedirs(workdir)

            print("created folder: " + folder)

            # ADDED FOR RIG

            os.makedirs(workdir + "/images")
            os.makedirs(workdir + "/reconstruction")
            #os.makedirs(workdir + "/sparse")
            #os.makedirs(workdir + "/dense")
            #os.makedirs(workdir + "/rig")
            # -------------------



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
            name = str(start).replace('.', '_') + "_" + str(f) + ".jpg"
            filename = path + folder + "/" + name
            print("saving  as: " + filename)

            pimg.save(filename);

            response_list = [("file", name), ("folder", folder)]

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

def matrix_to_quaternion(R):
    tr = R[0][0] + R[1][1] + R[2][2]

    if (tr > 0):
        S = 0.5 / math.sqrt(tr+1.0)
        qw = 0.25 / S
        qx = (R[2][1] - R[1][2]) * S
        qy = (R[0][2] - R[2][0]) * S
        qz = (R[1][0] - R[0][1]) * S
    elif ((R[0][0] > R[1][1]) and (R[0][0] > R[2][2])):
        S = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        qw = (R[2][1] - R[1][2]) / S
        qx = 0.25 * S
        qy = (R[0][1] + R[1][0]) / S
        qz = (R[0][2] + R[2][0]) / S
    elif (R[1][1] > R[2][2]):
        S = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        qw = (R[0][2] - R[2][0]) / S
        qx = (R[0][1] + R[1][0]) / S
        qy = 0.25 * S
        qz = (R[1][2] + R[2][1]) / S
    else:
        S = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        qw = (R[1][0] - R[0][1]) / S
        qx = (R[0][2] + R[2][0]) / S
        qy = (R[1][2] + R[2][1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])



@app.route("/api/cv/get_transformation/", methods=['POST']) #TODO FILTR KAMER (JEN CAMERA_0 pouzit)
def api_get_transformation():
    try:
        global folder
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        os.system(path + 'COLMAP.bat model_converter --input_path ' + path + folder + '/dense/sparse --output_path ' + path + folder + '/dense/sparse --output_type TXT')                  #zmeneno pro rig
        print()
        numOfCams = request.json['numberOfCams']
        #print()
        #print(request.json)
        numOfCamTypes = 0
        with open(path + folder + "/dense/sparse/cameras.txt") as myfile:
            line = myfile.readline()
            while not line.startswith('# Number of cameras:'):
                line = myfile.readline()

            numOfCamTypes = int(line.split(' ')[-1].split('\n')[0])


        with open(path + folder + "/dense/sparse/images.txt", 'r') as file:                  #zmeneno pro rig
            line = file.readline()
            while not line.startswith('# Number of images:'):
                line = file.readline()

            #print(line)
            numOfColCams = int(line.split(' ')[4].split(',')[0])

            #if numOfColCams != numOfCams:
                #print('Number of cameras does not match, can\'t calculate transformation\n')
            #    raise NameError('Number of cameras does not match, can\'t calculate transformation')

            colCamsDict = {}
            colRotMats = {}
            colRotQuat = []     # tohle se ale potom netridi -> chyba?
            colTransDict = {}

            for i in range(numOfColCams):
                info = file.readline()
                file.readline()
                parsed = info.split(' ')
                # print(parsed)
                #QW, QX, QY, QZ
                id = parsed[9].split('\n')[0]
                print()
                print(id)
                if id.startswith('camera_0'):                   #zmeneno pro rig
                    q = np.array([float(parsed[1]), float(parsed[2]), float(parsed[3]), float(parsed[4])])
                    colRotQuat.append(q[0])
                    colRotQuat.append(q[1])
                    colRotQuat.append(q[2])
                    colRotQuat.append(q[3])
                    R = quaternion_to_matrix(q)
                    t = np.array([float(parsed[5]), float(parsed[6]), float(parsed[7])])
                    id = id.split('/')[1] .split('.')[0]                 #zmeneno pro rig
                    print(id)
                    colCamsDict[id] = np.matmul(-R.transpose(), t)
                    colRotMats[id] = R
                    colTransDict[id] = t
                    print(id[0])
                    # C = R' * t

        colCams = []
        colRot = []
        colTrans = []
        for key in sorted(colCamsDict):
            colCams.append(colCamsDict[key])
            colRot.append(colRotMats[key])
            colTrans.append(colTransDict[key])

        print()

        holCamsDict = {}
        #cams = request.json['cameras']

        for i in range(0, numOfCams):
            t = np.array([float(request.json['cameras'][i]['position']['x']), float(request.json['cameras'][i]['position']['y']), float(request.json['cameras'][i]['position']['z'])])
            q = np.array([float(request.json['cameras'][i]['rotation']['w']), float(request.json['cameras'][i]['rotation']['x']), float(request.json['cameras'][i]['rotation']['y']), float(request.json['cameras'][i]['rotation']['z'])])
            R = quaternion_to_matrix(q)
            holID = request.json['cameras'][i]['imageID']
            #holCamsDict[request.json['cameras'][i]['imageID']] = np.matmul(-R.transpose(), t)
            holCamsDict[holID] = t
            print(holID)
            #C = R' * t
            #print(holCams[i])

        holCams = []
        for key in sorted(holCamsDict):
            if key in colCamsDict:
                holCams.append(holCamsDict[key])



        if len(holCams) != len(colCams):
                print('Number of cameras does not match, can\'t calculate transformation\n')
                raise NameError('Number of cameras does not match, can\'t calculate transformation')

        transform = calculate_transform(holCams, colCams)

        ### ROTACE KAMER ###
        global rotation
        numOfColCams = int(numOfColCams / numOfCamTypes)                 #pridano pro rig
        for i in range(0, numOfColCams):
            index = i * 4
            q = np.array([colRotQuat[index], colRotQuat[index+1], colRotQuat[index + 2], -colRotQuat[index + 3]])
            R = quaternion_to_matrix(q)
            rotated = np.matmul(R, rotation)
            q2 = matrix_to_quaternion(rotated)
            colRotQuat[index] = q2[0]
            colRotQuat[index + 1] = q2[1]
            colRotQuat[index + 2] = q2[2]
            colRotQuat[index + 3] = q2[3]


        ###  Kalkulace rotace ###


        XHx = [None] * (numOfColCams * 5)
        XHy = [None] * (numOfColCams * 5)
        XHz = [None] * (numOfColCams * 5)

        with open(path + folder + "/dense/sparse/cameras.txt", 'r') as file:        #zmeneno pro rig
            line = file.readline()
            while not line.startswith('# Number of cameras:'):
                line = file.readline()

            info = file.readline()                  #moved from for for rig
            parsed = info.split(' ')

            print('--------------Line---------------')
            print(info)


            width = int(parsed[2])
            height = int(parsed[3])
            fx = float(parsed[4])
            fy = float(parsed[5])
            pp1 = float(parsed[6])
            pp2 = float(parsed[7])

            for key in sorted(colCamsDict):
                camID = sorted(colCamsDict).index(key)
                u = np.array([[0, width, width, 0, 0], [0, 0, height, height, 0], [1, 1, 1, 1, 1]])
                K = np.array([[2 * fx, 0, 2.3 * pp1], [0, 2 * fy, 2.3 * pp2], [0, 0, 2.3]])
                global scale
                global translation
                Qxz = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
                rct = colTrans[camID]
                rcR = colRot[camID]

                print('--------------rc.t---------------')
                print(rct)
                print('--------------rc.R---------------')
                print(rcR)

                tc = np.transpose(np.tile(rct,(u.shape[1],1)))
                Xc = np.matmul(rcR.transpose(),np.matmul(np.linalg.inv(K), u) - tc)

                print('--------------Xc---------------')
                print(Xc)

                Xc = np.matmul(Qxz, Xc)

                print('--------------Xc---------------')
                print(Xc)


                txh = np.tile(translation,(5,1))

                XH = np.transpose(scale * np.matmul(np.transpose(Xc), rotation) + txh)

                print('--------------XH---------------')
                print(XH)

                for j in range(0, 5): #XH[rada, sloupec]
                    XHx[(5 * (camID)) + j] = XH[0, j]
                    XHy[(5 * (camID)) + j] = XH[1, j]
                    XHz[(5 * (camID)) + j] = XH[2, j]


        transform['rotcams'] = colRotQuat
        transform['XHx'] = XHx
        transform['XHy'] = XHy
        transform['XHz'] = XHz
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

def decimate_model():
    global folder
    # folder = "1547206402"
    print("Decimating model")
    path = os.path.dirname(os.path.realpath(__file__)) + "/"
    modelpath = path + folder + "/dense"                  #zmeneno pro rig

    os.system('.\Blender\\blender.exe --background --python "' + path + 'decimation_script.py" -- "' + modelpath + '"')
    return

@app.route("/api/cv/download_model/", methods=['GET'])
def api_download_model():
    try:
        global folder
        #folder = "1547206402"
        print("Getting path")
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        modelpath = path + folder + "/dense/0"
        filename = modelpath + "/decimated.ply"

        decimate_model()
        #os.system('.\Blender\\blender.exe --background --python "' + path + 'decimation_script.py" -- "' + modelpath +'"')


        print("Getting file")
        verts = []
        colours = []
        faces = []

        with open(filename, 'r') as myfile:
            while True:
                data = myfile.readline()
                #print(data)
                if data.startswith('element vertex'):
                    numOfVerts = int(data.split(' ')[2].split('\n')[0])
                    print("Number of vertices: " + str(numOfVerts))
                elif data.startswith('element face'):
                    numOfFaces= int(data.split(' ')[2].split('\n')[0])
                    print("Number of faces: " + str(numOfFaces))
                elif data.startswith('end_header'):
                    break
            for i in range(0, numOfVerts):
                data = myfile.readline()
                parsed = data.split('\n')[0]
                parsed = parsed.split(' ')
                verts.append(float(parsed[0]))
                verts.append(float(parsed[1]))
                verts.append(float(parsed[2]))

                colours.append(float(parsed[3]))
                colours.append(float(parsed[4]))
                colours.append(float(parsed[5]))
                #print(parsed)

            global globmuY
            global globnormY
            global translation
            global scale
            global rotation

            for i in range(0, numOfVerts):
                index = i * 3
                vertex = np.array([verts[index], -1.0 * (verts[index+1]), verts[index+2]])
                #vertex = vertex - globmuY
                #vertex = vertex / globnormY
                vertex = scale * np.matmul(vertex, rotation) + translation

                verts[index] = vertex[0]
                verts[index+1] = vertex[1]
                verts[index+2] = vertex[2]


            for j in range(0, numOfFaces):
                data = myfile.readline()
                parsed = data.split('\n')[0]
                parsed = parsed.split(' ')
                num =  int(parsed[0])
                if not num == 3:
                    print(num)
                    raise ValueError('Face is not a triangle!')

                faces.append(int(parsed[1]))
                faces.append(int(parsed[2]))
                faces.append(int(parsed[3]))
                #print([num,int(parsed[1]), int(parsed[2]), int(parsed[3])])

       # verts, colours, faces = decimate(verts, colours, faces)
        mesh = {}
        print("Done")

        mesh['verts'] = verts
        mesh['faces'] = faces
        mesh['cols'] = colours

        response_list = mesh
        response_dict = dict(response_list)
        return flask.jsonify(response_dict)

    except Exception as err:

        print("ko:", err)




@app.route("/api/cv/get_textured_model/", methods=['GET'])
def api_get_textured_model():
    try:
        global folder
        #folder = "1562574263"
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        modelpath = path + folder + "/dense"                  #zmeneno pro rig
        filename = modelpath + "/decimated.ply"

        #decimate_model()


        print("Decimation and texture baking")

        os.system(
            '.\Blender\\blender.exe --background --python "' + path + 'decimation_texture_bake.py" -- "' + modelpath + '"')

        print("Getting file")
        verts = []
        colours = []
        faces = []
        uvs = []

        with open(filename, 'r') as myfile:
            while True:
                data = myfile.readline()
                # print(data)
                if data.startswith('element vertex'):
                    numOfVerts = int(data.split(' ')[2].split('\n')[0])
                    print("Number of vertices: " + str(numOfVerts))
                elif data.startswith('element face'):
                    numOfFaces = int(data.split(' ')[2].split('\n')[0])
                    print("Number of faces: " + str(numOfFaces))
                elif data.startswith('end_header'):
                    break
            for i in range(0, numOfVerts):
                data = myfile.readline()
                parsed = data.split('\n')[0]
                parsed = parsed.split(' ')
                verts.append(float(parsed[0]))
                verts.append(float(parsed[1]))
                verts.append(float(parsed[2]))

                uvs.append(float(parsed[3]))
                uvs.append(float(parsed[4]))

                colours.append(float(parsed[5]))
                colours.append(float(parsed[6]))
                colours.append(float(parsed[7]))
                # print(parsed)


            for i in range(0, numOfVerts):
                index = i * 3

                vertex = np.array([verts[index], -1.0 * (verts[index + 1]), verts[index + 2]])
                # vertex = vertex - globmuY
                # vertex = vertex / globnormY
                vertex = scale * np.matmul(vertex, rotation) + translation

                verts[index] = vertex[0]
                verts[index + 1] = vertex[1]
                verts[index + 2] = vertex[2]

            for j in range(0, numOfFaces):
                data = myfile.readline()
                parsed = data.split('\n')[0]
                parsed = parsed.split(' ')
                num = int(parsed[0])
                if not num == 3:
                    print(num)
                    raise ValueError('Face is not a triangle!')

                faces.append(int(parsed[1]))
                faces.append(int(parsed[2]))
                faces.append(int(parsed[3]))
                # print([num,int(parsed[1]), int(parsed[2]), int(parsed[3])])

        # verts, colours, faces = decimate(verts, colours, faces)
        mesh = {}
        print("Done")

        mesh['verts'] = verts
        mesh['faces'] = faces
        mesh['cols'] = colours
        mesh['uvs'] = uvs


        response_list = mesh
        response_dict = dict(response_list)
        return flask.jsonify(response_dict)


    except Exception as err:

        print("ko:", err)



@app.route("/api/cv/download_texture/", methods=['GET'])
def api_download_texture():
    try:
        print("Downloading Texture")
        #folder = "1562574263"
        global folder
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        modelpath = path + folder + "/dense"                  #zmeneno pro rig


        filename =  "texture.png"


        return flask.send_from_directory(modelpath, filename)


    except Exception as err:

        print("ko:", err)

def write_header_to_file(file):
    file.write("ImageFileName,"
               "FrameToOrigin.m11,FrameToOrigin.m12,FrameToOrigin.m13,FrameToOrigin.m14,"
               "FrameToOrigin.m21,FrameToOrigin.m22,FrameToOrigin.m23,FrameToOrigin.m24,"
               "FrameToOrigin.m31,FrameToOrigin.m32,FrameToOrigin.m33,FrameToOrigin.m34,"
               "FrameToOrigin.m41,FrameToOrigin.m42,FrameToOrigin.m43,FrameToOrigin.m44,"
               "CameraViewTransform.m11,CameraViewTransform.m12,CameraViewTransform.m13,CameraViewTransform.m14,"
               "CameraViewTransform.m21,CameraViewTransform.m22,CameraViewTransform.m23,CameraViewTransform.m24,"
               "CameraViewTransform.m31,CameraViewTransform.m32,CameraViewTransform.m33,CameraViewTransform.m34,"
               "CameraViewTransform.m41,CameraViewTransform.m42,CameraViewTransform.m43,CameraViewTransform.m44\n")



@app.route("/api/cv/save_sensor_data/", methods=['POST'])
def api_save_sensor_data():
    try:
        global folder

        path = os.path.dirname(os.path.realpath(__file__)) + "/"


        start = time.time()
        #global i
        for f in request.files:
            img = request.files[f]
            sensor = request.form["sensor"]
            pimg = Image.open(img.stream)
            if sensor == "0":
                pimg = pimg.transpose(Image.FLIP_TOP_BOTTOM)
            else:

                pimg = pimg.transpose(Image.ROTATE_270)
                #pimg = pimg.transpose(Image.FLIP_LEFT_RIGHT)
               # pimg = pimg.resize((480, 640), Image.ANTIALIAS) #mozna bude zase potreba

            camname = "camera_" + sensor

            if not os.path.exists(folder + "/images/" + camname):
                os.makedirs(path + folder + "/images/" + camname)

            if sensor == "0":
                name = request.form["name"].replace('.', '_') + ".jpg"
            else:
                name = request.form["name"].replace('.', '_') + ".pgm"

            filename = path + folder + "/images/" + camname + "/" + name
            print("saving  as: " + filename)

            pimg.save(filename)

            csvfilename = path + folder + "/" + camname + ".csv"
            fileexists = False
            if os.path.exists(csvfilename):
                fileexists = True

            with open(csvfilename, 'a') as file:
                if not fileexists:
                    write_header_to_file(file)
                file.write(camname + "/" + name + "," + request.form["frameToOrigin"] + "," + request.form["cameraViewTransform"] + "\n")
            #transposed.save(filename)
            #i = i + 1

        response_list = [("folder", folder)]

        response_dict = dict(response_list)

        #print(response_dict)
        end = time.time()
        print("Processing time: {} ".format(end - start))
        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


def registration(name, filename):
    global folder

    path = os.getcwd() + "/"
    workdir = path + folder

    listname = workdir + "/imagelist" + name + ".txt"
    with open(listname, 'a') as imagelist:
        imagelist.write(filename + '\n')

    numlines = sum(1 for line in open(listname))
    if numlines == 5:
        os.system(
            path + "COLMAP.bat feature_extractor --database_path " + workdir + "/database.db --image_path " + workdir + "/images --image_list_path " + listname + " --SiftExtraction.num_threads 2")

        os.system(
            path + "COLMAP.bat vocab_tree_matcher --database_path " + workdir + "/database.db --VocabTreeMatching.vocab_tree_path " + path + "vocab_tree.bin --VocabTreeMatching.match_list_path " + listname + " --SiftMatching.num_threads 2")


        '''os.system(
            path + "COLMAP.bat exhaustive_matcher --database_path " + workdir + "/database.db --SiftMatching.num_threads 2")

        os.system(
            path + "COLMAP.bat feature_extractor --database_path " + workdir + "/database.db --image_path " + workdir + "/images --ImageReader.single_camera_per_folder 1 --image_list_path " + listname + " --SiftExtraction.num_threads 2")'''

        '''
        os.system(
            path + "COLMAP.bat mapper --database_path " + workdir + "/database.db --image_path " + workdir + "/images --export_path " + workdir + "/sparse --Mapper.num_threads 2")
        
        os.system(
            path + "COLMAP.bat bundle_adjuster --input_path " + workdir + "/sparse --output_path " + workdir + "/sparse")'''



@app.route("/api/cv/register_images/", methods=['POST'])
def api_register_images():
    try:
        global folder

        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        path = path.replace('\\', '/')

        start = time.time()
        #global i
        for f in request.files:
            img = request.files[f]

            pimg = Image.open(img.stream)
            if request.form["sensor"] == "0":
                pimg = pimg.transpose(Image.FLIP_TOP_BOTTOM)
            else:

                pimg = pimg.transpose(Image.ROTATE_270) # for left-left
                pimg = pimg.transpose(Image.FLIP_LEFT_RIGHT)
                pimg = pimg.resize((240, 320), Image.ANTIALIAS)

            if not os.path.exists(folder + "/images/camera_" + request.form["sensor"]):
                os.makedirs(path + folder + "/images/camera_" + request.form["sensor"])

            name = request.form["name"].replace('.', '_') + ".jpg"
            filename = path + folder + "/images/camera_" + request.form["sensor"] + "/" + name
            print("saving  as: " + filename)

            pimg.save(filename)
            registration(name, "camera_" + request.form["sensor"] + "/" + name)
            #transposed.save(filename)
            #i = i + 1

        response_list = [("folder", folder)]

        response_dict = dict(response_list)

        #print(response_dict)
        end = time.time()
        print("Processing time: {} ".format(end - start))
        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


def create_config_file():
    global folder
    path = os.path.dirname(os.path.realpath(__file__)) + "/"

    workdir = path + folder

    with open(workdir + "/sparse/0/cameras.txt", 'r') as camfile:
        line = camfile.readline()
        while not line.startswith("# Number"):
            line = camfile.readline()

        numofcams = int(line.split(' ')[-1].split('\n')[0])
        with open(workdir + "/camsconfig.json", 'w') as config:
            config.write("[{\"ref_camera_id\": 1, \"cameras\": [")
            for i in range(0, numofcams):
                line = camfile.readline()
                camid = int(line.split(' ')[0])
                config.write("{\"camera_id\": 1, \"image_prefix\": \"camera_" + str(camid - 1) + "\"}")

                if i != numofcams - 1:
                    config.write(", ")
            config.write("]}]")




@app.route("/api/cv/run_reconstruction_for_rig/", methods=['POST'])
def api_run_reconstruction_for_rig():
    try:
        global folder
        path = os.path.dirname(os.path.realpath(__file__)) + "/"
        print(request.json)

        workdir = path + folder

        #schovat tohle kdyz prubezna registrace, pro prubeznou registraci problem s camera ID a rig config
        '''os.system(
            path + "COLMAP.bat feature_extractor --database_path " + workdir + "/database.db --image_path " + workdir + "/images --ImageReader.single_camera_per_folder 1 --SiftExtraction.num_threads 2")

        os.system(
            path + "COLMAP.bat exhaustive_matcher --database_path " + workdir + "/database.db --SiftMatching.num_threads 2")'''
        #-----------

        os.system(
            path + "COLMAP.bat mapper --database_path " + workdir + "/database.db --image_path " + workdir + "/images --export_path " + workdir + "/sparse --Mapper.num_threads 2"
        )

        os.system(
            path + "COLMAP.bat model_converter --input_path " + workdir + "/sparse/0 --output_path " + workdir + "/sparse/0 --output_type TXT"
        )

        create_config_file()

        os.system(
            path + "COLMAP.bat rig_bundle_adjuster --input_path " + workdir + "/sparse/0 --output_path " + workdir + "/rig --rig_config_path " + workdir + "/camsconfig.json"
        )

        os.system(
            path + "COLMAP.bat image_undistorter --image_path " + workdir + "/images --input_path " + workdir + "/rig --output_path " + workdir + "/dense --output_type COLMAP --max_image_size 2000")

        os.system(
            path + "COLMAP.bat patch_match_stereo --workspace_path " + workdir + "/dense --workspace_format COLMAP --PatchMatchStereo.geom_consistency true")

        os.system(
            path + "COLMAP.bat stereo_fusion --workspace_path " + workdir + "/dense --workspace_format COLMAP --input_type geometric --output_path " + workdir + "/dense/fused.ply")

        os.system(
            path + "COLMAP.bat poisson_mesher --input_path " + workdir + "/dense/fused.ply --output_path " + workdir + "/dense/meshed-poisson.ply --PoissonMeshing.num_threads 2")

        response_list = [("folder", folder)]

        response_dict = dict(response_list)


        return flask.jsonify(response_dict)

    except Exception as err:
        print("ko:", err)

    return "ok"


#TODO funkce pro rekonstrukci pomoci reconstruction_script.py






if __name__ == "__main__":
    # app.config.update(MAX_CONTENT_LENGTH=100*10**6)
    IP = socket.gethostbyname(socket.gethostname())
    app.run(port=9099
,host=IP)
    

