import insightface
import numpy as np
import cv2
from skimage import transform as trans
import os
import pickle
import modules.globals

FACE_ANALYSER = None
center_points_cur_frame = []
center_points_prev_frame = []
track_id = 0

def get_face_analyser():
    global FACE_ANALYSER

    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=modules.globals.execution_providers, allowed_modules=['recognition', 'detection'])
        FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER

def get_one_face(frame):
    face = get_face_analyser().get(frame)
    try:
        return max(face, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    except ValueError:
        return None

def get_many_faces(frame):
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None

def draw_on(img, faces):
    dimg = img.copy()
    if (type(faces) == list):
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int32)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int32)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            # if face.gender is not None and face.age is not None:
            #     cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            if face.identity is not None and 'identity' in face:
                cv2.putText(dimg,'%s,%d'%(extract_name_from_path(face.identity), face.distance), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    else:
        box = faces.bbox.astype(np.int32)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if faces.kps is not None:
            kps = faces.kps.astype(np.int32)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        # if faces.gender is not None and faces.age is not None:
        #     cv2.putText(dimg,'%s,%d'%(faces.sex,faces.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

        if face.identity is not None and 'identity' in face:
            cv2.putText(dimg,'%s,%d'%(extract_name_from_path(face.identity), face.distance), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    return dimg

def save_cut_frame(img, faces):
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int32)
        color = (0, 0, 255)
        if face.kps is not None:
            kps = face.kps.astype(np.int32)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
        cut_img = dimg[box[1] : box[3], box[0] : box[2]]
        cv2.imwrite(f'inputs/face{i}.png',face_aligin(dimg, face.kps))

def face_aligin(img,kps, size_output = (160,160)):
    src = np.array([
       [  54.70657349,   73.85186005],
       [ 105.04542542,   73.57342529],
       [  80.03600311,  102.48085785],
       [  59.35614395,  131.95071411],
       [ 101.04272461,  131.72013855]], dtype=np.float32)
    landmark = np.array(kps)
    
    dst = landmark.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    wrapped = cv2.warpAffine(img, M, size_output, borderValue = 0.0)
    return wrapped

def verify(faces, database: str):
    filenames = []
    embeddings = []

    if not os.path.exists(f'{database}/embeddings.pkl'):
        for filename in os.listdir(database):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(database, filename)
                imgbase = cv2.imread(file_path)
                face = get_one_face(imgbase)
                filenames.append(file_path)
                embeddings.append(face.embedding)

        with open(f'{database}/embeddings.pkl', 'wb') as f:
            pickle.dump({'filenames': filenames, 'embeddings': embeddings}, f)
        print("Lưu xong embeddings.pkl")

    with open(f'{database}/embeddings.pkl', 'rb') as f:
        data = pickle.load(f)

    filenames = data['filenames']
    embeddings = data['embeddings']
    index_min = 0
    distance_hold = 24

    if(len(filenames) + 1 != len(os.listdir(database))):
        for filename in os.listdir(database):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                file_path = os.path.join(database, filename)
                imgbase = cv2.imread(file_path)
                face = get_one_face(imgbase)
                filenames.append(file_path)
                embeddings.append(face.embedding)

        with open(f'{database}/embeddings.pkl', 'wb') as f:
            pickle.dump({'filenames': filenames, 'embeddings': embeddings}, f)
    else:
        if type(faces) == list:
            for face in faces:
                distance_min = np.linalg.norm(face.embedding - embeddings[index_min])
                for index, embedding in enumerate(embeddings):
                    distance = np.linalg.norm(face.embedding - embedding)
                    if (distance < distance_min):
                        distance_min = distance
                        index_min = index

                face['distance'] = distance_min
                if (face['distance'] < distance_hold):
                    face['identity'] = filenames[index_min]
                else: 
                    face['identity'] = "unknow"
        else:
            distance_min = np.linalg.norm(faces.embedding - embeddings[index_min])
            for index, embedding in enumerate(embeddings):
                distance = np.linalg.norm(faces.embedding - embedding)
                if (distance < distance_min):
                    distance_min = distance
                    index_min = index

            faces['distance'] = distance_min
            if (faces['distance'] < distance_hold):
                faces['identity'] = filenames[index_min]
            else: 
                faces['identity'] = "unknow"
            return [faces]
    return faces

def extract_name_from_path(path):
    filename = path.split('\\')[-1]
    name_without_extension = filename.split('.')[0]
    return name_without_extension
