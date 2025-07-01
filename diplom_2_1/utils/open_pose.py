import sys
sys.path.append('../')
import os
from os.path import join
from os import listdir
import cv2
from scipy import spatial
import math
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from openpose.body.estimator import BodyPoseEstimator
from openpose.utils import draw_body_connections, draw_keypoints

def delete_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f'Ошибка при удалении файла {file_path}. {e}')

def convert_keypoint(keypoints_model, keypoints_user):
    all_out_keypoints_model = []
    all_out_keypoints_user = []
    
    if len(keypoints_model) <= len(keypoints_user):
        for i in range(len(keypoints_model)):
            out_keypoints_model = []
            out_keypoints_user = []
            if len(keypoints_model[i])!=0 and len(keypoints_user[i])!=0:
                for j in range(len(keypoints_model[i][0])):
                    if keypoints_model[i][0][j][2]==1 and keypoints_user[i][0][j][2]==1:
                        out_keypoints_model.append(keypoints_model[i][0][j][:2])
                        out_keypoints_user.append(keypoints_user[i][0][j][:2])
                all_out_keypoints_model.append(out_keypoints_model)
                all_out_keypoints_user.append(out_keypoints_user)
        
    else:
        for i in range(len(keypoints_user)):
            out_keypoints_model = []
            out_keypoints_user = []
            if len(keypoints_model[i])!=0 and len(keypoints_user[i])!=0:
                for j in range(len(keypoints_user[i][0])):
                    if keypoints_model[i][0][j][2]==1 and keypoints_user[i][0][j][2]==1:
                        out_keypoints_model.append(keypoints_model[i][0][j][:2])
                        out_keypoints_user.append(keypoints_user[i][0][j][:2])
                all_out_keypoints_model.append(out_keypoints_model)
                all_out_keypoints_user.append(out_keypoints_user)
    return all_out_keypoints_model, all_out_keypoints_user

def transform_keypoints(key_points_model, key_points_input):
    model_key_points = np.asarray(key_points_model)
    input_key_points = np.asarray(key_points_input)
 
# С помощью расширенной матрицы можно осуществить умножение вектора x на матрицу A и добавление вектора b за счёт единственного матричного умножения.
# Расширенная матрица создаётся путём дополнения векторов "1" в конце.
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]
 
# Расширим наборы ключевых точек до [[ x y 1] , [x y 1]]
    Y = pad(model_key_points)
    X = pad(input_key_points)
# Решим задачу наименьших квадратов X * A = Y
# и найдём матрицу аффинного преобразования A.
    A, res, rank, s = np.linalg.lstsq(X, Y)
    A[np.abs(A) < 1e-10] = 0  # превратим в "0" слишком маленькие значения

# Теперь, когда мы нашли расширенную матрицу A,
# мы можем преобразовать входной набор ключевых точек
    transform = lambda x: unpad(np.dot(pad(x), A))
    input_transform = transform(input_key_points)
    return input_transform

def cosine_similarities(features_a, features_b):

# Вычисляем косинусное сходство для каждой пары векторов
    similarities = []
    similarities_mean = 0
    count = 1
    for i in range(len(features_a)):
        for j in range(len(features_b)):
            similarity = 1 - spatial.distance.cosine(features_a[i], features_b[j])
            similarities.append((i, j, similarity))
            similarities_mean += similarity
            count += 1

# Выводим результаты
    return similarities_mean/count

def draw_frames(image, step, cosine_sim):
    # Window name in which image is displayed
    window_name = 'Image'
    img = np.array(image)
    
    # text
    text1 = 'cosine_similarities: ' + str(cosine_sim[step])
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org1 = (5, 35)
    org2 = (5, 330)

    # fontScale
    fontScale = 1
 
    # Red color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 2
 
    # Using cv2.putText() method
    img = cv2.putText(img, text1, org1, font, fontScale, color, thickness, cv2.LINE_AA, False)

    
    

    # Displaying the image
    return img 

def open_pose(ex = str(1), path_coach = '/mount/src/data_science_project/diplom_2_1/data_input/coach/', path_user = '/mount/src/data_science_project/diplom_2_1/data_input/user/', filename_output ='/mount/src/data_science_project/diplom_2_1/data_output/output.png'):
    count = 0
    delete_files(path_user + 'frames/')
    delete_files(path_coach + 'frames/')
    videoFile_coach = path_coach + 'ex'+ ex +'.mp4'
    dim = (640, 360)
    cap_coach = cv2.VideoCapture(videoFile_coach)   # загрузка видео 
    frameRate_coach = int(cap_coach.get(5)/2) # частота кадров
    x=1
    
    while(cap_coach.isOpened()):
      frameId_coach = cap_coach.get(1) # номер текущего кадра
      ret, frame = cap_coach.read()
      if (ret != True):
          break
      elif (frameId_coach % math.floor(frameRate_coach) == 0):
          filename ="video1_%d.jpg" % count;count+=1
          full_filename = path_coach + 'frames/' + str(filename)
          
          # resize image
          resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
          cv2.imwrite(full_filename, resized_frame)
    
    cap_coach.release()
    videoFile_user = path_user + 'ex'+ ex +'.mp4'
    
    cap_user = cv2.VideoCapture(videoFile_user)   # загрузка видео 
    frameRate_user = int(cap_user.get(5)/2) # частота кадров
    x=1
    count = 0
    while(cap_user.isOpened()):
      frameId_user = cap_user.get(1) # номер текущего кадра
      ret, frame = cap_user.read()
      if (ret != True):
          break
      elif (frameId_user % math.floor(frameRate_user) == 0):
          filename ="video1_%d.jpg" % count;count+=1
          full_filename = path_user + 'frames/' + str(filename)
          resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
          cv2.imwrite(full_filename, resized_frame)
    
    cap_user.release()
    estimator = BodyPoseEstimator(pretrained=True)
    all_preds_model = []
    for file in os.listdir(path_coach + 'frames/'):
        if file.endswith(".jpg"):
            image_src = cv2.imread(os.path.join(path_coach + 'frames/', file))
            keypoints = estimator(image_src)
            all_preds_model.append(keypoints)
    all_preds_input = []
    for file in os.listdir(path_user + 'frames/'):
        if file.endswith(".jpg"):
            image_src = cv2.imread(os.path.join(path_user + 'frames/', file))
            keypoints = estimator(image_src)
            all_preds_input.append(keypoints)
    all_model_keypoints, all_user_keypoints = convert_keypoint(all_preds_model, all_preds_input)
    cosine_sim = []

    for i in range(len(all_model_keypoints)):
        trans_input = transform_keypoints(all_model_keypoints[i], all_user_keypoints[i])
        current_cosine_sim = round(cosine_similarities(all_model_keypoints[i], trans_input),2)
        cosine_sim.append(current_cosine_sim)  
    list_img_model = []
    list_img_user = []
    
    for i in range(len(all_model_keypoints)):
        filename ="video1_%d.jpg" % i;i+=1
        full_filename_user = path_user + 'frames/' + str(filename)
        full_filename_model = path_coach + 'frames/' + str(filename)
        img_user = Image.open(full_filename_user).convert('RGB')
        img_model = Image.open(full_filename_model).convert('RGB')
        img_res = draw_frames(img_user, i-1, cosine_sim)
        img_res_model = draw_frames(img_model, i-1, cosine_sim)
        list_img_user.append(img_res)
        list_img_model.append(img_res_model)
    grid_model = cv2.vconcat(list_img_model)
    grid_user = cv2.vconcat(list_img_user)
    result_image = cv2.hconcat([grid_model, grid_user])
    cv2.imwrite(filename_output, result_image)
