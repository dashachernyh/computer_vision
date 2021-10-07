import cv2
import numpy as np


def check_size(img_param, x_p, y_p, w_p, h_p):  # проверка, что не вышли за размеры изображения
    img_h, img_w = img_param.shape[:2]
    if (x_p + w_p) > img_w:
        w_p = img_w - x_p
    if(y_p + h_p) > img_h:
        h_p = img_h - y_p


def clarity(sours_img, t, k):  # увеличение резкости
    gaussian = cv2.GaussianBlur(sours_img, (5, 5), 1)
    result = np.zeros(sours_img.shape, dtype='uint8' )

    for i in range(sours_img.shape[0]):
        for j in range(sours_img.shape[1]):
            result[i][j] = sours_img[i][j] - gaussian[i][j]
            if result[i][j][0] <= t[0] & result[i][j][1] <= t[1] & result[i][j][2] <= t[2]:
                result[i][j] = sours_img[i][j]
            else:
                result[i][j] = sours_img[i][j] + k * result[i][j]
    return result


def finaly_filtr(m, f1, f2):  # фильтр, для пункта 9
    one = [1, 1, 1]
    m3 = cv2.merge((m, m, m))  # создание трех канального ч-б
    result = np.zeros(f1.shape, dtype='uint8' )
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            result[i][j] = m3[i][j]*f2[i][j] + (one - m3[i][j])*f1[i][j]
    return result


# Лабораторная работа №1

img = cv2.imread('images/img3.jpg')
img_copy = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = cv2.CascadeClassifier('neuron_face.xml')
coord_faces = faces.detectMultiScale(img_gray, scaleFactor=1.9, minNeighbors=1)
for(x, y, w, h) in coord_faces:
    cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
cv2.imshow('face', img_copy)
cv2.waitKey(1000)

coup_x = round(x - w*0.1/2)
coup_y = round(y - h*0.1/2)
inc_w = round(w*1.1)
inc_h = round(h*1.1)

check_size(img, coup_x, coup_y, inc_w, inc_h)

coup_img = img[coup_y:(coup_y + inc_h), coup_x:(coup_x+inc_w)]
face_img = coup_img.copy()  # будем использовть для пунктов 7 - 9
cv2.imshow('crop_face', coup_img)
cv2.waitKey(1000)

# работаем с обрезанным изображением. перевод в бинарное
coup_img = cv2.cvtColor(coup_img, cv2.COLOR_BGR2GRAY)
coup_img = cv2.Canny(coup_img, 100, 130)

cv2.imshow('face_contours', coup_img)
cv2.waitKey(1000)

contours, _ = cv2.findContours(coup_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
con_copy = []
i = 0
for j in range(len(contours)):
    minRect = cv2.minAreaRect(contours[j])  # (x,y) , (w, h), angle rotation
    s_minRect = minRect[1][0]*minRect[1][1]
    if s_minRect >= 10:
        con_copy.append(contours[j])
img_cont = np.zeros(coup_img.shape, dtype='uint8')
cv2.drawContours(img_cont, con_copy, -1, (255, 255, 255), 1)  # рисуем все контуры

cv2.imshow('face_contours_new', img_cont)
cv2.waitKey(1000)

# наращивание
kernel = np.ones((5, 5), np.uint8)
img_cont = cv2.dilate(img_cont, kernel, iterations=1)
cv2.imshow('dilate', img_cont)
cv2.waitKey(1000)

# Сглаживание фольтром Гаусса  и нормализация
img_cont = cv2.GaussianBlur(img_cont, (5, 5), 1)
cv2.imshow('gaussian', img_cont)
cv2.waitKey(1000)
img_norm = cv2.normalize(img_cont, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
cv2.imshow('normalize', img_norm)
cv2.waitKey(1000)

# применение фильтров к обрезанному изабражению лица
bil_img = cv2.bilateralFilter(face_img, 10, 55, 55)

cv2.imshow('bilateral', bil_img)
cv2.waitKey(1000)

# t_t = [15, 15, 15]
# sharp_img = clarity(face_img, t_t, 20)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharp_img = cv2.filter2D(face_img, -1, kernel)
cv2.imshow('sharp', sharp_img)
cv2.waitKey(1000)

fin_img = finaly_filtr(img_norm, bil_img, sharp_img)
cv2.imshow('finaly', fin_img)
cv2.waitKey(1000)

cv2.destroyAllWindows()



