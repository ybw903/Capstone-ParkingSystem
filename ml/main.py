import requests,json
import time
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pytesseract

MAX_DIAG_MULTIPLYER = 5  # 5
MAX_ANGLE_DIFF = 12.0  # 12.0
MAX_AREA_DIFF = 0.5  # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3  # 3

serverURL = "https://qiuutucpaj.execute-api.ap-northeast-2.amazonaws.com/dev"
info = {'idparking': 1001, 'parking_name': 'chungbuk', 'default_bill': {'unit': '분', 'time': 30, 'cost': 3000},
        'add_bill': {'unit': '분', 'time': 30, 'cost': 8000}}
localURL = "http://127.0.0.1:5000"

def find_chars(contour_list, possible_contours):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])


        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # recursive
        recursive_contour_list = find_chars(unmatched_contour,possible_contours)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx

def fetchCarPlate(img):
    height, width, channel = img.shape
    print(height, width, channel)

    # img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ########## 모폴로지 연산 추가?
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(img_gray, imgTopHat)
    img_gray_m = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    ##############

    img_blurred = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred, maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )
    cv2.imshow('window', img_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(
        img_thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    cv2.drawContours(temp_result, contours= contours, contourIdx=-1, color=(255,255,255))
    cv2.imshow('window', temp_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    contours_dict = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    cv2.imshow('window', temp_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)


    result_idx = find_chars(possible_contours, possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255),
                          thickness=2)

    cv2.imshow('ok', temp_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):

        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        cv2.imshow('ok', plate_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > MIN_AREA \
                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))

        chars = pytesseract.image_to_string(img_result, lang='kor2', config='--psm 7')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        print(result_chars)
        if(len(result_chars)>6):
            plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

    print(len(plate_infos))
    info = plate_infos[0]
    chars = plate_chars[0]

    print(chars)
    # return chars
    img_out = img.copy()
    #
    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
                  color=(0, 255, 0), thickness=2)
    fontpath = "font/gulim.ttc"
    font = ImageFont.truetype(fontpath, 30)
    img_pil = Image.fromarray(img_out)
    draw = ImageDraw.Draw(img_pil)
    draw.text((info['x'] + 30, info['y'] - 30), chars, font=font, fill=(0, 255, 0))
    img_out = np.array(img_pil)
    cv2.imshow("result", img_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return chars
###################################################################################################################
def checkBooking(data):
    data['now'] = time.localtime(time.time())
    print(data['now'])
    uri = serverURL + "/booking/check-in"
    print(json.dumps(data))
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    res = requests.post(uri, headers=headers,data=json.dumps(data))
    print(res.status_code)
    if (res.status_code == 200):
        return True
    else:
        return False

def checkOut(data):
    data['now'] = time.localtime(time.time())

    uri = serverURL + "/booking/check-out"
    headers = {'Content-Type': 'application/json; charset=utf-8'}

    res = requests.post(uri,headers=headers,data = json.dumps(data))
    print(res.status_code)
    if(res.status_code ==200):
        print(res.json())
        return {'out': True, 'result':res.json()}



img = cv2.imread('test3.jpg')
if __name__ == '__main__':
    IN = False
    OUT = False
    # fetchCarPlate(img)
    while True:

        '''
        적외선 센서 인식부분

        '''
        n= 0
        n = int(input())
        if n==1:
            IN=True
            n=0
        elif n==2:
            OUT=True
            n=0
        if(IN):
            car_plate = fetchCarPlate(img)
            print(car_plate)
            info['car_plate'] = car_plate
            #if checkBooking(info):
            #   print("어서오세요")
            # else: print("예약 후 이용 부탁드립니다")
            IN=False

        if(OUT):
            info['car_plate'] = fetchCarPlate(img)
            outData = checkOut(info)
            #print(outData)
            #if(outData['out']):
            #    print(outData['result']['cost']+'원 청구')
            #else:
            #    print("retry")
            OUT=False



