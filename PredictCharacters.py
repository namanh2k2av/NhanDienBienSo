import cv2
import numpy as np
import pickle

# model_filename = './finalized_model.sav'
# svc_model = pickle.load(open(model_filename, 'rb'))
letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z'
        ]

def detect_license_plate(image):
    # Lọc ảnh để loại bỏ nhiễu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Tìm kiếm các vùng có độ tương phản cao trong ảnh
    edges = cv2.Canny(blur_image, 50, 150)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=0)

    # Tìm kiếm các vùng có hình dạng giống biển số xe
    contour_list = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_list = contour_list[0]

    # Loại bỏ các vùng không phải biển số xe
    license_plate_contours = []
    for contour in contour_list:
        area = cv2.contourArea(contour)
        
        if area > 5000 :#and area < 15000:
            license_plate_contours.append(contour)
            break


    return license_plate_contours

def detect_character_regions_1(img):

    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để giảm nhiễu và tăng cường ký tự
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Phát hiện cạnh sử dụng Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Áp dụng kỹ thuật morpohology để tìm vùng chứa ký tự
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Tìm contours trên ảnh đã xử lý
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm vùng chứa ký tự dựa trên diện tích và hình dạng
    char_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1100 < area < 4300:  # Thay đổi ngưỡng diện tích tùy thuộc vào đặc điểm của ảnh
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(h) / w
            if 1.5 < aspect_ratio < 7.5:  # Chọn các vùng có tỷ lệ chiều rộng/chiều cao phù hợp
                char_regions.append((x, y, x + w, y + h))
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị ảnh và vùng chứa ký tự
    # cv2.imshow("Anh goc", img)
    # cv2.imshow("Vung chua ky tu", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return char_regions
  
def detect_character_regions_2(img):

    # Chuyển đổi ảnh sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Làm mờ ảnh để giảm nhiễu và tăng cường ký tự
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)

    # Phát hiện cạnh sử dụng Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Áp dụng kỹ thuật morpohology để tìm vùng chứa ký tự
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=0)

    # Tìm contours trên ảnh đã xử lý
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Tìm vùng chứa ký tự dựa trên diện tích và hình dạng
    char_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 1500:  # Thay đổi ngưỡng diện tích tùy thuộc vào đặc điểm của ảnh
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.1 < aspect_ratio < 10:  # Chọn các vùng có tỷ lệ chiều rộng/chiều cao phù hợp
                char_regions.append((x, y, x + w, y + h))
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hiển thị ảnh và vùng chứa ký tự
    # cv2.imshow("Anh goc", img)
    # cv2.imshow("Vung chua ky tu", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return char_regions

def custom_sort(tuple_list):

    sorted_list = sorted(tuple_list, key=lambda x: x[1])
    a = sorted_list[0]
    result = []
    current_group_a = []
    current_group_b = []
    for i in range(len(sorted_list)):
        if abs(a[1] - sorted_list[i][1]) <= 20:
            current_group_a.append(sorted_list[i])
        else:
            current_group_b.append(sorted_list[i])

    current_group_a = sorted(current_group_a, key=lambda x: x[0])
    current_group_b = sorted(current_group_b, key=lambda x: x[0])
    result.extend(current_group_a)
    result.extend(current_group_b)
    return result


def recognize_license_plate(image, model):
    license_plate_contours = detect_license_plate(image)
    result = []
    for i, contour in enumerate(license_plate_contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        license_plate_roi = image[y+2:y+h-2, x+2:x+w-2]
        if license_plate_roi.shape[1] / license_plate_roi.shape[0] > 2.5:
            height = 60
            scale_percent = height / license_plate_roi.shape[0]
            width = int(license_plate_roi.shape[1] * scale_percent)
            license_plate_roi = cv2.resize(license_plate_roi,(width, height))
            # cv2.imshow('a',license_plate_roi)
            # cv2.waitKey(0)
            a = detect_character_regions_2(license_plate_roi)
        else:
            height = 200
            scale_percent = height / license_plate_roi.shape[0]
            width = int(license_plate_roi.shape[1] * scale_percent)
            license_plate_roi = cv2.resize(license_plate_roi,(width, height))
            # cv2.imshow('a',license_plate_roi)
            # cv2.waitKey(0)
            a = detect_character_regions_1(license_plate_roi)
        sorted_coordinates = custom_sort(a)
            
        for counter_idx in sorted_coordinates:
            x1, y1, x2, y2 = counter_idx
            img_pred = license_plate_roi[y1-2:y2+2, x1-2:x2+2]
            
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
            img_pred = cv2.resize(img_pred, (75, 100))
            # cv2.imshow('b',img_pred)
            # cv2.waitKey(0)
            flat_bin_image = img_pred.reshape(-1)
            pred = model.predict([flat_bin_image])
            # print(letters[int(pred)])
            result.append(pred[0])
    result = ''.join(result)
    ht = ''
    if len(result) == 9:
        ht = result[:2] + '-' + result[2:4] + ' ' + result[4:]
    else:
        ht = result[:3] + '-' + result[3:]
    return ht


# if __name__ == "__main__":
#     image = cv2.imread("C:\\Project\\KhaiPhaDuLieu\\bien6.jpg")
#     image = cv2.resize(image, (780, 500))
#     predictions = recognize_license_plate(image, svc_model)
#     print(predictions)
#     cv2.imshow('img',image)
#     cv2.waitKey(0)
    