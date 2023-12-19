import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from skimage.io import imread
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

letters = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z'
        ]
height = 100
width = 75

def read_training_data(training_directory):
    image_data_train = []
    target_data_train = []
    image_data_test = []
    target_data_test = []
    list_dir = os.listdir(training_directory)
    for each_letter in tqdm(list_dir):
        print(each_letter)
        img_dirs = os.listdir(training_directory + '/' + each_letter)
        for img in img_dirs[:100]:
            image_path = os.path.join(training_directory, each_letter, img)
            binary_image = imread(image_path, as_gray=True)
            flat_bin_image = binary_image.reshape(-1)
            image_data_train.append(flat_bin_image)
            target_data_train.append(each_letter)
        for img in img_dirs[100:200]:
            image_path = os.path.join(training_directory, each_letter, img)
            binary_image = imread(image_path, as_gray=True)
            flat_bin_image = binary_image.reshape(-1)
            image_data_test.append(flat_bin_image)
            target_data_test.append(each_letter)

    return (np.array(image_data_train), np.array(target_data_train),np.array(image_data_test), np.array(target_data_test))

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")

    print(accuracy_result * 100)

print('reading data')
training_dataset_dir = './dataset'
image_data_train, target_data_train, image_data_test, target_data_test = read_training_data(training_dataset_dir)
print('reading data completed')

# the kernel can be 'linear', 'poly' or 'rbf'
svc_model = SVC(kernel='sigmoid', probability=True)

cross_validation(svc_model, 4, image_data_train, target_data_train)

print('training model')

svc_model.fit(image_data_train, target_data_train)
pred_y = svc_model.predict(image_data_test)
accuracy = accuracy_score(target_data_test, pred_y)
class_re = classification_report(target_data_test, pred_y)

print('accuracy score: ', accuracy)
print('classification report: ', class_re)

import pickle
print("model trained.saving model..")
filename = 'models/finalized_model_sig_1.sav'
pickle.dump(svc_model, open(filename, 'wb'))
print("model saved")