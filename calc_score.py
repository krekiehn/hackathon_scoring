import json
import numpy as np
case_key = 'image'
prediction_key = 'prediction'


def read_json(ground_truth=r'ground_truth.json', prediction=r'example_prediction.json'):
    with open(ground_truth) as f:
        ground_truth_json = json.load(f)
    with open(prediction) as f:
        prediction_json = json.load(f)
    return ground_truth_json, prediction_json


def calc_score(ground_truth_json: list, prediction_json: list):
    # build for groups of error, one for every class (0,1,2,3)
    class_error = {0: [], 1: [], 2: [], 3: []}
    class_mean_error = {0: 0, 1: 0, 2: 0, 3: 0}

    # calc square error for every case:
    for i in range(len(ground_truth_json)):
        file_name = ground_truth_json[i][case_key]
        # find some case in prediction, maybe not the same order
        for j in range(len(prediction_json)):
            if prediction_json[j][case_key] == file_name:
                label = int(ground_truth_json[i][prediction_key])
                error_square = (float(ground_truth_json[i][prediction_key]) - float(prediction_json[j][prediction_key]))**2
                print(ground_truth_json[i][prediction_key])
                print(prediction_json[j][prediction_key])
                # add error to group of errors corresponding to the ground-truth label
                class_error[label].append(error_square)
                break
            # last possible prediction for the actual case
            elif j == len(prediction_json) - 1:
                # no prediction for actual cases founded: append max error 3**2 = 9
                label = int(ground_truth_json[i][prediction_key])
                error_square = 3**2
                # add error to group of errors corresponding to the ground-truth label
                class_error[label].append(error_square)
                break

    # calc mean of class square error per class (independent on the number of cases):
    for label in class_error.keys():
        error_arr = np.array(class_error[label])
        error_mean = error_arr.mean()
        class_mean_error[label] = error_mean
    # calc mean of class mean square error:
    error_mean_mean = 0
    for label in class_mean_error.keys():
        error_mean_mean += class_mean_error[label]
    error_mean_mean = error_mean_mean/len(class_mean_error.keys())

    return error_mean_mean



