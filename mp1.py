# Starter code for CS 165B MP1 Fall 2023
import numpy as np

# Centroid calculation 
def compute_centroid(data_points):
    data = np.array(data_points)
    centroid = np.mean(data, axis=0)
    return centroid

# Discriminant calculation 
def compute_discr(point, centroid1, centroid2):
    midpoint = (centroid1 + centroid2) / 2
    direction = centroid2 - centroid1
    return np.dot(point - midpoint, direction)
    
def classifier(point, centroids):
    #centroids[0] = A
    #centroids[1] = B
    #centroids[2] = C
    discr_AB = compute_discr(point, centroids[0], centroids[1])
    discr_AC = compute_discr(point, centroids[0], centroids[2])
    discr_BC = compute_discr(point, centroids[1], centroids[2])

    # check ties: if A, B || A, C || A, B, C --> return A
    # if B, C --> return B
    if ((discr_AB == 0) or (discr_AC == 0) or (discr_AB == 0 and discr_BC == 0)):
        return 'A'
    elif (discr_BC == 0):
        return 'B'

    # Classification based on discriminant values
    if (discr_AB < 0):
        if discr_AC <= 0: 
            return 'A'
        else: 
            return 'C'
    else:
        if(discr_BC <= 0): 
            return 'B'
        else: 
            return 'C'

def compute_metr(predicted_classes, actual_classes):
    classes = ['A', 'B', 'C']
    total_tpr = 0
    total_fpr = 0
    total_accuracy = 0
    total_precision = 0
    for elem in classes:
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for predicted, actual in zip(predicted_classes, actual_classes):
            if actual == elem:
                if predicted == elem: # true positive
                    TP += 1
                else: # false negative
                    FN += 1
            else: 
                if predicted == elem: # false posistive 
                    FP += 1
                else: # true negative 
                    TN += 1

        #if statements to account for 0 denoms.
        if (TP + FN != 0):
            total_tpr += TP / (TP + FN)
        else:
            total_tpr += 0

        if (FP + TN != 0):
            total_fpr += FP / (FP + TN)
        else:
            total_fpr += 0

        if (TP + FP + FN + TN != 0):
            total_accuracy += (TP + TN) / (TP + FP + FN + TN)
        else: 
            total_accuracy += 0

        if (TP + FP != 0): 
            total_precision += TP / (TP + FP)
        else: 
            total_precision += 0

    # Average the metrics across all classes
    avg_tpr = total_tpr / len(classes)
    avg_fpr = total_fpr / len(classes)
    avg_accuracy = total_accuracy / len(classes)
    avg_precision = total_precision / len(classes)
    avg_error_rate = 1 - avg_accuracy

    return {
        "tpr": avg_tpr,
        "fpr": avg_fpr,
        "error_rate": avg_error_rate,
        "accuracy": avg_accuracy,
        "precision": avg_precision
    }







def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """


    # TODO: IMPLEMENT
    #pass

    # we need to compute discr.s of the training data 
    # 1) centroids stored in dict
    # 2) compare discr.s w centroids 
    # 3) finds the optimal one, continue 
    centroids = []
    predicted_classes = []
    actual_classes = []
    #num_classes_0 should be 3, class_sizes_list_0 holds the training data classifications
    num_classes_0, *class_sizes_list_0 = training_input[0]
    data_index = 1
    for num_data_points in class_sizes_list_0:
        data_points = training_input[data_index:data_index+num_data_points]
        centroids.append(compute_centroid(data_points))
        data_index += num_data_points

    # test data loop 
    num_classes_1, *class_sizes_list_1 = testing_input[0]
    data_index = 1
    for class_index, num_data_points in enumerate(class_sizes_list_1, 1):
        for num_classes_1 in range(num_data_points):
            point = testing_input[data_index]
            predicted_class = classifier(point, centroids)
            predicted_classes.append(predicted_class)
            if class_index == 1:
                actual_class = 'A'
            elif class_index == 2: 
                actual_class = 'B'
            elif class_index == 3: 
                actual_class = 'C'
            actual_classes.append(actual_class)
            data_index += 1

    return compute_metr(predicted_classes, actual_classes)







