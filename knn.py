import math
import csv

def preprocess(data, flag):
    """
    label encoding for the discrete features
    """
    for i in range(len(data)):
        if i != 0 and flag == 0:
            data[i][0] = 1 if data[i][0] == 'Male' else 0 # gender
            data[i][1] = 1 if data[i][1] == 'Yes' else 0 # SeniorCitizen
            data[i][2] = 1 if data[i][2] == 'Yes' else 0 # Partner
            data[i][3] = 1 if data[i][3] == 'Yes' else 0 # Dependents
            data[i][4] = float(data[i][4])
            data[i][5] = 1 if data[i][5] == 'Yes' else 0 # PhoneService

            # MultipleLines
            if data[i][6] == 'Yes':
                data[i][6] = 2
            elif data[i][6] == 'No':
                data[i][6] = 1
            else:
                data[i][6] = 0
            
            # InternetService
            if data[i][7] == 'Fiber optic':
                data[i][7] = 2
            elif data[i][7] == 'DSL':
                data[i][7] = 1
            else:
                data[i][7] = 0
            
            # OnlineSecurity
            if data[i][8] == 'Yes':
                data[i][8] = 2
            elif data[i][8] == 'No':
                data[i][8] = 1
            else:
                data[i][8] = 0
            
            # OnlineBackup
            if data[i][9] == 'Yes':
                data[i][9] = 2
            elif data[i][9] == 'No':
                data[i][9] = 1
            else:
                data[i][9] = 0
            
            # DeviceProtection
            if data[i][10] == 'Yes':
                data[i][10] = 2
            elif data[i][10] == 'No':
                data[i][10] = 1
            else:
                data[i][10] = 0
            
            # TechSupport
            if data[i][11] == 'Yes':
                data[i][11] = 2
            elif data[i][11] == 'No':
                data[i][11] = 1
            else:
                data[i][11] = 0
            
            # StreamingTV
            if data[i][12] == 'Yes':
                data[i][12] = 2
            elif data[i][12] == 'No':
                data[i][12] = 1
            else:
                data[i][12] = 0
            
            # StreamingMovies
            if data[i][13] == 'Yes':
                data[i][13] = 2
            elif data[i][13] == 'No':
                data[i][13] = 1
            else:
                data[i][13] = 0

            # Contract
            if data[i][14] == 'One year':
                data[i][14] = 2
            elif data[i][14] == 'Two year':
                data[i][14] = 1
            else:
                data[i][14] = 0
            
            # PaperlessBilling
            if data[i][15] == 'Yes':
                data[i][15] = 1
            else:
                data[i][15] = 0
            
            # PaymentMethod
            if data[i][16] == 'Electronic check':
                data[i][16] = 3
            elif data[i][16] == 'Bank transfer (automatic)':
                data[i][16] = 2
            elif data[i][16] == 'Credit card (automatic)':
                data[i][16] = 1
            else:
                data[i][16] = 0

            data[i][17] = float(data[i][17])
            data[i][18] = float(data[i][18])

        elif i != 0 and flag == 1:
            data[i][0] = 1 if data[i][0] == 'Yes' else 0 # Churn

    return data

def normalization(data):
    """
    for the dataset, do the normalization
    """
    col_means = []
    col_stds = []
    normalized_data = []
    
    for col in range(len(data[0])):
        # calculate column means
        col_values = [row[col] for row in data]
        col_mean = sum(col_values) / len(col_values)
        col_means.append(col_mean)

        # calculate column stds
        variance = sum((value - col_mean) ** 2 for value in col_values) / len(col_values)
        col_std = math.sqrt(variance)
        col_stds.append(col_std)
    
    # normalization
    for row in data:
        normalized_row = [(x - col_means[i]) / col_stds[i] for i, x in enumerate(row)]
        normalized_data.append(normalized_row)
    
    return normalized_data

def pearson_correlation(x, y):
    """
    calculate the pearson correlation between x and y
    """
    n = len(x)
    meanx = sum(x) / n
    meany = sum(y) / n
    
    covariance = sum((x[i] - meanx) * (y[i] - meany) for i in range(n))
    stdx = (sum((x[i] - meanx) ** 2 for i in range(n))) ** 0.5
    stdy = (sum((y[i] - meany) ** 2 for i in range(n))) ** 0.5
    
    std = stdx * stdy
    
    if std == 0:
        return 0
    return covariance / std

def feature_selection(data, target_feature, threshold=0.15):
    """
    feature selection based on pearson correlation
    data: dataset
    target_feature: the label's(Churn) index, we want to select the features that has high correlation with target feature
    threshold: the threshold of correlation between two features, If below threshold remove the feature
    return: the selected features index
    """
    selected_features = []
    
    for feature_index in range(len(data[0])-1):
        feature = [row[feature_index] for row in data]
        target = [row[target_feature] for row in data]
        correlation = pearson_correlation(feature, target)
        
        if abs(correlation) > threshold:
            selected_features.append(feature_index)
    
    return selected_features
    
# Euclidean Distance
def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

# use Gaussian function to calculate the weight
def gaussian(dist, a=1, b=0, c=9):
    return a * math.e ** (-(dist-b) ** 2 / (2 * c ** 2))

# class KNN (weighted KNN)
class KNN:
    def __init__(self, k=47):
        self.k = k
        self.train_data = []
        self.train_labels = []
    
    def fit(self, data, labels):
        self.train_data = data
        self.train_labels = labels
    
    # prediction function
    def predict(self, point):
        # calculate the distances between target point and training points
        distances = []
        for i, train_point in enumerate(self.train_data):
            dist = euclidean_distance(train_point, point)
            distances.append((dist, self.train_labels[i]))
        
        # sort the distances in the ascending order
        distances.sort(key=lambda x: x[0])
        
        # select the nearliest k neighbors
        nearest_neighbors = distances[:self.k]
        
        # get the labels of k neighbors
        labels = [label for _, label in nearest_neighbors]
        labels = [ label[0] for label in labels]

        # each neighbor adopt Gaussian function to calculate its weight
        weights = [gaussian(dist) for dist, _ in nearest_neighbors]

        yes_weight = 0
        no_weight = 0
        for i, label in enumerate(labels):
            if label == 1:
                yes_weight += weights[i]
            else:
                no_weight += weights[i]

        if yes_weight >= no_weight:
            return 1
        else:
            return 0

if __name__ == "__main__":
    
    with open('train.csv', newline='') as trainfile:
        train_data = csv.reader(trainfile)
        train_data = list(train_data)
    
    with open('train_gt.csv', newline='') as train_labelfile:
        train_labels = csv.reader(train_labelfile)
        train_labels = list(train_labels)

    train_data = preprocess(train_data, 0) # feature preprocess
    train_labels = preprocess(train_labels, 1) # labelpreprocess
    train_data = train_data[1:]
    train_labels = train_labels[1:]
    train_data = normalization(train_data) # normalization
    for i in range(len(train_data)):
        train_data[i].append(train_labels[i][0])
    selected_features = feature_selection(train_data,19,0.1) # feature selection based on pearson correlation target_feature = Churn
    for i in range(len(train_data)):
        train_data[i] = train_data[i][:-1]
    train_data = [[row[col]for col in selected_features] for row in train_data] # use the selected features as new dataset

    """
    create KNN object with k neighbors, k's value selection is usually picked as sqrt(# of samples)
    In this case sqrt(2242) = 47.34... 
    """
    model = KNN(k=47) 
    
    # training model
    model.fit(train_data, train_labels)

    with open('val.csv', newline='') as valfile:
        val_data = csv.reader(valfile)
        val_data = list(val_data)
    
    with open('val_gt.csv', newline='') as val_labelfile:
        val_labels = csv.reader(val_labelfile)
        val_labels = list(val_labels)
    
    val_data = preprocess(val_data, 0) # feature preprocess
    val_labels = preprocess(val_labels, 1) # label preprocess
    val_data = val_data[1:]
    val_data = normalization(val_data) # normalization
    val_data = [[row[col] for col in selected_features] for row in val_data] # use the selected features as new dataset

    with open('test.csv', newline='') as testfile:
        test_data = csv.reader(testfile)
        test_data = list(test_data)

    test_data = preprocess(test_data, 0) # feature preprocess
    test_data = test_data[1:]
    test_data = normalization(test_data) # normalization
    test_data = [[row[col] for col in selected_features] for row in test_data] # use the selected features as new dataset

    # prediction
    val_pred = []
    test_pred = []
    for i in range(len(val_data)):
        val_pred.append(model.predict(val_data[i]))
    
    for i in range(len(test_data)):
        test_pred.append(model.predict(test_data[i]))

    with open('val_pred.csv', 'w', newline='') as val_predfile:
        writer = csv.writer(val_predfile)
        writer.writerow(['Churn'])
        for i in range(len(val_pred)):
            if val_pred[i] == 1:
                writer.writerow(['Yes'])
            else:
                writer.writerow(['No'])
    
    with open('test_pred.csv', 'w', newline='') as test_predfile:
        writer = csv.writer(test_predfile)
        writer.writerow(['Churn'])
        for i in range(len(test_pred)):
            if test_pred[i] == 1:
                writer.writerow(['Yes'])
            else:
                writer.writerow(['No'])