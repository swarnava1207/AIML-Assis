#################### Q1 : PERCEPTRON ####################

import oracle
import numpy as np
import matplotlib.pyplot as plt

data = oracle.q1_get_cifar100_train_test(23746)
train_data = data[0]
test_data = data[1]

print("Test data : ", test_data[0])
train_features = np.array([x[0] for x in train_data])
train_labels = np.array([x[1] for x in train_data])
test_features = np.array([x[0] for x in test_data])
test_labels = np.array([x[1] for x in test_data])   

print(train_features.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_labels.shape)
def perceptron(train_features, train_labels, iterations):
    w = np.zeros(train_features.shape[1])
    b = 0
    i = 0
    while i < iterations:
        #print(i)
        flag = 0
        for j in range(train_features.shape[0]):
            if train_labels[j] * (np.dot(w, train_features[j]) + b) <= 0:
                w += train_labels[j] * train_features[j]
                b += train_labels[j]
                flag = 1
        if flag == 0:
            break
        i += 1 
        

    #print("Iterations done: ", i)
    return w,i
# w, b, i = perceptron(train_features, train_labels, 100000000000)
# print("iterations: ", i)
def test_perceptron(iterations,test_features, test_labels):
    w, i = perceptron(train_features, train_labels, iterations)
    misclassified = 0
    for j in range(test_features.shape[0]):
        if test_labels[j] * (np.dot(w, test_features[j])) <= 0:
            misclassified += 1
    return misclassified/test_features.shape[0]
iterations = [i * 1000 for i in range(1, 20)]
error = []
for i in iterations:
    print(i)
    error.append(test_perceptron(i, test_features, test_labels))
    
# plt.plot(iterations, error)
# plt.xlabel('Iterations')
# plt.ylabel('Error')
# plt.title('Error vs Iterations')
# plt.show()

from cvxopt import matrix, solvers
def SVM_with_slack(train_features, train_labels, C):
    n = train_features.shape[0]
    m = train_features.shape[1]
    P = np.zeros((m+1+n, m+1+n))
    for i in range(m):
        P[i][i] = 1
    q = np.zeros(m+1+n)
    for i in range(m+1, m+1+n):
        q[i] = C
    h = np.zeros(2*n)
    for i in range(n):
        h[i] = -1
    G = np.zeros((2*n, m+1+n))
    for i in range(n):
        for j in range(m):
            G[i][j] = -train_labels[i] * train_features[i][j]
        G[i][m] = -train_labels[i]
        G[i][m+1+i] = -1
    for i in range(n):
        G[n+i][m+1+i] = -1
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    w = np.array(sol['x'][:m])
    b = sol['x'][m]
    w = w.reshape((m,))
    return w, b

def SVM_with_slack_dual(train_features, train_labels, C):
    n = train_features.shape[0]
    m = train_features.shape[1]
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            P[i][j] = train_labels[i] * train_labels[j] * np.dot(train_features[i], train_features[j])
    P = matrix(P)
    q = -1 * np.ones(n)
    q = matrix(q)
    G = np.zeros((2*n, n))
    for i in range(n):
        G[i][i] = -1
        G[n+i][i] = 1
    h = np.zeros(2*n)
    for i in range(n):
        h[n+i] = C
    G = matrix(G)
    h = matrix(h)
    A = np.zeros(n)
    for i in range(n):
        A[i] = train_labels[i]
    A = matrix(A, (1, n))
    b = matrix(0.0)
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alpha = np.array(sol['x'])
    w = np.zeros(m)
    for i in range(n):
        w += alpha[i] * train_labels[i] * train_features[i]
    b = 0
    for i in range(n):
        if alpha[i] > 0:
            b = train_labels[i] - np.dot(w, train_features[i])
            break
    return w, b
import time
C = 170
start = time.time()
w, b = SVM_with_slack(train_features, train_labels, C)
end = time.time()
print("Time taken by primal SVM: ", end-start)
start = time.time()
w, b = SVM_with_slack_dual(train_features, train_labels, C)
end = time.time()
print("Time taken by dual SVM: ", end-start)    
train_features_separable = []
train_labels_separable = []
C = 170
w,b = SVM_with_slack(train_features, train_labels, C)
for i in range(train_features.shape[0]):
    if train_labels[i] * (np.dot(w, train_features[i]) + b) > 0:
        train_features_separable.append(train_features[i])
        train_labels_separable.append(train_labels[i])
train_features_separable = np.array(train_features_separable)
train_labels_separable = np.array(train_labels_separable)
print(train_features_separable.shape)
print(train_labels_separable.shape) 
inseparable_indices = []
for i in range(train_features.shape[0]):
    if train_labels[i] * (np.dot(w, train_features[i]) + b) <= 0:
        inseparable_indices.append(i)
print(inseparable_indices)
np.savetxt('inseparable_23746.csv', inseparable_indices, delimiter=',')
print(inseparable_indices.shape)
def test_svm(test_features, test_labels,c):
    w, b = SVM_with_slack(train_features_separable, train_labels_separable, c)
    misclassified = 0
    for j in range(test_features.shape[0]):
        if test_labels[j] * (np.dot(w, test_features[j]) + b) <= 0:
            misclassified += 1
    return misclassified/test_features.shape[0]
c = [i * 10 for i in range(11, 15)]
c = [i+0.5 for i in range(112,120)]
c = [120]
error = []
for i in c:
    error.append(test_svm(test_features, test_labels, i))
import cvxopt.solvers

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

def train_kernelized_svm(X, y, C=1.0, gamma=0.5):
    n_samples, n_features = X.shape
    K = compute_kernel_matrix(X, gamma)
    
    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-np.ones(n_samples))
    A = cvxopt.matrix(y, (1, n_samples), 'd')
    b = cvxopt.matrix(0.0)
    G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
    h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * C)))
    
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alphas = np.ravel(solution['x'])
    
    support_vectors = alphas > 1e-5
    sv_X = X[support_vectors]
    sv_y = y[support_vectors]
    sv_alphas = alphas[support_vectors]
    
    b = np.mean([y_k - np.sum(sv_alphas * sv_y * K[support_vectors, k]) 
                 for k, y_k in enumerate(sv_y)])
    
    return sv_X, sv_y, sv_alphas, b, gamma

def test_kernelized(train_features,train_labels, gamma, C):
    sv_X, sv_y, sv_alphas, b, gamma = train_kernelized_svm(train_features, train_labels, C, gamma)
    misclassified = 0
    for j in range(test_features.shape[0]):
        prediction = b
        for i in range(sv_X.shape[0]):
            prediction += sv_alphas[i] * sv_y[i] * gaussian_kernel(sv_X[i], test_features[j], gamma)
        if prediction * test_labels[j] <= 0:
            misclassified += 1
    return misclassified/test_features.shape[0]


gammas = [0.1, 0.5, 1, 2, 5]
C_values = [0.1, 0.5, 1, 2, 5]
error = []
for gamma in gammas:
    for C in C_values:
        error.append(test_kernelized(train_features_separable, train_labels_separable, gamma, C))
print(error)


###################### Q2 : Logistic Regression, MLP and PCA #####################

import oracle

oracle.q2_get_mnist_jpg_subset(23746)
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split    
from sklearn.preprocessing import StandardScaler
data_dir = 'q2_data'
images = []
for i in range(10):
    folder = os.path.join(data_dir, str(i))
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        images.append((img,i))





X = np.array([img.flatten() for img, label in images])
y = np.array([label for img, label in images])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)


class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = MLP()


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset = MNISTDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)


model.train()
for epoch in range(5):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')


X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)  
print(X_test.shape)
print(y_test.shape)  
model.eval()
y_pred = model(X_test)
y_pred = torch.argmax(y_pred, dim=1)
num_correct = torch.sum(y_pred == y_test)
num_total = len(y_test)
confusion_matrix = np.zeros((10, 10))
for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred[i]] += 1 


print(f'Accuracy: {num_correct / num_total}')
print("Recall: ", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
print("Precision: ", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))
print("F1 Score: ", 2 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0)))
print("Confusion Matrix: ")
print(confusion_matrix)







data_dir = 'q2_data'
images = []

for i in range(10):
    folder = os.path.join(data_dir, str(i))
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        images.append((img, i)) 



X = np.array([img.flatten() for img, label in images])
y = np.array([label for img, label in images])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

class MNISTDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

model = CNN()

criterion = nn.CrossEntropyLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset = MNISTDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

model.train()
for epoch in range(5):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')


X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
model.eval()
y_pred = model(X_test)
y_pred = torch.argmax(y_pred, dim=1)
num_correct = torch.sum(y_pred == y_test)
num_total = len(y_test)
confusion_matrix = np.zeros((10, 10))
for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred[i]] += 1

print(f'Accuracy: {num_correct / num_total}')
print("Recall: ", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
print("Precision: ", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))
print("F1 Score: ", 2 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0)))
print("Confusion Matrix: ")
print(confusion_matrix)






data_dir = 'q2_data'
images = []



for i in range(10):
    folder = os.path.join(data_dir, str(i))
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        images.append((img, i)) 
np.random.shuffle(images)

def PCA(k):
    
    y = []
    for img, label in images:
        X.append(img.flatten())
        y.append(label)

    X = np.array(X)
    y = np.array(y)


    scaler = StandardScaler()
    X = scaler.fit_transform(X)

   
    cov = np.cov(X.T)

    eigvals, eigvecs = np.linalg.eig(cov)


    sorted_indices = np.argsort(eigvals)[::-1]  
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]    

   
    eigvecs = eigvecs[:, :k]
    print(eigvecs.shape)
    print(X.shape)


    X_pca = X.dot(eigvecs)
    print(X_pca.shape)
 
    X_reconstructed = X_pca.dot(eigvecs.T)
    print(X_reconstructed.shape)
    return X_pca, X_reconstructed




k = [10,20,50,100,200,500]

for i in k:
    X_pca, X_reconstructed = PCA(i)
    
    index = 0
    plt.imshow(X_reconstructed[index].reshape(28, 28), cmap='gray')
    plt.title('Reconstructed Image with k = ' + str(i))
    plt.show()
plt.imshow(images[0][0].reshape(28, 28), cmap='gray')
plt.title('Original Image')



X_pca, X_reconstructed = PCA(100)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


class Dataset_PCA(Dataset):
    def __init__(self, X, y):
        self.X = X.float() 
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    


class MLP_PCA(nn.Module):
    def __init__(self):
        super(MLP_PCA, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

model = MLP_PCA()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = Dataset_PCA(torch.tensor(X_train), torch.tensor(y_train)) 
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

num_epochs = 5
losses = []
for epoch in range(num_epochs):
    loss_sum = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')

        loss_sum += loss.item()
    losses.append(loss_sum)

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)
model.eval()
y_pred = model(X_test)
y_pred = torch.argmax(y_pred, dim=1)
num_correct = torch.sum(y_pred == y_test)
num_total = len(y_test)
confusion_matrix = np.zeros((10, 10))
for i in range(len(y_test)):
    confusion_matrix[y_test[i], y_pred[i]] += 1


print(f'Accuracy: {num_correct / num_total}')
print("Recall: ", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))
print("Precision: ", np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0))
print("F1 Score: ", 2 * np.diag(confusion_matrix) / (np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0)))
print("Confusion Matrix: ")
print(confusion_matrix)


X_pca,X_reconstructed = PCA(100)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

class LogisticRegressionOVR:
    def __init__(self, num_classes=10, learning_rate=0.1, epochs=2):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.num_classes, n_features))
        self.bias = np.zeros(self.num_classes)
        
        for i in range(self.num_classes):
            y_binary = (y == i).astype(np.float32)
            for _ in range(self.epochs):
                linear_model = np.dot(X, self.weights[i]) + self.bias[i]
                y_pred = self.sigmoid(linear_model)
                error = y_pred - y_binary
                self.weights[i] -= self.learning_rate * np.dot(error, X) / n_samples
                self.bias[i] -= self.learning_rate * np.mean(error)
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights.T) + self.bias
        return self.sigmoid(linear_model)

    def confusion_matrix(self, y_true, y_pred):
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for i in range(len(y_true)):
            confusion_matrix[y_true[i], y_pred[i]] += 1
        return confusion_matrix


clf = LogisticRegressionOVR()
clf.train(X_train, y_train)
y_scores = clf.predict_proba(X_test)

def compute_roc_auc(y_true, y_scores, num_classes=10):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        sorted_indices = np.argsort(-y_scores[:, i])
        y_sorted = y_binary[sorted_indices]
        tpr = np.cumsum(y_sorted) / np.sum(y_sorted)
        fpr = np.cumsum(1 - y_sorted) / np.sum(1 - y_sorted)
        auc_score = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})')

    recall = np.diag(clf.confusion_matrix(y_test, np.argmax(y_scores, axis=1)))/np.sum(clf.confusion_matrix(y_test, np.argmax(y_scores, axis=1)), axis=1)
    precision = np.diag(clf.confusion_matrix(y_test, np.argmax(y_scores, axis=1)))/np.sum(clf.confusion_matrix(y_test, np.argmax(y_scores, axis=1)), axis=0)
    print("Recall: ", recall)
    print("Precision: ", precision)
    F1 = 2 * recall * precision / (recall + precision)
    print("F1 Score: ", F1)
    accuracy = np.sum(np.diag(clf.confusion_matrix(y_test, np.argmax(y_scores, axis=1))) / len(y_test))
    print("Accuracy: ", accuracy)
    print("Confusion Matrix: ")
    print(clf.confusion_matrix(y_test, np.argmax(y_scores, axis=1)))
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Multi-Class Logistic Regression')
    plt.legend(loc='lower right')
    plt.show()

compute_roc_auc(y_test, y_scores)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_function(y, y_pred):
    return -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))

def gradient_descent(X, y, lr=0.1, epochs=50):
    m, n = X.shape
    theta = np.zeros(n)
    for _ in range(epochs):
        z = np.dot(X, theta)
        y_pred = sigmoid(z)
        gradient = np.dot(X.T, (y_pred - y)) / m
        theta -= lr * gradient
    return theta

def predict(X, theta):
    print(X.shape, theta.shape)
    return sigmoid(np.dot(X, theta))

def confusion_matrix(y_true, y_pred):
    num_classes = len(np.unique(y_true))
    matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        matrix[y_true[i], y_pred[i]] += 1
    return matrix

def train_logistic_regression_ova(X_train, y_train, X_test, y_test):
    num_classes = 10
    m, n = X_train.shape
    X_train = np.hstack([np.ones((m, 1)), X_train])  
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    classifiers = []
    roc_aucs = []
    
    for digit in range(num_classes):
        y_binary = (y_train == digit).astype(int)
        theta = gradient_descent(X_train, y_binary)
        classifiers.append(theta)
        
        y_test_binary = (y_test == digit).astype(int)
        y_score = predict(X_test, theta)
        
        fpr, tpr, _ = roc_curve(y_test_binary, y_score)
        auc_score = auc(fpr, tpr)
        roc_aucs.append(auc_score)
        
        plt.plot(fpr, tpr, label=f'Class {digit} (AUC = {auc_score:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for One-vs-All Logistic Regression')
    plt.legend()
    plt.show()
    
    return classifiers, roc_aucs

classifiers, roc_aucs = train_logistic_regression_ova(X_train, y_train, X_test, y_test)

y_preds = []
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
for classifier in classifiers:
   
    y_pred = predict(X_test, classifier)
    y_preds.append(y_pred)
y_preds = np.array(y_preds)
y_pred = np.argmax(y_preds, axis=0)
confusion_matrix(y_test, y_pred)    

recall = np.diag(confusion_matrix(y_test, y_pred))/np.sum(confusion_matrix(y_test, y_pred), axis=1)
precision = np.diag(confusion_matrix(y_test, y_pred))/np.sum(confusion_matrix(y_test, y_pred), axis=0)
print("Recall: ", recall)
print("Precision: ", precision)
F1 = 2 * recall * precision / (recall + precision)
print("F1 Score: ", F1)
accuracy = np.sum(np.diag(confusion_matrix(y_test, y_pred)) / len(y_test))
print("Accuracy: ", accuracy)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


############################ Q3 : REGRESSION #########################

import oracle 
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

X_train, y_train, X_test, y_test = oracle.q3_linear_1(23746)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def linear_regression(X, y):
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

def predict(X, w):
    return X @ w

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

w = linear_regression(X_train, y_train)
print(w)
y_pred = predict(X_test, w)
print("Test data :",mse(y_test, y_pred))
y_pred = predict(X_train, w)
print("Train data :",mse(y_train, y_pred))



def Ridge_regression(X, y, alpha):
    w = np.linalg.inv(X.T @ X + alpha*np.eye(X.shape[1])) @ X.T @ y
    return w

def predict(X, w):
    return X @ w

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

w = Ridge_regression(X_train, y_train, 1)
print(w)

y_pred = predict(X_test, w)
print("Test data :",mse(y_test, y_pred))
y_pred = predict(X_train, w)
print("Train data :",mse(y_train, y_pred))


import oracle 
import numpy as np
import matplotlib.pyplot as plt
X_train, y_train, X_test, y_test = oracle.q3_linear_2(23746)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
w = linear_regression(X_train, y_train)
print(w)
np.savetxt("w_ols_23746.csv", w, delimiter=",")
y_pred = predict(X_test, w)
print("Test data :",mse(y_test, y_pred))
y_pred = predict(X_train, w)
print("Train data :",mse(y_train, y_pred))

w = Ridge_regression(X_train, y_train, 1)
print(w)
np.savetxt("w_rr_23746.csv", w, delimiter=",")
y_pred = predict(X_test, w)
print("Test data :",mse(y_test, y_pred))
y_pred = predict(X_train, w)
print("Train data :",mse(y_train, y_pred))
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
import oracle

datafile = oracle.q3_stocknet(23746)

dataframe = pd.read_csv(f'stocknet-dataset/price/raw/{datafile}.csv')
dataframe = dataframe['Close']
dataframe_original = dataframe.copy()
data_normalizer = StandardScaler()
dataframe = data_normalizer.fit_transform(dataframe.values.reshape(-1, 1))
dataframe = dataframe.flatten()

def prepare_dataset(dataframe, window_size):
    num_samples = len(dataframe)
    feature_matrix = np.zeros((num_samples - window_size, window_size))
    for idx in range(num_samples - window_size):
        feature_matrix[idx] = dataframe[idx:idx + window_size]
    target_vector = dataframe[window_size:]
    split_idx = int(0.8 * len(feature_matrix))
    return (feature_matrix[:split_idx], feature_matrix[split_idx:],
            target_vector[:split_idx], target_vector[split_idx:])

def visualize_predictions(X_train, Y_train, X_test, Y_test, weight_vector, bias_term, window_size, dataframe_original, data_normalizer):
    Y_predicted_scaled = X_test @ weight_vector + bias_term
    Y_test_original = data_normalizer.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    Y_predicted_scaled = -Y_predicted_scaled  
    min_val, max_val = dataframe_original.min(), dataframe_original.max()
    Y_predicted = min_val + (Y_predicted_scaled - Y_predicted_scaled.min()) * (max_val - min_val) / (Y_predicted_scaled.max() - Y_predicted_scaled.min())

    moving_avg = []
    train_end_idx = len(Y_train)
    for idx in range(len(Y_test)):
        start_idx = max(0, train_end_idx - window_size + idx)
        end_idx = train_end_idx + idx
        moving_avg.append(np.mean(dataframe_original[start_idx:end_idx]))

    moving_avg = np.array(moving_avg)
    plt.figure(figsize=(14, 6))
    plt.plot(Y_test_original, label='Actual Closing Price', color='blue', linewidth=0.7)
    plt.plot(Y_predicted, label='Predicted Closing Price', color='green', linestyle='--', linewidth=1)
    plt.plot(moving_avg, label=f'Previous {window_size}-Day Average', color='orange', linestyle='-.', linewidth=1.5)
    plt.xlabel('Time (Days)')
    plt.ylabel('Price (Original Scale)')
    plt.title(f'Linear SVR Prediction vs Actual vs Previous {window_size}-Day Average')
    plt.ylim(0, max(np.max(Y_test_original), np.max(Y_predicted), np.max(moving_avg)) * 1.05)
    plt.legend()
    plt.grid(True)
    plt.show()

def train_linear_svr(feature_matrix, target_vector, reg_param=1):
    epsilon = 0.1
    num_samples, num_features = feature_matrix.shape
    gram_matrix = feature_matrix @ feature_matrix.T
    
    quadratic_matrix = np.block([
        [gram_matrix, -gram_matrix],
        [-gram_matrix, gram_matrix]
    ])
    linear_term = np.hstack([epsilon + target_vector, epsilon - target_vector])
    inequality_matrix = np.vstack([
        np.eye(2 * num_samples),
        -np.eye(2 * num_samples)
    ])
    inequality_vector = np.hstack([reg_param * np.ones(2 * num_samples), np.zeros(2 * num_samples)])
    equality_matrix = np.hstack([np.ones(num_samples), -np.ones(num_samples)]).reshape(1, -1)
    equality_vector = np.array([0.0])

    quadratic_matrix = matrix(quadratic_matrix)
    linear_term = matrix(linear_term)
    inequality_matrix = matrix(inequality_matrix)
    inequality_vector = matrix(inequality_vector)
    equality_matrix = matrix(equality_matrix)
    equality_vector = matrix(equality_vector)

    solvers.options['show_progress'] = False
    solution = solvers.qp(quadratic_matrix, linear_term, inequality_matrix, inequality_vector, equality_matrix, equality_vector)
    
    alpha_vals = np.array(solution['x']).flatten()[:num_samples]
    alpha_star_vals = np.array(solution['x']).flatten()[num_samples:]
    weight_vector = np.sum((alpha_vals - alpha_star_vals).reshape(-1, 1) * feature_matrix, axis=0)
    bias_term = np.mean(target_vector - feature_matrix @ weight_vector)
    return weight_vector, bias_term

window_sizes = [7, 30, 90]
regularization_val = 170
for window_size in window_sizes:
    X_train, X_test, Y_train, Y_test = prepare_dataset(dataframe, window_size)
    weight_vector, bias_term = train_linear_svr(X_train, Y_train, regularization_val)
    visualize_predictions(X_train, Y_train, X_test, Y_test, weight_vector, bias_term, window_size, dataframe_original, data_normalizer)


import numpy as np
import cvxopt

def RBF_kernel(x1, x2, gamma):
    squared_distance = np.sum(x1**2, axis=1).reshape(-1, 1) + np.sum(x2**2, axis=1) - 2 * np.dot(x1, x2.T)
    return np.exp(-gamma * squared_distance)

def RBF_solve(X_train, Y_train, gamma, C=1.0, epsilon=0.1):
    n = len(Y_train)
    K = RBF_kernel(X_train, X_train, gamma)

    
    P = np.block([[K, -K], [-K, K]])  
    q = np.hstack([epsilon + Y_train, epsilon - Y_train])

    G = np.vstack([
        np.eye(2 * n), 
        -np.eye(2 * n)
    ])

    h = np.hstack([
        np.ones(2 * n) * C,  
        np.zeros(2 * n)      
    ])

    A = np.hstack([np.ones(n), -np.ones(n)]).reshape(1, -1)
    b = np.array([0.0])


    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    cvxopt.solvers.options['show_progress'] = False
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)


    alphas = np.array(solution['x']).flatten()
    alpha_val = alphas[:n]
    alpha_star = alphas[n:]


    w = np.dot((alpha_val - alpha_star), K)


    support_vector_indices = np.where((alpha_val > 1e-5) | (alpha_star > 1e-5))[0]

    if len(support_vector_indices) > 0:
        b = np.mean(
            Y_train[support_vector_indices] 
            - np.dot((alpha_val - alpha_star), K[:, support_vector_indices])
        )
    else:
        b = 0.0

    return alpha_val, alpha_star, w, b

def visualise_RBF(X_train, Y_train, X_test, Y_test, alpha_val, alpha_star, b, gamma, t, dataframe_original, scaler): 
   
    K_test = RBF_kernel(X_train, X_test, gamma)


    Y_pred_scaled = np.dot((alpha_val - alpha_star), K_test) + b
    Y_pred_scaled = -Y_pred_scaled


    Y_pred = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).flatten()
    Y_test_original = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()


    prev_avg = []
    train_end = len(Y_train)

    for i in range(len(Y_test)):
        avg_window_start = max(0, train_end - t + i)
        avg_window_end = train_end + i
        prev_avg.append(np.mean(dataframe_original[avg_window_start:avg_window_end]))

    prev_avg = np.array(prev_avg)


    plt.figure(figsize=(14, 6))
    
    plt.plot(Y_test_original, label='Actual Closing Price', color='blue', linewidth=0.8)
    plt.plot(Y_pred, label='Predicted Closing Price', color='green', linestyle='--', linewidth=0.8)
    plt.plot(prev_avg, label=f'Previous {t}-Day Average', color='orange', linestyle='-.', linewidth=0.8)
    
    plt.xlabel('Time (Days)')
    plt.ylabel('Price (Original Scale)')
    plt.title(f'RBF SVR Prediction vs Actual vs Previous {t}-Day Average | γ = {gamma}')
    
    plt.legend()
    plt.grid(True)


    plt.ylim(bottom=0)

    plt.show()

gammas = [1,0.1, 0.01, 0.001]
ts = [7, 30, 90]
for t in ts:
    for gamma in gammas:
        X_train, X_test, Y_train, Y_test = prepare_dataset(dataframe, t)
        alpha_val, alpha_star, w, b = RBF_solve(X_train, Y_train, gamma)
        visualise_RBF(X_train, Y_train, X_test, Y_test, alpha_val, alpha_star, b, gamma, t, dataframe_original, data_normalizer)
