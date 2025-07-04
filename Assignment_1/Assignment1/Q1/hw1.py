###########   Q1 -- FISHER DISCRMINANT ANALYSIS   ###########
import sys
import os
import numpy as np
# Add the path to the oracle file
sys.path.append(os.path.join(os.path.dirname(__file__), 'oracle'))
import oracle
import matplotlib.pyplot as plt
import random
# Determine the mean and covariance of the data for different classes with different values of n 
def mean_covariance_estimate(n,class_num,train_img,train_labels):
    all_vec = []
    c = 0
    for i in range(len(train_img)):
        if c < n:
            if train_labels[i] == class_num:
                c+=1
                data_vec = []
                for j in [0,1,2]:
                    for k in range(32): 
                        for l in range(32):
                            data_vec.append(train_img[i][j][k][l])
                data_vec = np.array(data_vec)
                all_vec.append(data_vec)
        else:
            break
    print(len(all_vec))
    all_vec = np.array(all_vec)
    mean = np.mean(all_vec, axis=0)
    covariance = np.cov(all_vec, rowvar=False)
    return mean,covariance
# Calculate the norm of the mean and covariance for different values of n
def class_conditional(class_num,train_img,train_labels):
    n = [50,100,500, 1000, 2000, 4000]
    mean_norm = []
    covariance_norm = []
    for i in n : 
        datas = mean_covariance_estimate(i, class_num,train_img,train_labels)
        print(datas[1].shape)
        mean_norm.append(np.linalg.norm(datas[0],ord = 2))
        covariance_norm.append(np.linalg.norm(datas[1],ord = 'fro'))
    return mean_norm,covariance_norm
# Plot the norms of the mean and covariance for different values of n for different classes
def mean_covariance_plots(class_num,train_img,train_labels):
    x = [50,100,500, 1000, 2000, 4000]
    mean,covariance = class_conditional(class_num,train_img,train_labels)
    plt.plot(x,mean , label='mean', color='b', linestyle='-', linewidth=2)
    plt.plot(x,covariance , label='covariance', color='r', linestyle='-', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.show()
# Now we load the data class-wise in a list called 'all_vec'
def all_data_classwise(train_img,train_labels):
    all_vec_0 = []
    all_vec_1 = []
    all_vec_2 = []
    all_vec_3 = []
    for i in range(len(train_img)):
            if train_labels[i] == 0:
                data_vec = []
                for j in [0,1,2]:
                    for k in range(32):
                        for l in range(32):
                            data_vec.append(train_img[i][j][k][l])
                data_vec = np.array(data_vec)
                all_vec_0.append(data_vec)
    for i in range(len(train_img)):
            if train_labels[i] == 1:
                data_vec = []
                for j in [0,1,2]:
                    for k in range(32):
                        for l in range(32):
                            data_vec.append(train_img[i][j][k][l])
                data_vec = np.array(data_vec)
                all_vec_1.append(data_vec)
    for i in range(len(train_img)):
            if train_labels[i] == 2:
                data_vec = []
                for j in [0,1,2]:
                    for k in range(32):
                        for l in range(32):
                            data_vec.append(train_img[i][j][k][l])
                data_vec = np.array(data_vec)
                all_vec_2.append(data_vec)
    for i in range(len(train_img)):
            if train_labels[i] == 3:
                data_vec = []
                for j in [0,1,2]:
                    for k in range(32):
                        for l in range(32):
                            data_vec.append(train_img[i][j][k][l])
                data_vec = np.array(data_vec)
                all_vec_3.append(data_vec)
    all_vec_0 = np.array(all_vec_0)
    all_vec_1 = np.array(all_vec_1)
    all_vec_2 = np.array(all_vec_2)
    all_vec_3 = np.array(all_vec_3)
    all_vec = [all_vec_0,all_vec_1,all_vec_2,all_vec_3]
    return all_vec
# Take a random sample of size n from the data for a given class
def random_sampling(n,class_num,train_img,train_labels):
    random_sample = []
    all_vec = all_data_classwise(train_img,train_labels)
    random_list = random.sample([i for i in range(5000)],n)
    for i in random_list:
        random_sample.append(all_vec[class_num][i])
    random_sample = np.array(random_sample)
    return random_sample
# Now, we make a funcion to calculate the mean and covariance of any given data
def mean_covariance(data):
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)
    return mean, covariance
# We make a function which takes a size n, gets 20 random samples of size n for each class, and calculates the Fisher Discriminant.
# Then it returns the objective values,i.e., the sum of 3 greatest eigenvalues for each of the 20 iterations and returns the list of objective values.
def fisher_multiclass(n,train_img,train_labels):
    objective_value_list = []
    for iter in range(20):
        random_sample_0 = random_sampling(n, 0,train_img,train_labels)
        random_sample_1 = random_sampling(n, 1,train_img,train_labels)
        random_sample_2 = random_sampling(n, 2,train_img,train_labels)
        random_sample_3 = random_sampling(n, 3,train_img,train_labels)
        mean_0, covariance_0 = mean_covariance(random_sample_0)
        mean_1, covariance_1 = mean_covariance(random_sample_1)
        mean_2, covariance_2 = mean_covariance(random_sample_2)
        mean_3, covariance_3 = mean_covariance(random_sample_3)

        means = [mean_0, mean_1, mean_2, mean_3]
        covariances = [covariance_0, covariance_1, covariance_2, covariance_3]
        num_classes = 4  
        num_features = random_sample_0.shape[1]  
        S_W = np.zeros((num_features, num_features))
        for cov in covariances:
            S_W += cov

        overall_mean = np.mean(means, axis=0)  
        S_B = np.zeros((num_features, num_features))
        
        for i in range(num_classes):
            mean_diff = (means[i] - overall_mean).reshape(num_features, 1)
            S_B += n * np.dot(mean_diff, mean_diff.T)

        eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.pinv(S_W), S_B))  # Use pinv for stability
        eigen_values = np.real(eigen_values)
        eigen_vectors = np.real(eigen_vectors)
        idx = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        eigen_values = eigen_values[:3]
        eigen_vectors = eigen_vectors[:, :3]
        objective_value_list.append(sum(list(eigen_values)))

    return objective_value_list
# Now we plot the boxplot for the objective values for different values of n
def box_plot(train_img,train_labels):
    n_values = [2500, 3500, 4000, 4500, 5000]
    objective_values = {}
    for n in n_values:
        objective_values[n] = fisher_multiclass(n,train_img,train_labels)
    plt.figure(figsize=(10, 6))
    plt.boxplot([objective_values[n] for n in n_values], labels=n_values)
    plt.xlabel("Number of Samples (n)")
    plt.ylabel("Objective Function Value")
    plt.title("Box Plot of Objective Values for Different n in Fisher’s Discriminant")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
# Now we determine the FLD for the whole dataset, i.e., n = 5000 for each class
def FLD_whole(train_img,train_labels):
    all_vec = all_data_classwise(train_img,train_labels)
    mean_0, covariance_0 = mean_covariance_estimate(5000, 0)
    mean_1, covariance_1 = mean_covariance_estimate(5000, 1)
    mean_2, covariance_2 = mean_covariance_estimate(5000, 2)
    mean_3, covariance_3 = mean_covariance_estimate(5000, 3)

    means = [mean_0, mean_1, mean_2, mean_3]
    covariances = [covariance_0, covariance_1, covariance_2, covariance_3]

    num_classes = 4  
    num_features = 3072 
    S_W = np.zeros((num_features, num_features))
    for cov in covariances:
        S_W += cov
    overall_mean = np.mean(means, axis=0)  
    S_B = np.zeros((num_features, num_features))

    for i in range(num_classes):
        mean_diff = (means[i] - overall_mean).reshape(num_features, 1)
        S_B += 5000 * np.dot(mean_diff, mean_diff.T)

    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.pinv(S_W), S_B))  

    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)

    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    eigen_values = eigen_values[:3]
    eigen_vectors = eigen_vectors[:, :3]

    # Now we project the whole training data on the 3-dimensional space spanned by the 3 eigenvectors
    projections = []
    projections_0 = []
    projections_1 = []
    projections_2 = []
    projections_3 = []
    for i in range(len(all_vec[0])):
        projection = np.dot(all_vec[0][i],eigen_vectors)
        projections_0.append(projection)
    for i in range(len(all_vec[1])):
        projection = np.dot(all_vec[1][i],eigen_vectors)
        projections_1.append(projection)
    for i in range(len(all_vec[2])):
        projection = np.dot(all_vec[2][i],eigen_vectors)
        projections_2.append(projection)
    for i in range(len(all_vec[3])):
        projection = np.dot(all_vec[3][i],eigen_vectors)
        projections_3.append(projection)
    projections_0 = np.array(projections_0)
    projections_1 = np.array(projections_1)
    projections_2 = np.array(projections_2)
    projections_3 = np.array(projections_3)

    projections = [projections_0, projections_1, projections_2, projections_3]
    return projections

# Now we plot the 3D scatter plot for the projected data
def plot_3d(train_img,train_labels):
    colors = ['r', 'g', 'b', 'y']
    labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    projections = FLD_whole(train_img,train_labels)
    for i, proj in enumerate(projections):
        ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c=colors[i], label=labels[i], alpha=0.7)
    ax.set_xlabel("Fisher Component 1")
    ax.set_ylabel("Fisher Component 2")
    ax.set_zlabel("Fisher Component 3")
    ax.set_title("3D Projection of Data using Fisher’s Linear Discriminant")
    ax.legend()
    plt.show()
# Now we make a function to classify the test data using the Fisher Discriminant for various values of n, given test data.
# We assume multivariate normal distribution for the data and calculate the probability of the data belonging to each class.
# We first project the test data on the 3-dimensional space spanned by the 3 eigenvectors and then calculate the probability of the data belonging to each class.
def classifier_var(n,train_img,train_labels,test_img,test_labels):
    random_sample_0 = random_sampling(n, 0,train_img,train_labels)
    random_sample_1 = random_sampling(n, 1,train_img,train_labels)
    random_sample_2 = random_sampling(n, 2,train_img,train_labels)
    random_sample_3 = random_sampling(n, 3,train_img,train_labels)
    mean_0, covariance_0 = mean_covariance(random_sample_0)
    mean_1, covariance_1 = mean_covariance(random_sample_1)
    mean_2, covariance_2 = mean_covariance(random_sample_2)
    mean_3, covariance_3 = mean_covariance(random_sample_3)

    means = [mean_0, mean_1, mean_2, mean_3]
    covariances = [covariance_0, covariance_1, covariance_2, covariance_3]

    num_classes = 4
    num_features = random_sample_0.shape[1]  
    S_W = np.zeros((num_features, num_features))
    for cov in covariances:
        S_W += cov

    overall_mean = np.mean(means, axis=0)  
    S_B = np.zeros((num_features, num_features))
    
    for i in range(num_classes):
        mean_diff = (means[i] - overall_mean).reshape(num_features, 1)
        S_B += n * np.dot(mean_diff, mean_diff.T)


    eigen_values, eigen_vectors = np.linalg.eig(np.dot(np.linalg.pinv(S_W), S_B))  
    
   
    eigen_values = np.real(eigen_values)
    eigen_vectors = np.real(eigen_vectors)

    idx = np.argsort(eigen_values)[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]
    eigen_values = eigen_values[:3]
    eigen_vectors = eigen_vectors[:, :3]

    projected_mean = []
    projected_covariance = []
    for i in range(4):
        projected_mean.append(np.dot(means[i], eigen_vectors))
        projected_covariance.append(np.dot(np.dot(eigen_vectors.T, covariances[i]), eigen_vectors))
    accuracy = 0
    test_data =[]
    for i in range(len(test_img)):
        data_vec = []
        for j in [0,1,2]:
            for k in range(32):
                for l in range(32):
                    data_vec.append(test_img[i][j][k][l])
        data_vec = np.array(data_vec)
        test_data.append(data_vec)
    for i in range(len(test_data)):
        projected_data = np.dot(test_data[i], eigen_vectors)
        p_1 = np.exp(-0.5 * np.dot(np.dot((projected_data - projected_mean[0]), np.linalg.inv(projected_covariance[0])), (projected_data - projected_mean[0]).T))
        p_2 = np.exp(-0.5 * np.dot(np.dot((projected_data - projected_mean[1]), np.linalg.inv(projected_covariance[1])), (projected_data - projected_mean[1]).T))
        p_3 = np.exp(-0.5 * np.dot(np.dot((projected_data - projected_mean[2]), np.linalg.inv(projected_covariance[2])), (projected_data - projected_mean[2]).T))
        p_4 = np.exp(-0.5 * np.dot(np.dot((projected_data - projected_mean[3]), np.linalg.inv(projected_covariance[3])), (projected_data - projected_mean[3]).T))
        p = [p_1, p_2, p_3, p_4]
        if np.argmax(p) == test_labels[i]:
            accuracy += 1
    return accuracy/len(test_data)
# Now we calculate the accuracy for different values of n
def accuracy_test(train_img,train_labels,test_img,test_labels):
    n = [2500,3500,4500,5000]
    accuracy = []
    for i in n:
        accuracy.append(classifier_var(i,train_img,train_labels,test_img,test_labels))
    print(accuracy)
    return accuracy
# Now we plot the accuracy for different values of n
def accuracy_plot(train_img,train_labels,test_img,test_labels):
    accuracy = accuracy_test(train_img,train_labels,test_img,test_labels)
    x = [2500,3500,4500,5000]
    plt.plot(x,accuracy , label='accuracy', color='b', linestyle='-', linewidth=2)
    plt.legend()
    plt.grid(True)
    plt.show()

def Q1():
 
   


    # Load the data from oracle
    res = oracle.q1_fish_train_test_data(23746)
    attributes = res[0]
    train_img = res[1]
    train_labels = res[2]
    test_img = res[3]
    test_labels = res[4]
    train_img = np.array(train_img)
    train_labels = np.array(train_labels)
    test_img = np.array(test_img)
    test_labels = np.array(test_labels)
    print(attributes)
    print(train_img.shape)
    print(train_labels.shape)
    print(test_img.shape)
    print(test_labels.shape)
    
    # Uncomment whatever needed

    mean_covariance_plots(0,train_img,train_labels)
    mean_covariance_plots(1,train_img,train_labels)
    mean_covariance_plots(2,train_img,train_labels)
    mean_covariance_plots(3,train_img,train_labels)
    box_plot(train_img,train_labels)
    plot_3d(train_img,train_labels)
    accuracy_plot(train_img,train_labels,test_img,test_labels)


###########   Q2 -- BAYES CLASSIFICATION   ###########
import sys
import sklearn
import os
import numpy as np
import random
from sklearn.model_selection import KFold
import oracle

# Determine means and covariances for each feature,i.e., each component in the 784 sized vector, in both the classes
def mean_covariances(data):
    data_17 =[]
    data_30 = []
    for i in range(len(data[0])):
        #print(data[0][i])
        if data[0][i][0] == 17:
            data_17.append([np.longdouble(data[0][i][j]) for j in range(len(data[0][i]))])
        if data[0][i][0] == 30:
            data_30.append([np.longdouble(data[0][i][j]) for j in range(len(data[0][i]))])
    data_17 = np.array(data_17)
    data_30 = np.array(data_30)

    mean_list_17 = []
    std_list_17 = []
    mean_list_30 = []
    std_list_30 = []
    for i in range(1, 785):
        mean_list_17.append(np.longdouble(np.mean(data_17[:, i])))
        std_list_17.append(np.longdouble(np.std(data_17[:, i])))
        mean_list_30.append(np.longdouble(np.mean(data_30[:, i])))
        std_list_30.append(np.longdouble(np.std(data_30[:, i])))

    return mean_list_17,std_list_17,mean_list_30,std_list_30
# Now we make a Bayes classifier which takes the test data and classifies it as 17 or 30 based on the probability of the data belonging to each class
def Bayes(epsilon, x,data,mean_list_17,std_list_17,mean_list_30,std_list_30,p_1 = 0.5):
    values = mean_covariance(data)
    p_1 = np.longdouble(p_1)
    # log_p_17 = 0
    # log_p_30 = 0
    mean_list_17 = values[0]
    std_list_17 = values[1]
    mean_list_30 = values[2]
    std_list_30 = values[3]
    p_17 = 1
    p_30 = 1

    for i in range(784):
        if std_list_17[i] == 0 or std_list_30[i] == 0:
            continue
        p_17 *= np.longdouble(np.exp(-0.5 * (x[i] - mean_list_17[i]) ** 2 / std_list_17[i] ** 2))/ (std_list_17[i] * np.sqrt(2 * np.pi))
        p_30 *= np.longdouble(np.exp(-0.5 * (x[i] - mean_list_30[i]) ** 2 / std_list_30[i] ** 2))/ (std_list_30[i] * np.sqrt(2 * np.pi))

    if p_17*p_1/(p_17*p_1+p_30*(1-p_1)) >=  0.5 + epsilon:
        return 17
    elif p_17*p_1/(p_17*p_1+p_30*(1-p_1)) <= 0.5 - epsilon:
        return 30
    return 0
# Now we calculate the accuracy of the classifier for different values of epsilon
def classification(e,test,data,p_1 = 0.5):
    misclassified = 0
    rejection = 0
    non_rejected = 0
    mean_list_17,std_list_17,mean_list_30,std_list_30 = mean_covariances(data)
    for i in range(len(test)):
        val = Bayes(e,test[i][1:],mean_list_17,std_list_17,mean_list_30,std_list_30,p_1)
        if val == 0:
            rejection += 1
        else :
            non_rejected += 1
            if val != test[i][0]:
                misclassified += 1
       
    print(misclassified)
    print(rejection)
    return misclassified/non_rejected, rejection/len(test)
# We now plot the misclassification rate with changing epsilon for a 50-50 split of the data in between the two classes
def plot_50_50(test,data):
    x = [0.01,0.1,0.25,0.4]
    data_new = [classification(i,test,data) for i in x]
    y = [data_new[i][0] for i in range(len(data_new))]
    z = [data_new[i][1] for i in range(len(data_new))]

    import matplotlib.pyplot as plt
    plt.plot(x, y)
    #plt.plot(x, z)
    plt.grid(True)
    plt.legend(['Misclassification Rate',])
    plt.xlabel('Epsilon')
    plt.ylabel('Rate')
    plt.title('Misclassification Rate and Rejection Rate vs Epsilon')
    plt.show()
    print(y)
    print(z)
# Now we accumulate datas with the same label in two lists
def accumulate(data):
    test_17 = []
    for i in range(len(data[1])):
        if data[1][i][0] == 17:
            test_17.append(data[1][i])
    test_17 = np.array(test_17)
    test_30 = []
    for i in range(len(data[1])):
        if data[1][i][0] == 30:
            test_30.append(data[1][i])
    test_30 = np.array(test_30)
    return test_17,test_30
# Now we make randomised splits of the data in ratios 60-40,80-20,90-10 and 99-1
def split_60_40(data):
    test_17,test_30 = accumulate(data)
    random.shuffle(test_17)
    random.shuffle(test_30)

    test_17_60 = test_17[:int(0.6*len(test_17))]
    test_30_40 = test_30[:int(0.4*len(test_30))]
    return test_17_60,test_30_40

def split_80_20(data):
    test_17,test_30 = accumulate(data)
    random.shuffle(test_17)
    random.shuffle(test_30)

    test_17_80 = test_17[:int(0.8*len(test_17))]
    test_30_20 = test_30[:int(0.2*len(test_30))]
    return test_17_80,test_30_20

def split_90_10(data):
    test_17,test_30 = accumulate(data)
    random.shuffle(test_17)
    random.shuffle(test_30)

    test_17_90 = test_17[:int(0.9*len(test_17))]
    test_30_10 = test_30[:int(0.1*len(test_30))]
    return test_17_90,test_30_10

def split_99_1(data):
    test_17,test_30 = accumulate(data)
    random.shuffle(test_17)
    random.shuffle(test_30)

    test_17_99 = test_17[:int(0.99*len(test_17))]
    test_30_1 = test_30[:int(0.01*len(test_30))]
    return test_17_99,test_30_1
# Now we calculate the misclassification rate for different splits of the data
def plot_60_40(data):
    epsilons = [0.1,0.25,0.4]
    test_17_60,test_30_40 = split_60_40(data)
    data_60_40 = [classification(i,np.concatenate((test_17_60,test_30_40)),data,0.6) for i in epsilons]
    misclass_60_40 = [data_60_40[i][0] for i in range(len(data_60_40))]
    rej_60_40 = [data_60_40[i][1] for i in range(len(data_60_40))]
    print(misclass_60_40)
    print(rej_60_40)
    plt.plot(epsilons, misclass_60_40)
def plot_80_20(data):
    epsilons = [0.1,0.25,0.4]
    test_17_80,test_30_20 = split_80_20(data)
    data_80_20 = [classification(i,np.concatenate((test_17_80,test_30_20)),data,0.8) for i in epsilons]
    misclass_80_20 = [data_80_20[i][0] for i in range(len(data_80_20))]
    rej_80_20 = [data_80_20[i][1] for i in range(len(data_80_20))]
    print(misclass_80_20)
    print(rej_80_20)
    plt.plot(epsilons, misclass_80_20)
def plot_90_10(data):
    epsilons = [0.1,0.25,0.4]
    test_17_90,test_30_10 = split_90_10(data)
    data_90_10 = [classification(i,np.concatenate((test_17_90,test_30_10)),data,0.9) for i in epsilons]
    misclass_90_10 = [data_90_10[i][0] for i in range(len(data_90_10))]
    rej_90_10 = [data_90_10[i][1] for i in range(len(data_90_10))]
    print(misclass_90_10)
    print(rej_90_10)
    plt.plot(epsilons, misclass_90_10)
def plot_99_1(data):
    epsilons = [0.1,0.25,0.4]
    test_17_99,test_30_1 = split_99_1(data)
    data_99_1 = [classification(i,np.concatenate((test_17_99,test_30_1)),data,0.99) for i in epsilons]
    misclass_99_1 = [data_99_1[i][0] for i in range(len(data_99_1))]
    rej_99_1 = [data_99_1[i][1] for i in range(len(data_99_1))]
    print(misclass_99_1)
    print(rej_99_1)
    plt.plot(epsilons, misclass_99_1)
# The following function gives mean and covariances for the data at specified index
def mean_variance(index,data):
    data_17 = []
    data_30 = []
    for i in index :
        if data[0][i][0] == 17:
            data_17.append([np.longdouble(data[0][i][j]) for j in range(len(data[0][i]))])
        if data[0][i][0] == 30:
            data_30.append([np.longdouble(data[0][i][j]) for j in range(len(data[0][i]))])
    data_17 = np.array(data_17)
    data_30 = np.array(data_30)
    mean_list_17 = []
    std_list_17 = []
    mean_list_30 = []
    std_list_30 = []
    for i in range(1, 785):
        mean_list_17.append(np.mean(data_17[:, i]))
        std_list_17.append(np.std(data_17[:, i]))
        mean_list_30.append(np.mean(data_30[:, i]))
        std_list_30.append(np.std(data_30[:, i]))
    return mean_list_17,std_list_17,mean_list_30,std_list_30
# We now return various metrics by training Bayes classifier
def testing(index,values,data):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    rejection = 0
    for i in index:
        val = Bayes(0.25,data[0][i][1:],values[0],values[1],values[2],values[3])
        if val == 0:
            rejection += 1
            continue
        elif val == 17 and data[0][i][0] == 17:
            TP += 1
        elif val == 17 and data[0][i][0] == 30:
            FP += 1
        elif val == 30 and data[0][i][0] == 30:
            TN += 1
        elif val == 30 and data[0][i][0] == 17:
            FN += 1
    return TP,FP,TN,FN,rejection
# Now, we use K Fold Cross Validation to get the metrics for the classifier
def k_fold(data):
    average_miss = 0
    kf = KFold(n_splits=5, shuffle=False)
    miss = 0
    best_values = []
    for i, (train_index, test_index) in enumerate(kf.split(data[0])):
        print("Run", i)
        confusion_matrix = np.zeros((2,2))
        values = mean_variance(train_index)
        if i == 1 :
            best_values = values
        TP,FP,TN,FN,rejection = testing(test_index,values)
        confusion_matrix[0][0] = TP
        confusion_matrix[0][1] = FP
        confusion_matrix[1][0] = FN
        confusion_matrix[1][1] = TN
        print("Confusion:",confusion_matrix)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f1 = 2*recall*precision/(recall+precision)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        print("Recall:", recall)
        print("Precision:", precision)
        print("F1 Score:", f1)
        print("Accuracy:", accuracy)
        print("Rejection Rate:", rejection)
    return best_values
# We now classify the test data with the best values of mean and variance from the KFold Cross Validation
def test_on_test_data(data,values):
    rejection = 0
    non_rejection = 0
    misclassified = 0
    for i in range(len(data[1])):
        val = Bayes(0.25,data[1][i][1:],values[0],values[1],values[2],values[3])
        if val == 0:
            rejection += 1
            continue
        else :
            non_rejection += 1
            if val != data[1][i][0]:
                misclassified += 1
    return misclassified, rejection, non_rejection
# We retrieve the misclassification rate, rejection rate and non-rejection for the test data
def print_values(data):
    print("Testing on test data")
    best_values = k_fold(data)
    misclassified, rejection, non_rejection = test_on_test_data(data,best_values)
    print("Misclassified:",misclassified)
    print("Rejection:",rejection)
    print("Non-Rejection:",non_rejection)
    print("Misclassification Rate:",misclassified/non_rejection)

# We accumulate everything in the main function
def Q2():

   
    data = oracle.q2_train_test_emnist(23746,"EMNIST/emnist-balanced-train.csv","EMNIST/emnist-balanced-test.csv")
    print(data[0].shape)
    print(data[1].shape)
    print(data[0])

    # Uncomment whatever needed
    # plot_50_50(data)
    # plot_60_40(data)
    # plot_80_20(data)
    # plot_90_10(data)
    # plot_99_1(data)

    # print_values(data)


###########   Q3 -- DECISION TREE CLASSIFIER   ###########
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import csv
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from dtreeviz import model as dtreeviz_model
import sys
import os


# We add column names in he first row of the csv file "processed.cleveland.data" and store in a new file
def add_column_names(file_path="processed.cleveland.data", output_file="processed_cleveland_with_headers.csv"):
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "goal"
    ]
    with open(file_path, "r") as infile:
        lines = infile.readlines()

    with open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(column_names) 
        for line in lines:
            writer.writerow(line.strip().split(","))  

    print(f"File saved as {output_file} with column names added.")
# We write function for inserting missing values in the data, with mean for numerical columns and mode for categorical columns
def data_cleanup():
    add_column_names()
    df = pd.read_csv("processed_cleveland_with_headers.csv", na_values=["?"])  
    df = df.apply(pd.to_numeric, errors='coerce')
    print("Missing values before imputation:\n", df.isna().sum())
    num_imputer = SimpleImputer(strategy="mean")
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'exang', 'fbs', 'restecg', 'cp']
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_cols = ['slope', 'ca', 'thal']
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    print("\nMissing values after imputation:\n", df.isna().sum())
    print("\nCleaned Data:\n", df.head())
    df.to_csv("cleaned_data.csv", index=False)

# We then make a function to train a decision tree classifier on the cleaned data
def make_decision_tree(data):
    features =[]
    goals = []
    f = open('cleaned_data.csv', 'r')
    reader = csv.reader(f)
    for row in reader:
        features.append(list(row[:-1]))
        goals.append(row[-1])
    f.close()

    features = features[1:]
    goals = goals[1:]
    for i in range(len(goals)) :
        if goals[i] == '0':
            goals[i] = 0
        else:
            goals[i] = 1

    features = np.array(features)
    goals = np.array(goals)
    features = pd.DataFrame(features)
    goals = pd.DataFrame(goals)
    

    X_train, X_test, y_train, y_test = train_test_split(features, goals, test_size=0.2, random_state=98)
    print(type(X_train))

    splitter = data[1]
    max_depth = data[2]
    criteria = data[0]
    model = DecisionTreeClassifier(criterion=criteria, splitter=splitter, max_depth=max_depth, random_state=81)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix: \n", conf_matrix)

    class_report = classification_report(y_test, y_pred)
    print("Classification Report: \n", class_report)

    return model, X_train, y_train

# We visualise the decision tree using dtreeviz
def visualise(data):
    datas = make_decision_tree(data)
    model = datas[0]
    X_train = datas[1]
    y_train = datas[2]

    X_train_np = np.array(X_train)
    y_train_np = np.array(y_train)
    for i in range(len(X_train_np)):
        for j in range(len(X_train_np[i])):
            print(X_train_np[i][j])
            X_train_np[i][j] = float(X_train_np[i][j])
    new_y_train = []
    for i in range(len(y_train_np)):
        new_y_train.append(int(y_train_np[i]))
    y_train_np = np.array(new_y_train)
    df = pd.read_csv("cleaned_data.csv")
    viz = dtreeviz_model(
        model, 
        X_train_np, 
        y_train_np, 
        target_name="goal", 
        feature_names=list(df.columns[:-1]), 
        class_names=["No Disease", "Disease"]
    )

    
    viz.view().save("decision_tree.svg")
    print("Decision Tree visualization saved to 'decision_tree.svg'")

# Collect everything in the main function
def Q3():
    
    #Add the 'oracle' directory to the Python path
    data = oracle.q3_hyper(23746)
    print(data[0])
    print(data[1])
    print(data[2])


    # Uncomment whatever needed
    # data_cleanup()
    # visualise(data)   

Q1()