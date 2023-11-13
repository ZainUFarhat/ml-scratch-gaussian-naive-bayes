# ml-scratch-gaussian-naive-bayes
Gaussian Naive Bayes Algorithm

## **Description**
The following is my from scratch implementation of the Gaussian Naive Bayes algorithm.

### **Dataset**

I tested the performance of my model on three datasets: \
\
    &emsp;1. Breast Cancer Dataset \
    &emsp;2. Iris Dataset \
    &emsp;3. Diabetes Dataset

### **Walkthrough**

**1.** Need the following packages installed: sklearn, numpy, and matplotlib.

**2.** Once you made sure all these libraries are installed, evrything is simple, just head to main.py and execute it.

**3.** Since code is modular, main.py can easily: \
\
    &emsp;**i.** Load the three datasets \
    &emsp;**ii.** Split data into train and test sets \
    &emsp;**iii.** Build a gaussian naive bayes classifier \
    &emsp;**iv.** Fit the gaussian naive bayes classifier \
    &emsp;**v.** Predict on the test set \
    &emsp;**vi.** Plot scatter plots and decision boundaries for each dataset.

### **Results**

For each dataset I will share the test accuracy and show the decision boundary predictions.

**1.** Breast Cancer Dataset:

- Numerical Result:
     - Accuracy = 96.49%

- See visualizations below:

 ![alt text](https://github.com/ZainUFarhat/ml-scratch-gaussian-naive-bayes/blob/main/plots/bc/bc_decision_boundary.png?raw=true) 

 **2.** Iris Dataset:

- Numerical Result:
     - Accuracy = 100.0%

- See visualizations below:

    I will show the decision boundaries for both sepals and petals features of the dataset.

**-** Petal Decision Boundary:

![alt text](https://github.com/ZainUFarhat/ml-scratch-gaussian-naive-bayes/blob/main/plots/iris/iris_decision_boundaries_petal.png?raw=true)

**-** Sepal Decision Boundary:

![alt text](https://github.com/ZainUFarhat/ml-scratch-gaussian-naive-bayes/blob/main/plots/iris/iris_decision_boundaries_sepal.png?raw=true)

**3.** Diabetes Dataset:

- Numerical Result:
     - Accuracy = 71.91%

- See visualizations below:

 ![alt text](https://github.com/ZainUFarhat/ml-scratch-gaussian-naive-bayes/blob/main/plots/db/db_decision_boundary.png?raw=true) 