# datasets
from datasets import *

# Gaussian Naive Bayes
from GaussianNB import *

# utils
from utils import *

# set numpy random seed
np.random.seed(42)

def main():

    """
    Description:
        Trains and tests our Random Forest
    
    Parameters:
        None
    
    Returns:
        None
    """

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    test_size = 0.2
    random_state = 42
    dataset_name = 'Breast Cancer'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the breast cancer dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_breast_cancer()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'\nThe Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nGaussian Naive Bayes\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = gnb.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    print('Plotting...')

    # scatter plot of original data
    title_scatter = f'{dataset_name} - {feature_names[0]} vs. {feature_names[1]}'
    save_path_scatter = 'plots/bc/bc_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = feature_names[0], y_label = feature_names[1], 
                                class_names = class_names, savepath = save_path_scatter)

    # decision boundary
    resolution = 0.001
    title_boundary = f'{dataset_name} Decision Boundary - {feature_names[0]} vs. {feature_names[1]}'  
    save_path_boundary = 'plots/bc/bc_decision_boundary.png'
    visualize_decision_boundary(X = X[:, [0, 1]], y = y, title = title_boundary, x_label = feature_names[0],
                                y_label = feature_names[1], class_names = class_names ,resolution = resolution,
                                                          savepath = save_path_boundary)

    print('Please refer to plots/bc directory to view decision boundaries.')
    print('--------------------------------------------------------------------------------------------------------------\n')

    ######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Iris'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the iris dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_iris()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nGaussian Naive Bayes\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = gnb.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')

    resolution = 0.001

    iris_visualize(X, y, feature_names, class_names)


    print('Plotting Iris Sepals...')
    X_sepals = X[:, [0, 1]]
    visualize_decision_boundaries_iris(X = X_sepals, y = y, resolution = resolution, sepal_or_petal = 'Sepal')
    # Iris Petals
    print('Plotting Iris Petals...')
    X_petals = X[:, [2, 3]]
    visualize_decision_boundaries_iris(X = X_petals, y = y, resolution = resolution, sepal_or_petal = 'Petal')


    
    print('Please refer to plots/iris directory to view decision boundaries.')
    print('--------------------------------------------------------------------------------------------------------------\n')
    #######################################################################################################################################

    print('---------------------------------------------------Dataset----------------------------------------------------')
    # dataset hyperparameters
    dataset_name = 'Diabetes'
    
    # create an instance of Datasets class
    datasets = Datasets(test_size = test_size, random_state = random_state)

    # load the diabetes dataset
    X, y, feature_names, class_names, X_train, X_test, y_train, y_test = datasets.load_diabetes()

    print(f'Loading {dataset_name} Dataset...')
    print(f'\nThe Features of {dataset_name} Dataset are:', ', '.join(feature_names))
    print(f'The Labels of the {dataset_name} Dataset are:', ', '.join(class_names))
    print(f'\n{dataset_name} contains {len(X_train)} train samples and {len(X_test)} test samples.')
    print('---------------------------------------------------Model------------------------------------------------------')
    print('\nGaussian Naive Bayes\n')
    print('---------------------------------------------------Training---------------------------------------------------')
    print('Training...\n')

    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    print('Done Training!') 
    print('---------------------------------------------------Testing----------------------------------------------------')
    print('Testing...\n')
    predictions = gnb.predict(X_test)

    acc = accuracy_fn(y_true = y_test, y_pred = predictions)

    print('{0} Test Accuracy = {1}%'.format(dataset_name, acc))
    print('\nDone Testing!')
    print('---------------------------------------------------Plotting---------------------------------------------------')
    
    print('Plotting...')

    # scatter plot of original data
    feature_1, feature_2 = 'ldl', 'hdl'
    title_scatter = f'{dataset_name} - {feature_1} vs. {feature_2}'
    save_path_scatter = 'plots/db/db_scatter.png'
    scatter_plot(X = X, y = y, title = title_scatter, x_label = feature_1, y_label = feature_2, 
                                class_names = class_names, savepath = save_path_scatter)


    # decision boundary
    resolution = 0.001
    title_boundary = f'{dataset_name} Decision Boundary - {feature_1} vs. {feature_2}'  
    save_path_boundary = 'plots/db/db_decision_boundary.png'
    visualize_decision_boundary(X = X[:, [5, 6]], y = y, title = title_boundary, x_label = feature_1,
                                y_label = feature_2, class_names = class_names ,resolution = resolution,
                                                          savepath = save_path_boundary)

    
    print('Please refer to plots/db directory to view decision boundaries.')
    print('--------------------------------------------------------------------------------------------------------------')


    return None

if __name__ == '__main__':

    # run everything
    main()