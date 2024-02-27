from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools,algorithms
from sklearn.impute import SimpleImputer
data = load_breast_cancer()
X, y = data.data, data.target
n_features = X.shape[1]
n_arrays = 100
random_arrays = [np.random.randint(2, size=n_features) for _ in range(n_arrays)]
for i, array in enumerate(random_arrays):
    print(f"Random Array {i + 1}: {array}")
def crossover(array1, array2):
    crossover_point = np.random.randint(1, len(array1))
    new_array = np.concatenate((array1[:crossover_point], array2[crossover_point:]))
    return new_array
def mutate(array, mutation_prob=0.2):
    for i in range(len(array)):
        if np.random.rand() < mutation_prob:
            array[i] = 1 - array[i]  # Flip 0 to 1 or 1 to 0
    return array
new_arrays = []
for _ in range(n_arrays):
    parent1 = random_arrays[np.random.randint(len(random_arrays))]
    parent2 = random_arrays[np.random.randint(len(random_arrays))]
    
    child = mutate(parent1.copy())
    if np.random.rand() < 0.5:  # 50% probability of crossover
        child = crossover(child, mutate(parent2.copy()))
    
    new_arrays.append(child)

combined_arrays = random_arrays + new_arrays
print(combined_arrays)
for i, array in enumerate(combined_arrays):
    print(f"Array {i + 1}: {array}")

df_original = pd.DataFrame(X, columns=data.feature_names)
def create_new_dataset(array, original_dataset):
    new_dataset = original_dataset.copy()
    zero_indices = np.where(array == 0)[0]  
    new_dataset.drop(new_dataset.columns[zero_indices], axis=1, inplace=True)
    return new_dataset
new_datasets = []
for i, array in enumerate(combined_arrays):
    new_dataset = create_new_dataset(array, df_original)
    new_datasets.append(new_dataset)
    print(f"New Dataset {i + 1}:")
    print(new_dataset.head())
def calculate_error_rate(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return 1 - accuracy_score(y_test, y_pred)

def calculate_missing_values(dataset):
    available_values = dataset.apply(lambda x: (x < 0.2).sum(), axis=0).sum()
    missing_values = dataset.shape[0] * dataset.shape[1] - available_values
    return missing_values

imputer = SimpleImputer(strategy='mean')

dataset_info = []

for i, new_dataset in enumerate(new_datasets, start=1):
    new_dataset_imputed = pd.DataFrame(imputer.fit_transform(new_dataset), columns=new_dataset.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(new_dataset_imputed, y, test_size=0.2, random_state=42)
    
    error_rate = calculate_error_rate(X_train, X_test, y_train, y_test)
    
    missing_values = calculate_missing_values(new_dataset)
    
    dataset_info.append([error_rate, new_dataset.shape[1], missing_values])
    
    print(f"Dataset {i}: Error Rate: {error_rate}, Number of Features: {new_dataset.shape[1]}, Missing Values: {missing_values}")

dataset_info_array = np.array(dataset_info)

vectors = [tuple(info) for info in dataset_info_array]

for i, vector in enumerate(vectors, start=1):
    print(f"Vector {i}: {vector}")



creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))  # Minimize error rate and number of features, maximize missing values
creator.create("Individual", list, fitness=creator.FitnessMulti)

def evaluate(individual):
    # Extract the feature array from the individual
    feature_array = np.array(individual)
    
    # Create a new dataset based on the feature array
    new_dataset = create_new_dataset(feature_array, df_original)
    
    # Impute missing values in the new dataset
    new_dataset_imputed = pd.DataFrame(imputer.fit_transform(new_dataset), columns=new_dataset.columns)
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(new_dataset_imputed, y, test_size=0.2, random_state=42)
    
    # Calculate the error rate
    error_rate = calculate_error_rate(X_train, X_test, y_train, y_test)
    
    # Calculate the number of features
    num_features = len(new_dataset.columns)
    
    # Calculate the number of missing values
    missing_values = calculate_missing_values(new_dataset)
    
    # Return a tuple containing the calculated values
    return error_rate, num_features, missing_values,


toolbox = base.Toolbox()
toolbox.register("attr_bool", np.random.choice, [0, 1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=3)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
ref_points = tools.uniform_reference_points(nobj=3, p=12)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
pop = [creator.Individual(array) for array in combined_arrays]
algorithms.eaMuPlusLambda(pop, toolbox, mu=5, lambda_=400, cxpb=0.7, mutpb=0.2, ngen=200, stats=None, halloffame=None)
for idx, ind in enumerate(pop):
    print("Individual:", idx+1, "Genotype:", ind, "Fitness:", ind.fitness.values)
    print("Accuracy:", ind.fitness.values[0])
