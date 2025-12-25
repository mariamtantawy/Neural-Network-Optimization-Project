from random_search import random
import numpy as np
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad

def load_and_preprocess_cifar10():

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

def extract_features(images):
    images = preprocess_input(images * 255.0)
    features = vgg16.predict(images, verbose=0)
    features = features.reshape(features.shape[0], -1)

    return features


class GeneticOptimizer:
    def __init__(self, train_features, train_labels, val_features, val_labels,
                 population_size=4, num_generations=3, mutation_probability=0.2):

        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels

        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_probability = mutation_probability

        self.hyperparam_space = {
            "hidden_layers": [1, 2, 3],
            "neurons": [64, 128, 256, 512],
            "activation": ["relu", "tanh", "sigmoid"],
            "learning_rate": [1e-2, 1e-3, 1e-4],
            "batch_size": [32, 64, 128],
            "optimizer": ["sgd", "adam", "rmsprop", "adagrad"],
            "epochs": [3, 5]
        }

        self.best_hyperparameters = None
        self.best_validation_accuracy = -1

    def CreateRandomSolution(self):
        return {key: random.choice(self.hyperparam_space[key]) for key in self.hyperparam_space}

    def CreateChild(self, parent1, parent2):
        return {key: random.choice([parent1[key], parent2[key]]) for key in parent1}

    def MutateSolution(self, solution):
        for key in solution:
            if random.random() < self.mutation_probability:
                solution[key] = random.choice(self.hyperparam_space[key])
        return solution

    def BuildNeuralNetwork(self, inputSize, hp):
        model = Sequential()

        model.add(Dense(hp["neurons"], activation=hp["activation"], input_shape=(inputSize,)))

        for _ in range(hp["hidden_layers"] - 1):
            model.add(Dense(hp["neurons"], activation=hp["activation"]))

        model.add(Dense(10, activation="softmax"))

        optimizer_map = {
            "sgd": SGD(learning_rate=hp["learning_rate"]),
            "adam": Adam(learning_rate=hp["learning_rate"]),
            "rmsprop": RMSprop(learning_rate=hp["learning_rate"]),
            "adagrad": Adagrad(learning_rate=hp["learning_rate"])
        }

        model.compile(
            optimizer=optimizer_map[hp["optimizer"]],
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def EvaluateSolution(self, hp):
        input_dim = self.train_features.shape[1]
        model = self.BuildNeuralNetwork(input_dim, hp)

        history = model.fit(
            self.train_features, self.train_labels,
            validation_data=(self.val_features, self.val_labels),
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            verbose=0
        )

        return history.history["val_accuracy"][-1]

    def Run(self):
        population = [self.CreateRandomSolution() for _ in range(self.population_size)]

        for generation in range(self.num_generations):
            print(f"\n       Generation {generation + 1}/{self.num_generations}   ")

            scored_population = []

            for solution in population:
                score = self.EvaluateSolution(solution)
                scored_population.append((solution, score))
                print(f"Solution: {solution} --> Val Accuracy: {score:.4f}")

                if score > self.best_validation_accuracy:
                    self.best_validation_accuracy = score
                    self.best_hyperparameters = solution
                    print("New Best Found")

            scored_population.sort(key=lambda x: x[1], reverse=True)
            parents = [p[0] for p in scored_population[:self.population_size // 2]]

            new_population = parents.copy()
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(parents, 2)
                child = self.CreateChild(p1, p2)
                child = self.MutateSolution(child)
                new_population.append(child)

            population = new_population

        print("Best Hyperparameters:", self.best_hyperparameters)
        print("Best Validation Accuracy:", self.best_validation_accuracy)

        return self.best_hyperparameters, self.best_validation_accuracy
    
if __name__ == "__main__":
# Load Data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_cifar10()

    # Limit data size for faster testing (Genetic Algorithms are slow!)
    X_train_sub, y_train_sub = X_train[:2000], y_train[:2000]
    X_val_sub, y_val_sub = X_val[:500], y_val[:500]

    # Extract Features
    train_features = extract_features(X_train_sub)
    val_features = extract_features(X_val_sub)

    # Run Optimization
    optimizer = GeneticOptimizer(train_features, y_train_sub, val_features, y_val_sub)
    best_hp = optimizer.Run() 

