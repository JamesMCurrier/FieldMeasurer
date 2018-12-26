from random import randint
import sklearn
from sklearn.preprocessing import StandardScaler as SS
from sklearn.neural_network import MLPClassifier as NN
import pickle

def to_bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        print('Reached value that is neither True or False')
        exit(1)


def open_train(filename):
    with open(filename) as f:
        file = f.read()[:-1].split('\n')
        header = file[0].split(',')
        data = [i.split(',') for i in file[1:]]
        
    train = [[],[]]

    for line in data:
        pixel = (int(line[0]), int(line[1]), int(line[2]))
        train[0].append(pixel)
        train[1].append(to_bool(line[3]))
    return train


def make_model(data, shape =(100,)):
    global scaler
    scaler = SS()
    scaler.fit(data[0])
    data[0] = scaler.transform(data[0])

    model = NN()
    model.fit(data[0], data[1])

    M = (scaler, model)
    return M

def save_model(M, name):
    pickle.dump(M, open(name, 'wb'))

if __name__ == "__main__":
    file = input("Input the name of file containing color samples: ")
    print("\nMaking Model...\n")
    train_data = open_train(file)
    model = make_model(train_data)
    name = input("Enter name to save model as: ")
    save_model(model, name)    



