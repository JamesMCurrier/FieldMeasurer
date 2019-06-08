#!/usr/bin/python3
__author__ = 'James Currier'
__version__ = '2.2'



# Import required modules, ensuring sklearn is installed
from sys import exit as sys_exit

try: 
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline

except ModuleNotFoundError:
    print("sklearn is not installed; Exiting")
    sys_exit()

import pickle



def to_bool(s: str) -> bool:
    """Convert str to bool

    s: str
        string to convert
    """

    if s.lower() == 'true':
        return True

    elif s.lower() == 'false':
        return False

    else:
        print(f'Reached value that is neither True nor False: {s}')
        sys_exit()



def open_train(filename: str) -> list:
    """Open and properly transform training data

    filename: str
        name of csv file to read from
    """

    # Open csv file
    with open(filename) as f:

        # Split file by line
        file = f.read()[:-1].split('\n')
        headers = file[0].split(',')

        # Associate headers with data
        for head in ['R', 'G', 'B', 'Is Plant']:
            if head not in headers:
                print(f'Error: file must contain "{head}" column')
                sys_exit()

        body = [i.split(',') for i in file[1:]]
        data = {headers[i]: [j[i] for j in body] for i in range(len(headers))}

    # Make the training set
    train = [[],[]]

    for i in range(len(body)):
        pixel = (int(data["R"][i]), int(data["G"][i]), int(data["B"][i]))
        train[0].append(pixel)
        train[1].append(to_bool(data["Is Plant"][i]))

    return train



def make_model(data, shape=(100,)) -> "Model":
    """Make the predictive model
    
    data: list(list, list)
        list of training data
    shape: tuple(int)
        hidden layer sizes of the neural net
    """

    # Initialize the Pipeline
    nn = make_pipeline(StandardScaler(), MLPClassifier(shape, max_iter=1000))

    # Train the model
    nn.fit(data[0], data[1])

    return nn



def save_model(model: "Model", name: str):
    """Save the model

    model: sklearn Pipeline
        model to save
    name: str
        name to save the model to
    """
    with open(name, 'wb') as f:
        pickle.dump(model, f)



def main():
    file = input("Input the name of file containing color samples: ")
    train_data = open_train(file)
    print("\nMaking Model...\n")
    model = make_model(train_data)
    save_model(model, "NN.bin")



if __name__ == "__main__":
    main()
