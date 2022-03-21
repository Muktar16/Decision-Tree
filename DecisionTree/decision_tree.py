from csv import reader
from random import randint
import math


class Leaf: 
    def __init__(self, dataset):
        self.predictions = class_counts(dataset)


class Decision_Node:
    def __init__(self,condition,right_node,left_node):
        self.condition = condition
        self.right_node = right_node
        self.left_node = left_node


def read_data_from_file(filename):
	lines = reader(open(filename, "rt"))
	dataset = list(lines)
	return dataset

def convertToFloat(row):
	for i in range (0, len(row)):
            row[i] = float(row[i])
		    

def class_counts(dataset):
    counts = {}  
    for row in dataset:
        class_id = row[-1]
        if class_id not in counts:
            counts[class_id] = 0
        counts[class_id] += 1
    return counts

class Condition:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, row):
        val = row[self.column]
        return val >= self.value

def split(dataset, condition): 
    right_dataset, left_dataset = [], []
    for row in dataset:
        if condition.match(row):
            right_dataset.append(row)
        else:
            left_dataset.append(row)
    return right_dataset, left_dataset


def Entropy(dataset):
    counts = class_counts(dataset)
    entropy = 0.0
    for class_id in counts:
        prob_of_class = counts[class_id] / float(len(dataset))
        entropy = entropy - prob_of_class*math.log(prob_of_class,2)
    return entropy


def information_gain(left_dataset, right_dataset, current_entropy):
    p = float(len(left_dataset)) / (len(left_dataset) + len(right_dataset))
    return current_entropy - p * Entropy(left_dataset) - (1 - p) * Entropy(right_dataset)


def find_best_split(dataset):
    best_gain = 0 
    best_condition = None 
    current_entropy = Entropy(dataset)
    n_item_col = len(dataset[0]) - 1  

    for col in range(n_item_col): 

        values = set([row[col] for row in dataset]) 

        for val in values:  

            condition = Condition(col, val)
            right_dataset, left_dataset = split(dataset, condition)
            if len(right_dataset) == 0 or len(left_dataset) == 0:
                continue
            gain = information_gain(right_dataset, left_dataset, current_entropy)

            if gain >= best_gain:
                best_gain, best_condition = gain, condition

    return best_gain, best_condition



def create_decision_tree(dataset):
    
    gain, condition = find_best_split(dataset)
    if gain == 0:
        return Leaf(dataset)

    right_dataset, left_dataset = split(dataset, condition)
    
    right_node = create_decision_tree(right_dataset)
    left_node = create_decision_tree(left_dataset)
    
    return Decision_Node(condition, right_node, left_node)


def print_tree(node, prefix=""):
    if isinstance(node, Leaf):
        print ("Leaf_Node: ", node.predictions)
        return
    print ("Decision_Node " + str(node.condition.column) + "   " + str(node.condition.value))

    print (prefix + ' L--> ', end = "")
    print_tree(node.right_node, prefix + "   ")

    print (prefix + ' R--> ',
           end = "")
    print_tree(node.left_node, prefix + "   ")


def classify(row, node):  
    if isinstance(node, Leaf):
        return node.predictions

    if node.condition.match(row):
        return classify(row, node.right_node)
    else:
        return classify(row, node.left_node)


def create_folds(dataset,n_folds):
    folds = list()
    fold_size = int(len(dataset)/n_folds)
    for i in range (n_folds):
        fold = list()
        for j in range (fold_size):
            idx = randint(0, len(dataset)-1)
            fold.append(dataset.pop(idx))
        folds.append(fold)
    return folds   

    
def calculate_accuracy(folds):
    total_accuracy = 0.0
    for fold in folds:
        trainingData = list(folds) 
        trainingData.remove(fold)
        trainingData = sum(trainingData, [])
        testingData = fold
        my_tree = create_decision_tree(trainingData)
        print_tree(my_tree)
        
        total_row = 0
        matche_count = 0
        for row in testingData:
            total_row += 1
            classified = classify(row,my_tree)
            for class_id in classified.keys():
                if class_id == row[-1]:
                    matche_count += 1
        accuracy = (matche_count/total_row)*100
        total_accuracy += accuracy
        print()
        print('Accuracy: ', accuracy, '%')
        print()
    return total_accuracy


if __name__ == '__main__':
    dataset = read_data_from_file('wine.csv')
    for row in dataset:
        row = convertToFloat(row)
    
    n_folds = 10   
    folds = create_folds(dataset,n_folds)
    
    total_accuracy = calculate_accuracy(folds)
    
    average_accuracy = total_accuracy/ n_folds
    print('Mean Accuracy: ', average_accuracy,"%")
    print()