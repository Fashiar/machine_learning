import numpy as np
import pandas as pd
import time

class DecisionTreeNode(object):
    # Constructor
    def __init__(self, att, thr, left, right):  
        self.attribute = att
        self.threshold = thr
        # left and right are either binary classifications or references to 
        # decision tree nodes
        self.left = left     
        self.right = right   

    def print_tree(self,indent=''):
        if self.left  in [0,1]:
            print(indent+'       ','class=',self.left)
        else:
            self.left.print_tree(indent+'    ')
        print(indent,'if x['+str(self.attribute)+'] <=',self.threshold)
        if self.right  in [0,1]:
            print(indent+'       ','class=',self.right)
        else:
            self.right.print_tree(indent+'    ')

class DecisionTreeClassifier(object):
    # constructor
    def __init__(self, max_depth=10, min_samples_split=10, accuracy=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.accuracy = accuracy
        
    def fit(self, data, labels):
        self.root = self._build_tree(data, labels, depth=0)
        
    def predict(self, test_data):
        pred = np.zeros(len(test_data), dtype=int)
        for i in range(len(test_data)):
            pred[i] = self._predict_example(test_data[i], self.root)
        return pred
        
    def _build_tree(self, data, labels, depth):
        # check the base case (termination condition)
        mean_val = np.mean(labels)
        if depth==self.max_depth or len(data)<= self.min_samples_split or max(
                [mean_val, 1-mean_val])>= self.accuracy:
            return int(round(mean_val))
        else:
            depth += 1
            #all_thrs = np.mean(data, axis=0) # uncomment this line only to use mean as therashold
            all_thrs = self._get_all_thrs(data) #comment this line only to use mean as threshold
            # get the best attribute and threshold wth the highest gain
            best_split_col, best_split_val = self._get_best_split(data, labels, all_thrs)
            less, more = self._split_data(data, best_split_col, best_split_val)
            #recursivly build the tree
            left = self._build_tree(data[less], labels[less], depth)
            right = self._build_tree(data[more], labels[more], depth)
            

        return DecisionTreeNode(best_split_col, best_split_val, left, right)
    
    def _get_all_thrs(self, data):
        all_thrs = {}
        for index in range(data.shape[1]):
            all_thrs[index] = []
            unique_val = np.unique(data[:,index])
            
            for idx in range(len(unique_val)):
                if idx != 0:
                    current_val = unique_val[idx]
                    previous_val = unique_val[idx - 1]
                    thr = (current_val + previous_val)/2
                    all_thrs[index].append(thr)
        return all_thrs
        
    # Find the best cloumn to classify and the best threshold value
    def _get_best_split(self, data, labels, all_thrs):
        best_entropy = 999
        for col_index in range(data.shape[1]):
            for thr in all_thrs[col_index]: # comment this line only to use mean as threshold
            #thr = all_thrs[col_index] # uncomment this line and correct the indentation only to use mean as threshold
                less, more = self._split_data(data, col_index, thr)
                ent = self._entropy(labels[less], labels[more])
                
                if ent < best_entropy:
                    best_entropy = ent
                    best_split_col = col_index
                    best_split_val = thr
        return best_split_col, best_split_val
    
    def _split_data(self, data, split_col, thr):
        less = data[:,split_col] <= thr
        more = data[:, split_col] > thr
        return less, more

    # calculate entropy
    def _entropy(self, l, m):
        ent = 0
        for p in [l, m]:
            if len(p) > 0:
                pp = sum(p)/len(p)
                pn = 1 - pp
                if pp < 1 and pp > 0:
                    ent -= len(p)*(pp*np.log2(pp) + pn*np.log2(pn))
        ent = ent / (len(l) + len(m))
        return ent
        
    def _predict_example(self, example, tree):
        col_id = tree.attribute
        val = tree.threshold
    
        if example[col_id] <= val:
            answer = tree.left
        else:
            answer = tree.right
            
        if not isinstance(answer, DecisionTreeNode):
            return answer
        else:
            remaining_tree = answer
            return self._predict_example(example, remaining_tree)
     
    def display(self):
        print("model")
        self.root.print_tree()
        
    def confusion_matrix(self, pred, labels):
        cm = np.zeros((np.max(pred)+1, np.max(pred)+1))
        for i in range(len(pred)):
            cm[pred[i]][labels[i]] += 1
        return cm
    
# load and prepare data
x = []
y = []
infile = open("gamma_ray.txt","r")
for line in infile:
    y.append(int(line[-2:-1] =='g'))
    x.append(np.fromstring(line[:-2], dtype=float,sep=','))
infile.close()
    
x = np.array(x).astype(np.float32)
y = np.array(y) 

#Split data into training and testing
ind = np.random.permutation(len(y))
split_ind = int(len(y)*0.8)
x_train = x[ind[:split_ind]]
x_test = x[ind[split_ind:]]
y_train = y[ind[:split_ind]]
y_test = y[ind[split_ind:]]

# take a toy dataset
x_train = x_train[:5000]
x_test = x_test[:500]
y_train = y_train[:5000]
y_test = y_test[:500]

model = DecisionTreeClassifier(max_depth=5)
start = time.time()
model.fit(x_train, y_train)
elapsed_time = time.time()-start
print('Elapsed_time training  {0:.6f} '.format(elapsed_time))  

# display the model
print("model")
model.display() 

# train accuracy and confusion matrix
train_pred = model.predict(x_train)
train_acc = np.sum(train_pred == y_train)/len(train_pred)
print("Confusion matrix for training:\n", model.confusion_matrix(train_pred, y_train))
print('Train accuracy: ', train_acc) 

# test accuracy and confusion matrix
start = time.time()
test_pred = model.predict(x_test)
elapsed_time = time.time() - start
print('Elapsed_time testing {0:.6f}'.format(elapsed_time))

test_acc = np.sum(test_pred == y_test)/len(test_pred)
print("\nConfusion matrix for testing:\n", model.confusion_matrix(test_pred, y_test))
print('\nTest accuracy: ', test_acc)

# display 10 random prediction result
idx = np.random.randint(0, 500, 10)
dict = {"x[0]": x_test[idx,0], "x[1]": x_test[idx,1], "x[2]": x_test[idx,2], "x[3]": x_test[idx,3], "x[4]": x_test[idx,4],
        "x[5]": x_test[idx,5], "x[6]": x_test[idx,6], "x[7]": x_test[idx,7], "x[8]": x_test[idx,8], "x[9]": x_test[idx,9],
        "labels": y_test[idx], "prediction": test_pred[idx]}
df = pd.DataFrame(dict)
pd.set_option('display.width', 1000)
print(df)