import numpy as np
import pandas as pd
import time

class RegressionTreeNode(object):
    # Constructor
    def __init__(self, att, thr, left, right):  
        self.attribute = att
        self.threshold = thr
        # left and right are either binary classifications or references to 
        # decision tree nodes
        self.left = left     
        self.right = right   

    def print_tree(self,indent=''):
        # If prints the right subtree, corresponding to the condition x[attribute] > threshold
        # above the condition stored in the node
        if isinstance(self.right, np.float64):
            print(indent+'       ','pred=',self.right)
        else:
            self.right.print_tree(indent+'    ')
        
        print(indent,'if x['+str(self.attribute)+'] <=',self.threshold)
        
        if isinstance(self.left, np.float64):
            print(indent+'       ','pred=',self.left)
        else:
            self.left.print_tree(indent+'    ')

class RegressionTreeClassifier(object):
    # constructor
    def __init__(self, max_depth=10, min_samples_split=10, accuracy=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.accuracy = accuracy
        
    def fit(self, data, labels):
        self.root = self._id3(data, labels, depth=0)
        
    def predict(self, test_data):
        pred = np.zeros(len(test_data), dtype=np.float64)
        for i in range(len(test_data)):
            pred[i] = self._predict(self.root, test_data[i])
        return pred
        
    def _id3(self, data, labels, depth):
        # check the base case (termination condition)
        mean_val = np.mean(labels)
        if depth==self.max_depth or len(data)<= self.min_samples_split or max(
                [mean_val, 1-mean_val])>= self.accuracy:
            return mean_val
        else:
            depth += 1
            all_thrs = self._get_all_thrs(data)
            # get the best attribute and threshold wth the highest gain
            best_split_col, best_split_val = self._get_best_split(data, labels, all_thrs)
            less, more = self._split_data(data, best_split_col, best_split_val)
            #recursivly build the tree
            left = self._id3(data[less], labels[less], depth)
            right = self._id3(data[more], labels[more], depth)

        return RegressionTreeNode(best_split_col, best_split_val, left, right)
    
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
        best_mse = 999
        for col_index in range(data.shape[1]):
            for thr in all_thrs[col_index]:
                less, more = self._split_data(data, col_index, thr)
                mse = self._calculate_mse(labels[less], labels[more])
                
                if mse < best_mse:
                    best_mse = mse
                    best_split_col = col_index
                    best_split_val = thr
        return best_split_col, best_split_val
    
    def _split_data(self, data, split_col, thr):
        less = data[:,split_col] <= thr
        more = data[:, split_col] > thr
        return less, more
    
    def _calculate_mse(self, y_below, y_above):
        left = np.mean(y_below)
        right = np.mean(y_above)
        mse = np.sum((y_below-left)**2) + np.sum((y_above-right)**2)
        return mse
   
    def _predict(self, dt_node, x):
        if isinstance(dt_node, np.float64):
            return dt_node
        if x[dt_node.attribute] <= dt_node.threshold:
            return self._predict(dt_node.left, x)
        else:
            return self._predict(dt_node.right, x)
     
    def display(self):
        print("model")
        self.root.print_tree()
    
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

# take a small dataset
x_train = x_train[:5000]
x_test = x_test[:500]
y_train = y_train[:5000]
y_test = y_test[:500]

model = RegressionTreeClassifier(max_depth=5)
start = time.time()
model.fit(x_train, y_train)
elapsed_time = time.time()-start
print('Elapsed_time training  {0:.6f} '.format(elapsed_time)) 

# display the regression tree
print("model")
model.display() 

train_pred = model.predict(x_train)
print('Mean square error test set:',np.mean(np.square(train_pred-y_train)))

start = time.time()
test_pred = model.predict(x_test)
elapsed_time = time.time() - start
print('Elapsed_time testing {0:.6f}'.format(elapsed_time))
print('Mean square error test set:',np.mean(np.square(test_pred-y_test)))

# display 10 decision regression results
idx = np.random.randint(0, 500, 10)
dict = {"x[0]": x_test[idx,0], "x[1]": x_test[idx,1], "x[2]": x_test[idx,2], "x[3]": x_test[idx,3], "x[4]": x_test[idx,4],
        "x[5]": x_test[idx,5], "x[6]": x_test[idx,6], "x[7]": x_test[idx,7], "x[8]": x_test[idx,8], "x[9]": x_test[idx,9],
        "labels": y_test[idx], "prediction": test_pred[idx]}
df = pd.DataFrame(dict)
pd.set_option('display.width', 1000)
print(df) 