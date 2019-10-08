import numpy as np
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
            all_thrs = self._get_all_thrs(data) 
            # get the best attribute and threshold wth the highest gain
            best_split_col, best_split_val = self._get_best_split(data, labels, 
                                                                  all_thrs)
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
            for thr in all_thrs[col_index]: 
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

def load_gamma_ray(standarize=True):
    # load and prepare data
    x = []
    y = []
    infile = open("magic04.txt","r")
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
    
    skip = 2   
    x_train = x_train[::skip]
    y_train = y_train[::skip]
    x_test = x_test[::skip]
    y_test = y_test[::skip]
    
    if standarize:
        s = np.std(x_train, axis = 0)
        mu = np.mean(x_train, axis = 0)
        x_train = (x_train - s)/mu
        
        s = np.std(x_test, axis = 0)
        mu = np.mean(x_test, axis = 0)
        x_test = (x_test - s)/mu
    
    return x_train, y_train, x_test, y_test
 
def prob_select(prob, n):
    cs = np.cumsum(prob)
    R = np.sort(np.random.rand(n))
    S = []
    i = 0
    for r in R:
        while r > cs[i]:
            i += 1
        S.append(i)
    return S

def update_prob(wrong_idx, p):
    for i in range(len(wrong_idx)):
        if wrong_idx[i] == True:
            p[i] = p[i]*1.5
    return p/np.sum(p)

def max_vote(pred):
    pred = np.array(pred).T
    ens_pred = []
    for i in range(len(pred)):
        freq = np.array(np.unique(pred[i,:], return_counts=True)).T
        idx = np.argmax(freq[:,1])
        p = freq[idx,0]
        ens_pred.append(p)
    return ens_pred


def randomforest(x_train, y_train, x_test, y_test, trees=5, max_depth=10, 
                 ensemble="boosting"):
    if ensemble == "boosting":
        print("Randomforest with boosting....")
        prob = [1/len(x_train)]*len(x_train)
        pred = []
        for tree in range(trees):
            prob_idx = prob_select(prob, int(len(x_train)*0.8))
            p_x_train = x_train[prob_idx,:]
            p_y_train = y_train[prob_idx]
            
            model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(p_x_train, p_y_train)
            
            train_pred = model.predict(x_train)
            train_acc = np.mean(train_pred == y_train)
            
            test_pred = model.predict(x_test)
            test_acc = np.mean(test_pred == y_test)
            
            print("Tree:{}/{}  Train accuracy: {}  Test accuracy: {}"
                  .format(tree+1, trees, train_acc, test_acc))
            
            wrong_idx = train_pred != y_train
            prob = update_prob(wrong_idx, prob)
            
            pred.append(test_pred)
        return pred
    
    if ensemble=="bagging":
        print("Randomforest with bagging....")
        pred = []
        for tree in range(trees):
            idx = np.random.randint(len(x_train), size=int(len(x_train)*0.8))
            b_x_train = x_train[idx,:]
            b_y_train = y_train[idx]
            
            model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(b_x_train, b_y_train)
            
            train_pred = model.predict(x_train)
            train_acc = np.mean(train_pred == y_train)
            
            test_pred = model.predict(x_test)
            test_acc = np.mean(test_pred == y_test)
            
            print("Tree:{}/{}  Train accuracy: {}  Test accuracy: {}"
                  .format(tree+1, trees, train_acc, test_acc))
            
            pred.append(test_pred)
        return pred
    
    if ensemble=="randomization":
        print("Randomforest with randomization....")
        pred = []
        for tree in range(trees):
            idx = np.random.permutation(len(x_train))
            r_x_train = x_train[idx[:int(len(x_train)*0.8)]]
            r_y_train = y_train[idx[:int(len(x_train)*0.8)]]
    
            model = DecisionTreeClassifier(max_depth=max_depth)
            model.fit(r_x_train, r_y_train)
            
            train_pred = model.predict(x_train)
            train_acc = np.mean(train_pred == y_train)
            
            test_pred = model.predict(x_test)
            test_acc = np.mean(test_pred == y_test)
            
            print("Tree:{}/{}  Train accuracy: {}  Test accuracy: {}"
                  .format(tree+1, trees, train_acc, test_acc))
            
            pred.append(test_pred)
        return pred
        
    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_gamma_ray()
    
    start = time.time()
    
    method = "boosting"
    max_depth = 3
    trees = 3
    pred = randomforest(x_train, y_train, x_test, y_test, trees=trees, 
                        max_depth=max_depth, ensemble=method)

    ens_pred = max_vote(pred)
    elapsed_time = time.time() - start
    print("Ensemble time: {0:.6f} sec".format(elapsed_time))
    
    accuracy = np.mean(ens_pred == y_test)
    print("Ensemble test accuracy: {}".format(accuracy)) 
    