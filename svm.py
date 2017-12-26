import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm

print("Reading data...")
data = np.genfromtxt('data_with_plus.csv',delimiter=',')
data = data[1:,:]
X = data[:,1:]
Y = data[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train = X_train/255.0
X_test = X_test/255.0

print(X_train[1,:])

print("Fitting Model...")
model = svm.SVC(C=5,gamma=0.05)
model.fit(X_train, Y_train)

#save model
pickle.dump(model,open('svm.sav', 'wb'))
 
 
# load the model from disk
#print("Loading model...")
#model = pickle.load(open('model2.sav', 'rb'))
predictions = model.predict(X_test)
#print(predictions)
total = 0
correct = 0
for i,pred in enumerate(predictions):
    if(pred == Y_test[i]):
        correct += 1
    total += 1
accuracy = correct/total
print(accuracy)
#0.9801756885090218