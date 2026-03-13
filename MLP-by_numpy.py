import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

np.random.seed(42)
W1=np.random.uniform(-1,1,size=(2,4))
W2=np.random.uniform(-1,1,size=(4,1))
b1=np.zeros((1,4))
b2=np.zeros((1, 1))

epochs = 10000
learning_rate = 0.5

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def sig(out):
    return out*(1-out)

for epoch in range(epochs):
    Z1=np.dot(X,W1)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(A1,W2)+b2
    A2=sigmoid(Z2)

    dA2=A2-y
    dZ2=dA2*sig(A2)
    dW2=np.dot(A1.T,dA2*sig(A2))
    dZ1=np.dot(dZ2,W2.T)*sig(A1)
    dW1=np.dot(X.T,dZ1)
    db2=np.sum(dZ2,axis=0,keepdims=True)
    db1=np.sum(dZ1,axis=0,keepdims=True)

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    print("训练后的预测结果：")
    print(A2)




