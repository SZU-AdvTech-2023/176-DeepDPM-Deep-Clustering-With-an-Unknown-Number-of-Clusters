import torch
import matplotlib.pyplot as plt
x=torch.linspace(0,100,100).type(torch.FloatTensor)
rand=torch.randn(100)*10
y=x+rand
x_train=x[:-10]
x_test=x[-10:]
y_train=y[:-10]
y_test=y[-10:]
plt.figure(figsize=(10,8))
plt.title('house price')
# plt.scatter(x_train.data.numpy(),y_train.data.numpy(),marker='o')
# plt.xlim(0,100)
# plt.ylim()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
a=torch.rand(1,requires_grad=True)
b=torch.rand(1,requires_grad=True)
learning_rate=0.0001
for i in range(1000):
    predictions=a.expand_as(x_train)*x_train+b.expand_as(x_train)
    loss=torch.mean((predictions-y_train)**2)
    print("loss",loss)
    loss.backward()
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad

        a.grad.zero_()
        b.grad.zero_()
x_data=x_train.numpy()
plt.figure(figsize=(10,7))
xplot,=plt.plot(x_data,y_train.data.numpy(),'o')
yplot,=plt.plot(x_data,a.data.numpy()*x_data+b.data.numpy())
plt.xlabel('X')
plt.ylabel('Y')
str1=str(a.data.numpy()[0])+'x+'+str(b.data.numpy()[0])
plt.legend([xplot,yplot],['Data',str1])
plt.show()
