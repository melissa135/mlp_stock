import os.path
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from define_network import MLP
from sample_set import Sample_set
from torch.autograd import Variable


def incorrectness(mlp,loader):
    
    correct = 0.0
    incorrect = 0.0
    for i,data in enumerate(loader,0):
        input,target = data
        input = Variable(input)
        actual = target[0][0]
	output = mlp(input.float())
	if (output.data[0][0] * actual) >= 0:
	    correct = correct + 1
	else :
	    incorrect = incorrect + 1
    return incorrect/(correct+incorrect)


def early_stop(lst,ma):

    length = len(lst)
    sum_a = sum(lst[length - ma:length])
    length = length - 1
    sum_b = sum(lst[length - ma:length])
   
    if sum_a > sum_b:
	return True
    else :
	return False


def train(trainloader,testloader):

    max_epochs = 100
    moving_average = 30

    mlp = MLP()
    print mlp

    criterion = nn.L1Loss()
    optimizer = optim.Adam(mlp.parameters(),lr=0.001)

    mlp_list = []
    crt_list = []

    for epoch in range(0, max_epochs):

        current_loss = 0
        for i,data in enumerate(trainloader,0):

            input,target = data
            input,target = Variable(input),Variable(target)

            mlp.zero_grad()
            output = mlp(input.float())
            loss = criterion(output, target.float())
      
            loss.backward()
            optimizer.step()

            loss = loss.data[0]
            current_loss += loss

        #print ('[ %d ] loss : %.3f' % (epoch+1,current_loss))

        train_c = incorrectness(mlp,trainloader)
        test_c = incorrectness(mlp,testloader)
        print ('[ %d ] incorrectness: %.4f %.4f' % (epoch+1,train_c,test_c))
	
	mlp_list.append(mlp)
	crt_list.append(train_c+test_c)

	if epoch >= moving_average:
	    if early_stop(crt_list,moving_average):
		print 'Early stopping.'
		index = len(mlp_list) - moving_average/2
		return mlp_list[index]

        current_loss = 0
        
    return mlp


if __name__ == '__main__':

    path_ = os.path.abspath('.')

    ensembles = 5
    batchsize = 8

    trainset = Sample_set(path_+'/sz_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset,batch_size=batchsize,shuffle=True,num_workers=2)

    testset = Sample_set(path_+'/sz_test.csv')
    testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)

    for i in range(0,ensembles):
        
	print 'Training %d-th MLP.' % i
        mlp = train(trainloader,testloader)
        torch.save(mlp.state_dict(),path_+'/MLPs/mlp_%d.pth' % i)

