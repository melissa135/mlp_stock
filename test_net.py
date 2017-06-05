import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from define_network import MLP
from sample_set import Sample_set

if __name__ == '__main__':

    path_ = os.path.abspath('.')
    dirt = path_ + '/MLPs/'

    mlps = []

    for root, _, fnames in sorted(os.walk(dirt)):
        for fname in fnames:
            path = os.path.join(root, fname)
            mlp = MLP()
            mlp.load_state_dict(torch.load(path))
            mlps.append(mlp)

    testset = Sample_set(path_+'/sz_test.csv')
    testloader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)

    sz = 1.0
    asset = 1.0
    sz_list = []
    asset_list = []

    for i,data in enumerate(testloader,0):

        input,target = data
        input = Variable(input)
        actual = target[0][0]

        sum_output = 0.0
        for mlp in mlps:
            output = mlp(input.float())
            sum_output = sum_output + output.data[0][0]
        #print sum_output

        sz = sz * (1+actual/100.0)
        if sum_output >= 0.0 :
            asset = asset * (1+actual/100.0)
        print sz,asset
        sz_list.append(sz)
        asset_list.append(asset)

    X = range(0,len(sz_list))
    plt.figure(figsize=(12,8),dpi=80)
    plt.plot(X,sz_list,color='black',linewidth=1)
    plt.plot(X,asset_list,color='red',linewidth=1)
    #plt.savefig(path_+'/asset.png')
    plt.show()
