import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, input_size=10, layer1_size=32, layer2_size=16, output_size=1):
        super(MLP, self).__init__()

	self.input_size = input_size       
	self.layer1_size = layer1_size
	self.layer2_size = layer2_size
	self.output_size = output_size

        self.layer1 = nn.Linear(input_size, layer1_size)
	self.layer2 = nn.Linear(layer1_size, layer2_size)
        self.output = nn.Linear(layer2_size, output_size)

    def forward(self, x):
        x = F.dropout(self.layer1(x), p=0.1)
	x = F.tanh(x)
        x = F.tanh(self.layer2(x))
	x = self.output(x)
	#print x
        return x
