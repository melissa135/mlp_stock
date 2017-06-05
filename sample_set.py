import torch
import torch.utils.data as data
from pandas.io.parsers import read_csv

class Sample_set(data.Dataset):

    def __init__(self, filename):

	df = read_csv(filename)
	self.df = df
        self.w1 = 1
        self.w2 = 0.1

    def __getitem__(self, index):
	index = index + 5

        data = [ self.df['close_change'][index-1]*self.w1,
                 self.df['close_change'][index-2]*self.w1,
                 self.df['close_change'][index-3]*self.w1,
                 self.df['close_change'][index-4]*self.w1,
                 self.df['close_change'][index-5]*self.w1,
                 self.df['volume_change'][index-1]*self.w2,
                 self.df['volume_change'][index-2]*self.w2,
                 self.df['volume_change'][index-3]*self.w2,
                 self.df['volume_change'][index-4]*self.w2,
                 self.df['volume_change'][index-5]*self.w2 ]

	target = [ self.df['close_change'][index] ]

	data = torch.Tensor(data)
        target = torch.Tensor(target)
	return data, target

    def __len__(self):
        return len(self.df) - 5
