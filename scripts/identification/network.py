from torch import nn
class ClassificationHead(nn.Module):
    def __init__(self,dim_out,dim_in):
        super(ClassificationHead, self).__init__()
        self.out_features = dim_out
        self.linear = nn.Linear(in_features=dim_in, out_features=self.out_features)

    def forward(self, x):
        return self.linear(x)