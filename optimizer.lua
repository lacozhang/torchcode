require 'torch'
require 'nn'
require 'optim'
require 'xlua'

parser = torch.CmdLine()
parser:text('compare the performance of different optimization algorithm')
parser:option(

