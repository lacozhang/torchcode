require 'nn'
require 'xlua'
local loader = require 'iris_loader'

data = loader.load_data()

input_size = 4
output_size = 3

-- define original model