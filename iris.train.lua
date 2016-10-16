require 'nn'
require 'optim'
require 'xlua'
require 'requ'

local loader = require 'iris_loader'
local logger = optim.Logger("iris.log")
logger:setNames{'train error'}

torch.manualSeed(0)

data = loader.load_data()

input_size = 4
embedding_size = 2
output_size = 3

-- define original model
model = nn.Sequential()
model:add(nn.Linear(input_size, embedding_size))

-- define usual activation function
-- model:add(nn.Tanh())
model:add(nn.ReQU())

model:add(nn.Linear(embedding_size, output_size))
model:add(nn.LogSoftMax())

criteria = nn.ClassNLLCriterion()

-- model definition finished

param, gradParam = model:getParameters()
param:uniform(-0.01, 0.01)

for _, lin in pairs(model:findModules('nn.Linear')) do
    lin.bias:add(0.5)
end


function feval(x)

    if x ~= param then
        param:copy(x)
    end
    gradParam:zero()

    local outputs = model:forward( data.inputs )
    local loss = criteria:forward( outputs, data.targets )
    model:backward( data.inputs, criteria:backward(outputs, data.targets) )

    return loss, gradParam
end

function modelval()
    return criteria:forward(model:forward(data.inputs), data.targets)
end

function grad_check()
    param:normal(0, 0.01 ^ 2)
    local tempGrad = torch.Tensor()
    tempGrad:resizeAs(gradParam)
    tempGrad:zero()

    eps = 1e-7
    for i=1,param:nElement() do

        local temp = param[i]

        param[i] = param[i] + eps
        local before = modelval()
        param[i] = param[i] - 2*eps
        local after = modelval()
        tempGrad[i] = (before - after) / (2*eps)
        param[i] = temp
    end

    feval(param) 

    local erate = torch.norm(gradParam - tempGrad) / torch.norm(gradParam + tempGrad)
    print(erate)
end

grad_check()


sgdConfig = { learningRate = 0.1, learningRateDecay = 1e-1 }

for iter=1,200 do
    _, l = optim.sgd(feval, param, sgdConfig)
    logger:add{l[1]}
end
