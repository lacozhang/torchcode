require 'torch'
require 'nn'
require 'optim'
require 'os'

data = torch.Tensor{
   {40,  6,  4},
   {44, 10,  4},
   {46, 12,  5},
   {48, 14,  7},
   {52, 16,  9},
   {58, 18, 12},
   {60, 22, 14},
   {68, 24, 20},
   {74, 26, 21},
   {80, 32, 24}
}


std_result = {40.32, 42.92, 45.33, 48.85, 52.37, 57, 61.82, 69.78, 72.19, 79.42}


--- define model
model = nn.Sequential()
ninputs=2
noutputs=1
model:add( nn.Linear(ninputs, noutputs,false) )

--- define loss function

loss = nn.MSECriterion()

--- get parameters
x, dl_dx = model:getParameters()

niters=1e4

sgdParams = {
   learningRate=1e-3,
   learningRateDecay=1e-4,
   weightDecay=0,
   momentum=0,
   nesterov=false
}

--- start logger
logger = optim.Logger('lr.log')
logger:setNames({'loss', 'error'})

model:training()

for i=1,niters do

   print('iter ' .. i)

   lossvalue = 0

   local time = sys.clock()
   local shuffle = torch.randperm((#data)[1])
   
   for idx = 1, (#data)[1] do

      local feval = function(x_new)

	 if x ~= x_new then
	    x:copy(x_new)
	 end

	 dl_dx:zero()

	 local realidx = shuffle[idx]
	 local sample = data[realidx]
	 local input = sample[{{2,3}}]
	 local target = sample[{{1}}]

	 local modelPredict = model:forward(input)
	 local localloss = loss:forward(modelPredict, target)
	 model:backward(input, loss:backward(modelPredict, target))
	 lossvalue = lossvalue + localloss
	 return localloss, dl_dx
      end

      _,fs = optim.sgd(feval, x, sgdParams)
      
      lossvalue = lossvalue + fs[1]

      testloss = 0
      for i=1,(#data)[1] do
	 testloss = (model:forward(data[i][{{2,3}}])[1] - std_result[i]) ^ 2
      end
      testloss = testloss / (#data)[1]
      
      logger:add{lossvalue, testloss}
   end

   time = sys.clock() - time
   time = time / (#data)[1]
   print('average time for train a sample ' .. time*1000 .. ' ms')
   lossvalue = lossvalue / (#data)[1]
   print('average loss value is ' .. lossvalue)
end

logger:style{'-', '#'}
logger:plot()

---
print('solution provided by sgd')
print(x)

--- solution based on matrix solution
X = data[{{}, {2,3}}]
y = data[{{}, {1}}]

sol = torch.inverse(X:t()* X )* X:t()  * y
print(sol)
