require 'torch'
require 'nn'
require 'optim'
require 'xlua'
require 'os'

parser = torch.CmdLine()
parser:text('compare the performance of different optimization algorithm')
parser:option('-datadir', '/home/cazhan/src/torchtutorials/A_datasets/mnist.t7/', 'directory for train and test data')
parser:option('-opt', 'sgd', 'optimization method')
parser:option('-epochs', 100, 'number of training epochs')
parser:option('-batch', 128, 'batch size of each mini-batch')
parser:option('-help', false, 'need help or not')

-- 
parser:text('first order related option')
parser:option('-lr', 1e-6, 'initial learning rate')
parser:option('-lrd', 1e-5, 'learning rate decay')
parser:option('-wd', 1e-6, 'the l1 norm minimization method')
parser:option('-mom', 0, 'the momentum of gradient')
parser:option('-damp', 0, 'dampening of sgd')
parser:option('-nest', false, 'use nesterov or not')
parser:option('-log', 'train.log', 'log file name')

-- batch parameter
parser:text('batch optimization related')
parser:option('-maxiter', 50, 'max number of iterations for batch method')
parser:option('-feps', 1e-5, 'terminate criteria for function value change')
parser:option('-xeps', 1e-9, 'terminate criteria for parameter value change')
parser:option('-hist', 10, 'number of history for L-BFGS')
parser:option('-linesearch', 'func', 'line search method')


opt = parser:parse(arg or {})

if opt.help then
    print(parser)
    os.exit(0)
end

if opt.datadir == '' then
    print('msut provide data directory')
    os.exit(1)
end 
train_file = opt.datadir .. 'train_32x32.t7'
extra_file = opt.datadir .. 'extra_32x32.t7'
test_file = opt.datadir .. 'test_32x32.t7'

trsize = 60000 
testsize = 10000

loaded = torch.load(train_file, 'ascii')
trainData = {
    data = loaded.data:double(),
    labels = loaded.labels:long(),
    size = function() return trsize end
}

print('train data size')
print(#trainData.data)

loaded = torch.load(test_file, 'ascii')
testData = {
    data = loaded.data:double(),
    labels = loaded.labels:long(),
    size = function() return testsize end
}

ninputs=32*32
noutputs=10

print('test data' )
print(#testData.data)

torch.manualSeed(0)

perfLogger = optim.Logger(opt.log)
perfLogger:setNames({'trainloss', 'testloss', 'trainAccu', 'testAccu'})

-- define model & criteria
model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs, noutputs))

criteria = nn.CrossEntropyCriterion()

parameters, gradParameters = model:getParameters()
classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'}


function accuracy_perf(predMatrix, labels)
    _, idx = torch.max(predMatrix, 2)
    local errCnt=torch.sum(torch.eq(labels, idx))
    local accu = errCnt / labels:size(1)
    return accu
end

function loss_perf(predMatrix, labels)
    local f = 0.0
    for i=1,predMatrix:size(1) do
        f = f + criteria:forward(predMatrix[i], labels[i])
    end
    f = f / labels:size(1)
    return f
end

function eval_perf()

    model:evaluate()
    local trainPred = model:forward(trainData.data)
    local trainLoss = loss_perf(trainPred, trainData.labels)
    local trainAccuracy = accuracy_perf(trainPred, trainData.labels)

    local testPred = model:forward(testData.data)
    local testLoss = loss_perf(testPred, testData.labels)
    local testAccuracy = accuracy_perf(testPred, testData.labels)
    
    return trainLoss,testLoss,trainAccuracy,testAccuracy
end

function batch_train(opt, optMethod, optParam)

    timer = torch.Timer()
    for i=1,opt.batch do

        print('iter ' .. i)
        timer:reset()
        local feval = function(x)
    
            if parameters ~= x then
                parameters:copy(x)
            end
            gradParameters:zero()

            local pred = model:forward(trainData.data)
            local ferr = criteria:forward(pred, trainData.labels)        
            model:backward(trainData.data, criteria:backward(pred, trainData.labels))
            gradParameters:div(trainData.size())
            return ferr, gradParameters
        end

        _, fvals = optMethod(feval, parameters, optParam)
        print(fvals)
        trainLoss, testLoss, trainAccu, testAccu = eval_perf()
        perfLogger:add({trainLoss, testLoss, trainAccu, testAccu})
        print('costs ' .. (timer:time().real) .. ' seconds')
    end
end

function first_order_train(opt, sgdMethod, sgdParam)

    timer = torch.Timer()
    for epochIter = 1, opt.epochs do

        local trainloss, testloss, trainaccu, testaccu = eval_perf()
        print('epoch number ' .. epochIter)
        print('train loss ' .. trainloss .. ' |test loss ' .. testloss .. ' |train accu ' .. trainaccu .. ' |test accu ' .. testaccu)
        perm = torch.randperm(trainData:size())

        timer:reset()
        for sampleIndex=1,trainData:size(), opt.batch do

            local inputs = {}
            local targets = {}

            model:training()
            for t = sampleIndex, math.min(sampleIndex+opt.batch-1, trainData:size()) do
                table.insert(inputs, trainData.data[perm[t]])
                table.insert(targets, trainData.labels[perm[t]])
            end

            local feval = function(x)
                if x ~= parameters then
                    parameters:copy(x)
                end
                local f = 0.0
                gradParameters:zero()
                for i=1, #inputs do
                    local output = model:forward(inputs[i])
                    local err = criteria:forward(output, targets[i])
                    f = f + err
                    model:backward(inputs[i], criteria:backward(output, targets[i]))
                end
                gradParameters:div(#inputs)
                f = f / #inputs
                return f, gradParameters
            end 

            sgdMethod(feval, parameters, sgdParam)

            trainLoss, testLoss, trainAccu, testAccu = eval_perf()
            perfLogger:add({trainLoss, testLoss, trainAccu, testAccu})
        end
        print('every sample costs ' .. ((timer:time().real)/trainData.size())*1000 .. ' ms')
    end 
end

if opt.opt == 'sgd' then

    print('opt Sgd')
    config = {
        learningRate = opt.lr,
        learningRateDecay = opt.lrd,
        weightDecay = opt.wd,
        momentum = opt.mom,
        dampening = opt.damp,
        nesterov = opt.nest
    }
    first_order_train(opt, optim.sgd, config)
elseif opt.opt == 'adagrad' then
    print('opt AdaGrad')
    config = {
        learningRate = opt.lr,
        learningRateDecay = opt.lrd,
        weightDecay = opt.wd
    }
    first_order_train(opt, optim.adagrad, config)
elseif opt.opt == 'cg' then
    print('opt CG')
    config = {
        maxIter = opt.maxiter
    }
    batch_train(opt, optim.cg, config)
elseif opt.opt == 'lbfgs' then
    print('opt L-BFGS')
    config = {
        maxIter = opt.maxiter,
        tolFun = opt.feps,
        tolX = opt.xeps,
        nCorrection = opt.hist,
        learningRate = opt.lr,
        verbose = true
    }
    if opt.linesearch == 'func' then
        print('using linesearch with wolfe condition')
        config.lineSearch = optim.lswolfe
        config.lineSearchOptions = {}
    end
    batch_train(opt, optim.lbfgs, config)
end

