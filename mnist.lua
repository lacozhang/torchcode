require 'torch'
require 'nn'
require 'optim'
require 'xlua'
require 'logroll'


cmdLogger = logroll.print_logger()


cmdParser = torch.CmdLine()

cmdParser:text('Train models on MNIST data sets')
cmdParser:option('-train', '', 'train data')
cmdParser:option('-test', '', 'test data of MNIST datasets')
cmdParser:option('-iter', 1e3, 'number of iterations')
cmdParser:option('-batch', 10, 'minibatch size')
cmdParser:option('-lr', 1e-3, 'learning rate')
cmdParser:option('-lrd', 1e-4, 'learning rate decay')
cmdParser:option('-wd', 0, 'l2 regularization parameter')

opt = cmdParser:parse(arg or {})

function load_data(datafilepath)
	local data = torch.load(datafilepath, 'ascii')
	local dataX = data.data:double()
	dataX = dataX:reshape(dataX:size(1), dataX:nElement()/dataX:size(1))
	dataY = data.labels
	return dataX,dataY
end

function eval_model( model, criteria, dataX, dataY )
	-- body
	return criteria:forward(model:forward(dataX), dataY)
end

function main(opts)

	if string.len(opts.train) == 0 then
		cmdLogger.error('must specify train data')
		os.exit(1)
	end

	if string.len(opts.test) == 0 then
		cmdLogger.error('must specify test data')
		os.exit(1)
	end


	logger = optim.Logger('mnist.log')
	logger:setNames{'train loss'}

	loggerForTest = optim.Logger('mnist.test.log')
	loggerForTest:setNames{'test loss'}

	trainX, trainY = load_data(opts.train)
	testX, testY = load_data(opts.test)
	batchsize = opts.batch

	ninputs = 1024
	noutputs=10

	model = nn.Sequential()
	model:add(nn.Linear(ninputs, noutputs))
	model:add(nn.LogSoftMax())

	criterion = nn.ClassNLLCriterion()

	params, gradParams = model:getParameters()

	optConfig = {
	   learningRate=1e-3,
	   learningRateDecay=1e-4,
	   weightDecay=0,
	   momentum=0,
	   nesterov=false
	}

	for i=1,opts.iter do

		local time = sys.clock()
		local shuffle = torch.randperm(trainX:size(1))

		cmdLogger.info('start iter ' .. i)
		for startidx=1,trainX:size(1),batchsize do

			local inputs = {}
			local targets = {}
			xlua.progress(startidx, trainX:size(1))

			local indextable = {}

			for j=startidx, math.min(trainX:size(1), startidx + batchsize-1) do
				local realidx = shuffle[j]
				table.insert(indextable, realidx)
			end

			indextable = torch.LongTensor(indextable)

			inputs = trainX:index(1, indextable)
			targets = trainY:index(1, indextable):double()


			local feval = function ( new_x )
				-- body
				if params ~= new_x then
					params:copy(new_x)
				end
				gradParams:zero()

				local avgloss = 0.0
				local modelout = model:forward(inputs)
				local avgloss = criterion:forward(modelout, targets)
				local gradCriterion = criterion:backward(modelout, targets)
				model:backward(inputs, gradCriterion)

--				gradParams:div((#indextable)[1])

				return avgloss, gradParams
			end

			_, fs = optim.sgd(feval, params, optConfig)

			logger:add{fs[1]}
		end

		time = sys.clock() - time

		local epochsLoss = eval_model(model, criterion, trainX, trainY)

		local testLoss = eval_model(model, criterion, testX, testY)
		loggerForTest:add{testLoss}

		cmdLogger.info('\neach sample costs ' .. (time / trainX:size(1))*1000 .. ' ms')
		cmdLogger.info('train loss is ' .. epochsLoss / (trainX:size(1)))
		cmdLogger.info('test loss is ' .. testLoss)
	end

	loggerForTest:style{'-'}
	loggerForTest:plot()

	logger:style{'-'}
	logger:plot()
end

main(opt)