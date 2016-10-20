require 'torch'
require 'nn'
require 'nngraph'
require 'xlua'
require 'optim'

cmd = torch.CmdLine()
cmd:text()
cmd:option("-seed", 0, "random seed for peusdo random number generator")
cmd:option("-train", "../data/ptb.train.txt", "train file path")
cmd:option("-valid", "../data/ptb.valid.txt", "validation file path")
cmd:option("-test", "../data/ptb.test.txt", "test file path")
cmd:option("-batchsize", 16, "mini-batch size")
cmd:option("-seqlen", 10, "length of each sequence")
cmd:option("-embed", 100, "size of embedding")
cmd:option("-rnnsize", 100, "size of hidden layer")
cmd:option("-epochs", 100, "number of training epochs")
cmd:option("-clip", 5, "maximum value of gradient")
cmd:option("-lr", 1e-3, "learning rate for gradient update")
cmd:option("-lrd", 1e-4, "lr = lr_0 / (1 + nevals * lrd)")
cmd:option("-saveiter", 5, "save model every n iterations")

params = cmd:parse(arg)

function load_data(filepath)
    local f = io.open(filepath, "r")
    if not f then
        error("open file" .. filepath .. " failed")
    end

    local l = f:read("*a"):gsub("\n", "<eos>")
    local words = {}
    for word in l:gmatch("%S+") do
        words[#words+1] = word
    end
    return words
end

function build_vocab(words)
    local word2id = {}
    local wordid = 1
    for _, word in pairs(words) do
        if not word2id[word] then
            word2id[word] = wordid
            wordid = wordid + 1
        end
    end
    return word2id,wordid
end

function check_in_vocab(words, vocab)
    for _, word in pairs(words) do
        if not vocab[word] then
            print(word .. " not in vocab")
        end
    end
end

function build_nn_input(text, vocab, mbsize)
    
    -- convert text into tensor
    local wordtensor = torch.Tensor(#text)
    for idx,word in pairs(text) do
        if vocab[word] ~= nil then
            wordtensor[idx] = vocab[word]
        else
            error('word ' .. word .. ' do not appear in vocab')
        end
    end

    local Xwordtensor=torch.Tensor()
    local Ywordtensor=torch.Tensor()
    Xwordtensor:resizeAs(wordtensor:sub(1,-2))
    Xwordtensor:copy(wordtensor:sub(1,-2))

    Ywordtensor:resizeAs(wordtensor:sub(2,-1))
    Ywordtensor:copy(wordtensor:sub(2,-1))

    local function form_data_matrix(dat, rows, cols)
        local data_matrix = torch.Tensor(rows, cols)
        assert(data_matrix:nElement() <= dat:nElement())
        for i=1,mbsize do
            local start_index = (i-1)*rows + 1
            local end_index = i*rows
            data_matrix:narrow(2, i, 1):copy(dat:sub(start_index, end_index))
        end
        return data_matrix
    end

    local num_batches = torch.floor( Xwordtensor:size(1) / mbsize )

    Xdata_matrix = form_data_matrix(Xwordtensor, num_batches, mbsize)
    Ydata_matrix = form_data_matrix(Ywordtensor, num_batches, mbsize)
    return Xdata_matrix, Ydata_matrix
end

function clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

-- TODO: there is dropout parameter not used.
function build_rnn_network(vocab_size, embed_size, rnn_size, sequence_length, minibatch_size, dropout)

    local protos = {}
    local function build_part_network(vocab_size, embed_size, rnn_size, dropout)
        local input_t = nn.Identity()() 
        local prev_h = nn.Identity()()
        local embedding_t = nn.LookupTable(vocab_size, embed_size)(input_t)
        local curr_h = nn.Tanh()( nn.CAddTable()({nn.Linear(embed_size, rnn_size)(embedding_t), nn.Linear(rnn_size, rnn_size)(prev_h)}) )
        local output_t = nn.LogSoftMax()(nn.Linear(rnn_size, vocab_size)(curr_h))
        local rnn = nn.gModule({input_t, prev_h}, {curr_h, output_t})
        return rnn
    end

    protos.rnn = build_part_network(vocab_size, embed_size, rnn_size, dropout)
    protos.criteria = nn.ClassNLLCriterion()

    return protos
end

function state_table_clone(init)
    ret = {}
    for k,v in pairs(init) do
        ret[k] = v:clone()
    end
    return ret
end

local text_train = load_data(params.train)
local train_vocab,vocab_size = build_vocab(text_train)
local trainX, trainY = build_nn_input(text_train, train_vocab, params.batchsize)
print('vocab size ' .. vocab_size)

text_valid = load_data( params.valid )
check_in_vocab(text_valid, train_vocab)

text_test  = load_data( params.test )
check_in_vocab(text_test, train_vocab)

protos = build_rnn_network(vocab_size, params.embed, params.rnnsize, params.seqlen, params.batchsize, false)
param, gradParam = protos.rnn:getParameters()
clones = clone_many_times(protos.rnn, params.seqlen)

torch.manualSeed(params.seed)
timer = torch.Timer()
sgdoptimizer = {learningRate = params.lr, learningRateDecay=params.lrd}

for iter=1,params.epochs do


    print('start epoch ' .. iter)
    timer:reset()

    -- reset initial state & loss values
    local start_index = 1
    local init_state = { torch.zeros(params.batchsize, params.rnnsize) }
    local total_loss = 0
    local mbsteps = 0

    while start_index < trainX:size(1) do

        xlua.progress(start_index, trainX:size(1))
        local end_index = math.min(trainX:size(1), start_index + params.seqlen -1)
        
        if (end_index - start_index + 1) ~= params.seqlen then
            print('init h reset')
            start_index = end_index - params.seqlen - 1
            init_state = { torch.zeros(params.batchsize, params.rnnsize) }
        end
        
        function feval(x)

            if param ~= x then
                param:copy(x)
            end
            gradParam:zero()

            -- forward process
            local batchX = trainX:sub(start_index, end_index)
            local batchY = trainY:sub(start_index, end_index)

            local hidden = {}
            local predict = {}
            local losses = 0
            hidden[0] = state_table_clone(init_state)


            for seqidx=1,params.seqlen do
                local h_t, o_t = unpack(clones[seqidx]:forward({batchX:narrow(1, seqidx, 1):view(params.batchsize), unpack(hidden[seqidx-1])}))
                losses = losses + protos.criteria:forward(o_t, batchY:narrow(1, seqidx, 1):view(params.batchsize))
                hidden[seqidx] = {}    
                predict[seqidx] = {}
                table.insert(hidden[seqidx], h_t)
                table.insert(predict[seqidx], o_t)
            end

            -- backward process
            local dh = { [params.seqlen] = {torch.zeros(params.batchsize, params.rnnsize)} }
            for seqidx=params.seqlen,1,-1 do
                local output_t = unpack(predict[seqidx])
                local do_t = protos.criteria:backward(output_t, batchY:narrow(1, seqidx, 1):view(params.batchsize))
                local dinput_hidden_t = clones[seqidx]:backward({batchX:narrow(1, seqidx, 1):view(params.batchsize), unpack(hidden[seqidx-1])}, {unpack(dh[seqidx]), do_t })

                -- set the derivative to backward layer
                -- first input is x, the rest is derivative of hidden h
                dh[seqidx-1] = {}
                for hidx=2,#dinput_hidden_t do
                    table.insert(dh[seqidx-1], dinput_hidden_t[hidx])
                end
            end

            losses = losses / params.seqlen
            gradParam:div(params.seqlen)
            gradParam:clamp(-params.clip, params.clip)

            -- handle state transition
            init_state = {unpack(hidden[params.seqlen])}
            return losses, gradParam
        end

        _, l = optim.sgd(feval, param, sgdoptimizer)
        total_loss = total_loss + l[1]
        mbsteps = mbsteps + 1

        start_index = end_index + 1
    end

    collectgarbage()

    if iter % params.saveiter == 0 or iter == params.epochs then
        print('save model')
        local filename = 'rnn.' .. iter .. '.model.t7'
        torch.save(filename, protos)
    end

    local seconds = timer:time().real
    print('avg loss ' .. total_loss / mbsteps)
    print('epoch costs ' .. seconds .. ' seconds')
    print('each second train ' .. trainX:size(1)/seconds .. ' samples')
end
