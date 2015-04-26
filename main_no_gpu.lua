--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----

require('nngraph')
require('base')
local stringx = require('pl.stringx')
ptb = require('data')


-- Train 1 day and gives 82 perplexity.
--[[
local params = {batch_size=20,
                seq_length=35,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=10000,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
               ]]--

-- Trains 1h and gives test 115 perplexity.
-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=2,
--                 rnn_size=200,
--                 dropout=0,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=4,
--                 max_max_epoch=13,
--                 max_grad_norm=5}

function transfer_data(x)
  if params.use_gpu then
    return x:cuda()
  else
    return x
  end
end

--local state_train, state_valid, state_test
model = {}
--local paramx, paramdx

function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                              params.rnn_size)(x)}
  local next_s           = {}
  local split            = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), pred})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = {}
  if params.load_model then
    print("Loading")
    core_network = torch.load(params.model_load_fname).core_network
  else
    print("Creating")
    core_network = create_network()
  end
  print("Network loaded or created")
  paramx, paramdx = core_network:getParameters() -- why not local?
  model.s = {} -- model is global
  model.ds = {}
  model.start_s = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size,
                                                params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size,
                                                 params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size,
                                            params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
end

-- function load_model()
--   local l_model = torch.load(params.load_model_file)
--   model.core_network = l_model.core_network
-- end

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    -- print(i)
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    -- print(x, y, s)
    -- Why does forward return both output (model.s) and error?
    model.err[i], model.s[i], pred1 = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean() -- perplexity
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local dpred = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds, dpred})[3]
    g_replace_table(model.ds, tmp)
    if params.use_gpu then
      cutorch.synchronize()
    end
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1], pred1 = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end

function run_my_test()
  reset_state(state1)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state1.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  predictions = torch.zeros(len)
  for i = 1, (len - 1) do
    local x = state1.data[i]
    local y = state1.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1], pred1 = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    -- predictions[i+1] = torch.multinomial(pred1[{ 1,{} }], 1)
    -- print("model.s[1][4] = ", model.s[1][4])
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
    pred2 = pred1[{ 1,{} }]
    pred2:div(pred2:sum())
    predictions[i+1] = torch.multinomial(pred2, 1)
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
  return predictions
end

function main()
  if params.use_gpu then
    g_init_gpu(arg)
  end
  -- Transfer data to GPU
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid = {data=transfer_data(ptb.validdataset(params.batch_size))}
  state_test  = {data=transfer_data(ptb.testdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  setup() -- Set up model as global lua frame
  step = 0
  epoch = 0
  total_cases = 0
  beginning_time = torch.tic()
  start_time = torch.tic()
  print("Starting training.")
  words_per_step = params.seq_length * params.batch_size
  epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  --perps
  while epoch < params.max_max_epoch do
    perp = fp(state_train)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + params.seq_length * params.batch_size
    epoch = step / epoch_size
    if step % torch.round(epoch_size / 10) == 10 then
      wps = torch.floor(total_cases / torch.toc(start_time))
      since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(params.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end
    if step % epoch_size == 0 then
      run_valid()
      print("Saving model...")
      torch.save(params.model_save_fname, model)
      if epoch > params.max_epoch then
        params.lr = params.lr / params.decay
      end
    end
    if step % 33 == 0 then
      if params.use_gpu then
        cutorch.synchronize()
      end
      collectgarbage()
    end
  end
  
  print("Saving model...")
  torch.save('final_model.lstm', model)

  run_test()
  print("Training is over.")
end -- main


--------------------------------------------------------------------------------------
-- Command line options
--------------------------------------------------------------------------------------
if not params then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-batch_size', 20, 'Training batch size')
   cmd:option('-seq_length', 20, 'Sequence length')
   cmd:option('-layers', 2, 'Number of layers in network')
   cmd:option('-decay', 2, 'Decay')
   cmd:option('-rnn_size', 200, 'RNN size')
   cmd:option('-dropout', 0, 'LSTM dropout')
   cmd:option('-init_weight', 0.1, 'TODO')
   cmd:option('-lr', 1, 'Learning rate')
   cmd:option('-vocab_size', 10000, 'Vocabulary size')
   cmd:option('-max_epoch', 4, 'Maximum number of training epochs')
   cmd:option('-max_max_epoch', 13, 'TODO')
   cmd:option('-max_grad_norm', 5, 'Gradient normalization parameter')
   cmd:option('-use_gpu', false, 'Whether to run on GPU')
   cmd:option('-model_save_fname', 'model.lstm', 'Save model as file name')
   cmd:option('-model_load_fname', 'model.lstm', 'Model file to load')
   cmd:option('-load_model', false, 'Whether to load model')
   -- cmd:option('-pooling', 'max', '[max | logexp] pooling')
   -- cmd:option('-beta', 20, 'LogExp pooling beta parameter')
   -- cmd:option('-inputDim', 50, 'word vector dimension: [50 | 100 | 200 | 300]')
   -- cmd:option('-glovePath', '/scratch/courses/DSGA1008/A3/glove/', 'path to GloVe files')
   -- cmd:option('-dataPath', '/scratch/courses/DSGA1008/A3/data/train.t7b', 'path to data')
   -- cmd:option('-idfPath', '../idf/idf.csv', 'path to idf.csv file')
   -- cmd:option('-nTrainDocs', 10000, 'number of training documents in each class')
   -- cmd:option('-nTestDocs', 1000, 'number of test documents in each class')
   -- cmd:option('-nClasses', 5, 'number of classes')
   -- cmd:option('-nEpochs', 50, 'number of training epochs')
   -- cmd:option('-minibatchSize', 128, 'minibatch size')
   -- cmd:option('-learningRate', 0.1, 'learning rate')
   -- cmd:option('-learningRateDecay', 0.001, 'learning rate decay')
   -- cmd:option('-momentum', 0.1, 'SGD momentum')
   -- cmd:option('-model', 'linear_baseline', 'model function to be used [linear_baseline | linear_two_hidden | conv_baseline | conv_concat]')
   -- cmd:option('-seed', 0, 'manual seed for initial data permutation')
   -- cmd:option('-modelFileName' , 'model.net', 'filename to save model')
   -- cmd:option('-wordWeight', 'none', 'word vector weights ["none" | "tfidf"]')
   -- cmd:option('-normalize', 0, 'normalize bag of words [true | false]')
   -- cmd:text()
   params = cmd:parse(arg or {})
   -- opt.glovePath = opt.glovePath .. 'glove.6B.' .. opt.inputDim .. 'd.txt'
   -- opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
end

if params.use_gpu then
  print("Loading GPU dependencies")
  ok,cunn = pcall(require, 'fbcunn')
  if not ok then
      ok,cunn = pcall(require,'cunn')
      if ok then
          print("warning: fbcunn not found. Falling back to cunn") 
          LookupTable = nn.LookupTable
      else
          print("Could not find cunn or fbcunn. Either is required")
          os.exit()
      end
  else
      deviceParams = cutorch.getDeviceProperties(1)
      cudaComputeCapability = deviceParams.major + deviceParams.minor/10
      LookupTable = nn.LookupTable
  end
else
  LookupTable = nn.LookupTable
end

-- params.batch_size = 2
-- state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}

-- local params = {batch_size=20,
--                 seq_length=20,
--                 layers=2,
--                 decay=2,
--                 rnn_size=200,
--                 dropout=0,
--                 init_weight=0.1,
--                 lr=1,
--                 vocab_size=10000,
--                 max_epoch=4,
--                 max_max_epoch=13,
--                 max_grad_norm=5}
if params.load_model then
  -- Playing with sequences
  print("Command Line Parameters")
  for key, val in pairs(params) do
    print(key, val)
  end

  ptb.traindataset(params.batch_size)

  setup()
  test_str = "the president issued an executive order"
  data = stringx.replace(test_str, '\n', '<eos>')
  data = stringx.split(data)
  x = torch.zeros(#data)
  for i = 1,#data do
    x[i] = ptb.vocab_map[data[i]]
    x = x:resize(x:size(1), 1):expand(x:size(1), params.batch_size)
    print(data[i],x[{i,1}])
  end

  -- ptb.testdataset(params.batch_size)

  state1 = {}
  state1.pos = 1
  state1.data = x

  predictions = run_my_test()
else
  main()
end



