require 'torch'
require 'nn'

require 'VanillaRNN'
require 'LSTM'

require 'hdf5'

local utils = require 'util.utils'
local utf8 = require 'lua-utf8'
local cjson = require 'cjson'

local HM, parent = torch.class('nn.HarmModel', 'nn.Module')


function HM:__init(kwargs)
  self.idx_to_token = utils.get_kwarg(kwargs, 'idx_to_token')
  self.token_to_idx = {}
  self.vocab_size = 0
  for idx, token in pairs(self.idx_to_token) do
    self.token_to_idx[token] = idx
    self.vocab_size = self.vocab_size + 1
  end

  self.model_type = utils.get_kwarg(kwargs, 'model_type')
  self.wordvec_dim = utils.get_kwarg(kwargs, 'wordvec_size')
  self.rnn_size = utils.get_kwarg(kwargs, 'rnn_size')
  self.num_layers = utils.get_kwarg(kwargs, 'num_layers')
  self.dropout = utils.get_kwarg(kwargs, 'dropout')
  self.batchnorm = utils.get_kwarg(kwargs, 'batchnorm')

  local V, D, H = self.vocab_size, self.wordvec_dim, self.rnn_size

  self.net = nn.Sequential()
  self.rnns = {}
  self.bn_view_in = {}
  self.bn_view_out = {}

  self.net:add(nn.LookupTable(V, D))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    if self.model_type == 'rnn' then
      rnn = nn.VanillaRNN(prev_dim, H)
    elseif self.model_type == 'lstm' then
      rnn = nn.LSTM(prev_dim, H)
    end
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    if self.batchnorm == 1 then
      local view_in = nn.View(1, 1, -1):setNumInputDims(3)
      table.insert(self.bn_view_in, view_in)
      self.net:add(view_in)
      self.net:add(nn.BatchNormalization(H))
      local view_out = nn.View(1, -1):setNumInputDims(2)
      table.insert(self.bn_view_out, view_out)
      self.net:add(view_out)
    end
    if self.dropout > 0 then
      self.net:add(nn.Dropout(self.dropout))
    end
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, V))
  self.net:add(self.view2)
end


function HM:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)

  for _, view_in in ipairs(self.bn_view_in) do
    view_in:resetSize(N * T, -1)
  end
  for _, view_out in ipairs(self.bn_view_out) do
    view_out:resetSize(N, T, -1)
  end

  return self.net:forward(input)
end


function HM:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function HM:parameters()
  return self.net:parameters()
end


function HM:training()
  self.net:training()
  parent.training(self)
end


function HM:evaluate()
  self.net:evaluate()
  parent.evaluate(self)
end


function HM:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


function HM:encode_string(s)
  local encoded = torch.LongTensor(utf8.len(s))
  for i = 1, utf8.len(s) do
    local token = utf8.sub(s, i, i)
    local idx = self.token_to_idx[token]
    assert(idx ~= nil, 'Got invalid token ' .. token)
    encoded[i] = idx
  end
  return encoded
end


function HM:decode_string(encoded)
  assert(torch.isTensor(encoded) and encoded:dim() == 1)
  local s = ''
  for i = 1, encoded:size(1) do
    local idx = encoded[i]
    local token = self.idx_to_token[idx]
    s = s .. token
  end
  return s
end


--[[
Sample from the language model. Note that this will reset the states of the
underlying RNNs. Stops when `stop_char` is encountered.

Inputs:
- init: String of length T0
- max_length: Number of characters to sample

Returns:
- sampled: (1, max_length) array of integers, where the first part is init.
--]]
function HM:sample(kwargs)
  local u = utf8.escape
  local stop_char = self:encode_string(u'%1115'):view(1, -1)[1] -- see scripts/constants.py

  local T = utils.get_kwarg(kwargs, 'length', 100)
  local start_text = utils.get_kwarg(kwargs, 'start_text', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local sample = utils.get_kwarg(kwargs, 'sample', 0)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)

  local sampled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, first_t
  if #start_text > 0 then
    if verbose > 0 then
      print('Seeding with: "' .. start_text .. '"')
    end
    local x = self:encode_string(start_text):view(1, -1)
    local T0 = x:size(2)
    sampled[{{}, {1, T0}}]:copy(x)
    scores = self:forward(x)[{{}, {T0, T0}}]
    first_t = T0 + 1
  else
    if verbose > 0 then
      print('Seeding with uniform probabilities')
    end
    local w = self.net:get(1).weight
    scores = w.new(1, 1, self.vocab_size):fill(1)
    first_t = 1
  end

  local _, next_char = nil, nil
  for t = first_t, T do
    if sample == 0 then
      _, next_char = scores:max(3)
      next_char = next_char[{{}, {}, 1}]
    else
       local probs = torch.div(scores, temperature):double():exp():squeeze()
       probs:div(torch.sum(probs))
       next_char = torch.multinomial(probs, 1):view(1, 1)
    end
    sampled[{{}, {t, t}}]:copy(next_char)
    -- if next_char == stop_char then
    --   break
    -- end
    scores = self:forward(next_char)
  end

  self:resetStates()
  return self:decode_string(sampled[1])
end


--[[
Use the language model to fill in missing symbols given the other symbols.
Note that this will reset the states of the underlying RNNs.

Inputs:
- input: input string with missing tokens denoted by `blank_mask`
- blank_mask: unique UTF8 character denoting missing symbols to be filled in by model

Returns:
- filled: (1, len(input)) array of integers, where the `blank_mask` tokens have been predicted
          by the language model.
--]]
function HM:harmonize(kwargs)
  local u = utf8.escape
  local f = io.open(utils.get_kwarg(kwargs, 'input', ''), 'rb')
  local input = f:read('*all')
  f:close()
  local blank_mask = utils.get_kwarg(kwargs, 'blank_mask', u"%1130")
  local utf_to_txt = cjson.decode(io.open(utils.get_kwarg(kwargs, 'utf_to_txt', '../../scratch/utf_to_txt.json')):read())
  local sample = utils.get_kwarg(kwargs, 'sample', 0)
  local temperature = utils.get_kwarg(kwargs, 'temperature', 1)
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)

  local T = utf8.len(input)
  local txt_to_utf = reverseTable(utf_to_txt)

  local filled = torch.LongTensor(1, T)
  self:resetStates()

  local scores, note_scores
  if verbose > 0 then
    print('Harmonizing input text: "' .. input .. '"')
  end

  -- token indices to ignore when sampling for masked out tokens
  local ignore_tokens = { 'START', 'END', '(.)', '|||' }
  local ignore_idxs = torch.zeros(#ignore_tokens):long()
  for i = 1, #ignore_tokens do
    ignore_idxs[i] = self:encode_string(txt_to_utf[ignore_tokens[i]])[1]
  end

  local _, next_char, next_idx = nil, nil, nil

  -- copy first character (should be START) to initialize scores
  next_char = utf8.sub(input, 1, 1)
  next_idx = self:encode_string(next_char):view(1, -1)
  filled[{{}, {1, 1}}]:copy(next_idx)
  scores = self:forward(next_idx)

  -- process/infer remaining blank slots
  for t = 2, T do
    next_char = utf8.sub(input, t, t)
    if next_char == blank_mask then
      note_scores = scores:clone()
      note_scores:indexFill(3, ignore_idxs, 0) -- zero out scores for special symbols

      if sample == 0 then
        _, next_idx = note_scores:max(3)
        next_idx = next_idx[{{}, {}, 1}]
      else
        local probs = torch.div(scores, temperature):double():exp():squeeze()
        probs:div(torch.sum(probs))
        next_idx = torch.multinomial(probs, 1):view(1, 1)
      end
    else
      next_idx = self:encode_string(next_char):view(1, 1)
    end

    filled[{{}, {t, t}}]:copy(next_idx)
    scores = self:forward(next_idx)
  end

  self:resetStates()
  return self:decode_string(filled[1])
end

-- reverses a table
function reverseTable(t)
  local reversedTable = {}
  for k, v in pairs(t) do
    reversedTable[v] = k
  end
  return reversedTable
end

function HM:clearState()
  self.net:clearState()
end

--[[
Embeds a variable-length string by consuming the string and returning the memory cell.

Inputs:
- init: String of length T0

Returns:
- embedding: (1, #init) array of floats representing the memory cell.
--]]
function HM:embed_note(kwargs)
  local embed_utf_file = utils.get_kwarg(kwargs, 'embed_utf_file', '')
  local verbose = utils.get_kwarg(kwargs, 'verbose', 0)
  local out_dir = utils.get_kwarg(kwargs, 'out_dir', '~/data')

  self:resetStates()

  local embed_utf = assert(io.open(embed_utf_file, "r")):read()

  if verbose > 0 then
    print('Seeding with: "' .. embed_utf .. '"')
  end
  local x = self:encode_string(embed_utf):view(1, -1)
  local T0 = x:size(2)
  _ = self:forward(x)[{{}, {T0, T0}}]

  print('Writing input to ' .. out_dir .. '/input.h5')
  local myFile = hdf5.open(out_dir .. '/input.h5', 'w')
  myFile:write(out_dir .. '/input.h5', x)
  myFile:close()

  for i=1,#self.net.modules do
    local layer = self.net.modules[i]
    local out_path = out_dir .. '/outputs-' .. i .. '.h5'
    print('Writing layer outputs ' .. out_path .. ' with size: ' .. tostring(layer.output:double():size()))
    myFile = hdf5.open(out_path, 'w')
    myFile:write(out_path, layer.output:double():squeeze())
    myFile:close()
    if layer.cell ~= null then
      out_path = out_dir .. '/cell-' .. i .. '.h5'
      print('Writing cell ' .. out_path .. ' with size: ' .. tostring(layer.cell:double():size()))
      myFile = hdf5.open(out_path, 'w')
      myFile:write(out_path, layer.cell:double():squeeze())
      myFile:close()
    end
  end
  print(self.net.modules)

  self:resetStates()
  return self
end

