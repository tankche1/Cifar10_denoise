-- command setting
local nStep = opt.nStep
local hiddenSize = 512
actions = opt.actions
classes = 10

require 'dp'
require 'dpnn'
require 'rnn'
dofile('./models/ReinforceSample.lua')
dofile('./models/ReccurentPreProcessor.lua')

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64):add(nn.Dropout(0.3))
ConvBNReLU(64,64)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(64,128):add(nn.Dropout(0.4))
ConvBNReLU(128,128)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(128,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256):add(nn.Dropout(0.4))
ConvBNReLU(256,256)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(256,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512):add(nn.Dropout(0.4))
ConvBNReLU(512,512)
vgg:add(MaxPooling(2,2,2,2):ceil())
vgg:add(nn.View(512))

local rnn_input = vgg

local rnn_feedback = nn.Linear(hiddenSize, hiddenSize)

local rnn = nn.Recurrent(hiddenSize, rnn_input, rnn_feedback,nn.ReLU(), 99999)

local actor = nn.Sequential()
		:add(nn.Linear(hiddenSize, #actions))
		:add(nn.SoftMax())
		:add(nn.ReinforceSample())--A Reinforce subclass that implements the REINFORCE algorithm (ref. A) for a Categorical (i.e. Multinomial with one sample) probability distribution.  output is one hot. same shape as input
		:add(nn.ArgMax(2))

local preprocessor = nn.RecurrentPreProcessor(rnn, actor, nStep, {hiddenSize}, execute_fn)

local classifier = nn.Sequential()
		:add(nn.Linear(hiddenSize, classes))
		:add(nn.LogSoftMax())

local agent = nn.Sequential()
		:add(preprocessor)
		:add(nn.SelectTable(-1))-- change to Table
		:add(classifier)

-- baseline, which approximate the expected reward
local baseline = nn.Sequential()
		:add(nn.Constant(1,1))
		:add(nn.Add(1))

local concat = nn.ConcatTable():add(nn.Identity()):add(baseline)
local concat2 = nn.ConcatTable():add(nn.Identity()):add(concat)

agent:add(concat2)

return agent
