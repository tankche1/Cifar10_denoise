require 'xlua'
require 'optim'
require 'nn'
dofile './provider.lua'
dofile './util.lua'
local c = require 'trepl.colorize'

--[[
opt = lapp[[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 128)          batch size
   -r,--learningRate          (default 0.01)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default vgg_bn_drop)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   --type                     (default cuda)          cuda/float/cl
   --nStep                    (default 6)           nStep
   --addnoise                                       add noise
   --addblur                                        add blur
   --blur                                           de noise
   --noise                                          de blur
   --GPU                      (default 0)           GPU
   --noiseScale            (default 40.00)          noiseScale
]]
--]]

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Training a simple network')
cmd:text()
cmd:text('Options')


cmd:option('-save','logs','subdirectory to save logs')
cmd:option('-batchSize',128,'batch size')
cmd:option('-learningRate',0.01,'learning rate')
cmd:option('-learningRateDecay',1e-7,' learning rate decay')
cmd:option('-weightDecayy',0.0005,'weightDecay')
cmd:option('-momentum',0.9,'momentum')
cmd:option('-epoch_step',25,'epoch step')
cmd:option('-model','vgg_bn_drop','model name')
cmd:option('-max_epoch',300,'maximum number of iterations')
cmd:option('-backend','nn','backend')
cmd:option('-type','cuda','cuda/float/cl')
cmd:option('-nStep',6,'nStep')
cmd:option('-addnoise',false,'add noise')
cmd:option('-addblur',false,'add blur')
cmd:option('-blur',false,'de bur')
cmd:option('-noise',false,'de noise')
cmd:option('-GPU',0,'GPU')
cmd:option('-noiseScale',40.01,'noiseScale')
cmd:option('-load',false,'load model')

cmd:text()
cmd:text()
cmd:text()
cmd:text()

-- parse input params
opt = cmd:parse(arg)

--opt.noiseScale=opt.noiseScale:float()
print(opt)

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

if opt.noise or opt.blur or opt.model == 'agent' then
  print(c.blue '==>' ..' constructing simulating environment')
  dofile('./environment.lua')
  print(opt.actions)
end

print(c.blue '==>' ..' configuring model')
if opt.load then
    print(c.blue '==>' ..' loading model')
    model=torch.load('logs/model.net')
else 
    print('haha!')
    model = nn.Sequential()
    model:add(nn.BatchFlip():float())
    model:add(cast(nn.Copy('torch.FloatTensor', torch.type(cast(torch.Tensor())))))
    model:add(cast(dofile('models/'..opt.model..'.lua')))
    model:get(2).updateGradInput = function(input) return end

    if opt.backend == 'cudnn' then
       require 'cudnn'
       cudnn.benchmark=true
       cudnn.convert(model:get(3), cudnn)
    end
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load 'provider.t7'
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()

confusion = optim.ConfusionMatrix(10)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
if opt.model == 'agent' then
   criterion = cast(nn.ParallelCriterion(true)
      :add(nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert())) -- BACKPROP
      :add(nn.ModuleCriterion(nn.VRClassReward(model, opt.rewardScale), nil, nn.Convert()))) -- REINFORCE
else
   criterion = cast(nn.ClassNLLCriterion())
end

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}
geometry={32,32}
denoise_paremeters={}
opt.noiseScale=opt.noiseScale/255.0

epoch=1

function train()
  model:training()
  epoch = epoch or 1

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  
  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local targets = cast(torch.FloatTensor(opt.batchSize))
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    xlua.progress(t, #indices)
        
    --if t>10  then break  end

    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))
        
    inputs=inputs/255.0;
        
    for i=1,opt.batchSize do
			image.save('jpg/imgorigin'..i..'.jpg', inputs[i])
    end

    -- pre-processing data
    if opt.addblur then 
      print('Addblur!')
      inputs = blur(3, inputs)
    elseif opt.addnoise then
      --print('Please Complete This Part')
      
      inputs=inputs+torch.randn(opt.batchSize,3,geometry[1],geometry[2]):float()*opt.noiseScale
    end
        
    -- reset denoise perameters
	  if opt.noise  then
			--print('Initialize denoise paremeters!!!')
			for i=1,opt.batchSize do
				denoise_paremeters[i]={tolerance=0.2, tau=0.125, tv_weight=100}
		 	end
	  end
        
    if opt.model== 'vgg_bn_drop'  and opt.noise then
        for i=1,opt.batchSize do
			image.save('jpg/img'..i..'.jpg', inputs[i])
			
			-- deliver paremeters to python
			file = io.open('jpg/denoise_paremeters'..i..'.txt','w')
			file:write(denoise_paremeters[i].tolerance..","..denoise_paremeters[i].tau..","..denoise_paremeters[i].tv_weight..",")
			file:close()
			
		end
		os.execute('python denoise.py')
			
		for i=1,opt.batchSize do
			inputs[i]=image.load('jpg/img_denoise'..i..'.jpg')
		end
    end

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      if opt.model == 'agent' then confusion:batchAdd(outputs[1], targets) 
      else confusion:batchAdd(outputs, targets) end

      return f,gradParameters
    end
    optim.sgd(feval, parameters, optimState)
  end

  confusion:updateValids()
  print(('Train accuracy: '..c.cyan'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))

  train_acc = confusion.totalValid * 100

  confusion:zero()
  epoch = epoch + 1
end


function test()
  -- disable flips, dropouts and batch normalization
  model:evaluate()
  print(c.blue '==>'.." testing")
  local bs = 128
  for i=1,provider.testData.data:size(1),bs do
    xlua.progress(i,provider.testData.data:size(1))
        if i+bs-1>provider.testData.data:size(1) then
            break
        end
    local inputs = provider.testData.data:narrow(1,i,bs)
    inputs=inputs/255.0

    -- pre-processing data
    if opt.addblur then 
    	inputs = blur(3, inputs)
    elseif opt.addnoise then
        inputs=inputs+torch.randn(opt.batchSize,3,geometry[1],geometry[2]):float()*opt.noiseScale
    end
        
    -- reset denoise perameters
	if opt.noise then
	    --print('Initialize denoise paremeters!!!')
	    for i=1,opt.batchSize do
		    denoise_paremeters[i]={tolerance=0.2, tau=0.125, tv_weight=100}
		 end
	end
        
    if opt.model== 'vgg_bn_drop'  and opt.noise then
        for i=1,opt.batchSize do
			image.save('jpg/img'..i..'.jpg', inputs[i])
			
			-- deliver paremeters to python
			file = io.open('jpg/denoise_paremeters'..i..'.txt','w')
			file:write(denoise_paremeters[i].tolerance..","..denoise_paremeters[i].tau..","..denoise_paremeters[i].tv_weight..",")
			file:close()
			
		end
		os.execute('python denoise.py')
			
		for i=1,opt.batchSize do
			inputs[i]=image.load('jpg/img_denoise'..i..'.jpg')
		end
    end

    local outputs = model:forward(inputs)
    if opt.model == 'agent' then confusion:batchAdd(outputs[1], provider.testData.labels:narrow(1,i,bs))
    else confusion:batchAdd(outputs, provider.testData.labels:narrow(1,i,bs)) end
  end

  confusion:updateValids()
  print('Test accuracy:', confusion.totalValid * 100)
  
  if testLogger then
    paths.mkdir(opt.save)
    testLogger:add{train_acc, confusion.totalValid * 100}
    testLogger:style{'-','-'}
    testLogger:plot()

    if paths.filep(opt.save..'/test.log.eps') then
      local base64im
      do
        os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
        os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
        local f = io.open(opt.save..'/test.base64')
        if f then base64im = f:read'*all' end
      end

      local file = io.open(opt.save..'/report.html','w')
      file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <title>%s - %s</title>
      <img src="data:image/png;base64,%s">
      <h4>optimState:</h4>
      <table>
      ]]):format(opt.save,epoch,base64im))
      for k,v in pairs(optimState) do
        if torch.type(v) == 'number' then
          file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
        end
      end
      file:write'</table><pre>\n'
      file:write(tostring(confusion)..'\n')
      file:write(tostring(model)..'\n')
      file:write'</pre></body></html>'
      file:close()
    end
  end

  -- save model every 2 epochs
  if epoch % 2 == 0 then
    local filename = paths.concat(opt.save, 'model.net')
    print('==> saving model to '..filename)
    torch.save(filename,model)
  end

  confusion:zero()
end


for i=1,opt.max_epoch do
  
  --train()
  test()
  
end


