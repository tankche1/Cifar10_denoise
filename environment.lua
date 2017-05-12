-- @Author: xieshuqin
-- @Date:   2017-05-10 01:17:59
-- @Last Modified by:   xieshuqin
-- @Last Modified time: 2017-05-10 01:18:03
if opt.blur  then
   opt.actions = {'1','2','3','4','5','6','7','8'}
   execute_fn = 
	function (input,action)
    if opt.GPU > 0 then
      input = input:float()
    end 
		input_processed= input:clone()
		for i=1,opt.batchSize do
			--image.save('jpg/img_blur'..i..'.jpg', input[i])
			execute_action = opt.actions[action[i]]
			--execute action
			deblur_parameters[i].kernel_size=tonumber(execute_action)
			file = io.open('jpg/deblur_parameters'..i..'.txt','w')
			file:write(deblur_parameters[i].kernel_size)
			file:close()
		end
		os.execute('python deblur.py')
		for i=1,opt.batchSize do
			input_processed[i]=image.load('jpg/img_deblur'..i..'.jpg')
		end
    if opt.GPU > 0 then
      input_processed = input_processed:cuda()
    end
		return input_processed
	end 
-- tolerance=0.2, tau=0.125, tv_weight=100 +tolerance=0.03 +tau=0.01 +tv_weight=7
elseif opt.noise then
   opt.actions = {'+tolerance','-tolerance','+tau','-tau','+tv_weight','-tv_weight'}
   execute_fn = 
   	function (input,action) 
		--local x1 = sys.clock()

    if opt.GPU > 0 then
      input = input:float()
    end

		input_processed= input:clone()
        --print(input:size())
		for i=1,opt.batchSize do
			image.save('jpg/img'..i..'.jpg', input[i])

        	execute_action = opt.actions[action[i]]
		
			--execute action
			if execute_action=='+tolerance' then
				denoise_paremeters[i].tolerance=denoise_paremeters[i].tolerance+0.03 
			elseif execute_action=='-tolerance' then
				denoise_paremeters[i].tolerance=denoise_paremeters[i].tolerance-0.03
			elseif execute_action=='+tau' then
				denoise_paremeters[i].tau=denoise_paremeters[i].tau+0.01
			elseif execute_action=='-tau' then
				denoise_paremeters[i].tau=denoise_paremeters[i].tau-0.01
			elseif execute_action=='+tv_weight' then
				denoise_paremeters[i].tv_weight=denoise_paremeters[i].tv_weight+7
			elseif execute_action=='+tv_weight' then
				denoise_paremeters[i].tv_weight=denoise_paremeters[i].tv_weight-7
			end
			--print(denoise_paremeters[i].tolerance..","..denoise_paremeters[i].tau..","..denoise_paremeters[i].tv_weight..",")
			
			-- deliver paremeters to python
			file = io.open('jpg/denoise_paremeters'..i..'.txt','w')
			file:write(denoise_paremeters[i].tolerance..","..denoise_paremeters[i].tau..","..denoise_paremeters[i].tv_weight..",")
			file:close()
			
		end
		os.execute('python denoise.py')
			
		for i=1,opt.batchSize do
			input_processed[i]=image.load('jpg/img_denoise'..i..'.jpg')
		end
		--x1 = sys.clock()-x1
		--print(string.format("elapsed time: %.8f\n", x1))
    if opt.GPU > 0 then
      input_processed = input_processed:cuda()
    end
		return input_processed
	end

end
