-- @Author: xieshuqin
-- @Date:   2017-05-10 13:58:12
-- @Last Modified by:   xieshuqin
-- @Last Modified time: 2017-05-10 17:02:06

require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'

filters = {}

function blur(window, img)

	kernelsize = 2*window+1
	-- define filter
	if not filters[kernelsize] then
		sigma = 1/2.58/kernelsize*window

		gaussian = image.gaussian(kernelsize, sigma)
		gaussian:div(torch.sum(gaussian))

		kernel = torch.Tensor(3, 3, kernelsize, kernelsize):zero()
		kernel[1][1]:copy(gaussian)
		kernel[2][2]:copy(gaussian)
		kernel[3][3]:copy(gaussian)

		local kW, pW = kernelsize, (kernelsize-1)/2

		filters[kernelsize] = nn.SpatialConvolution(3, 3, kW, kW, 1, 1, pW, pW)

		-- -- initialize weights and bias
		filters[kernelsize].weight:copy(kernel)
		filters[kernelsize].bias:zero()
		-- convert to cuda
		filters[kernelsize] = filters[kernelsize]:cuda()

	end

	-- blur the image
	img_out = filters[kernelsize]:forward(img:cuda()):float()

	return img_out

end
