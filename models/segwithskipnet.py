import torch
import torch.nn as nn

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes)//4 + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features//4, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


# class PSPUpsample(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.PReLU()
#         )

#     def forward(self, x):
#         h, w = 2 * x.size(2), 2 * x.size(3)
#         p = F.upsample(input=x, size=(h, w), mode='bilinear')
#         return self.conv(p)


class segnetDown2(nn.Module):
	def __init__(self, in_size, out_size):
		super(segnetDown2, self).__init__()
		self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		outputs = self.conv2(outputs)
		unpooled_shape = outputs.size()
		outputs, indices = self.maxpool_with_argmax(outputs)
		return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
	def __init__(self, in_size, out_size):
		super(segnetDown3, self).__init__()
		self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
		self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

	def forward(self, inputs):
		outputs = self.conv1(inputs)
		outputs = self.conv2(outputs)
		outputs = self.conv3(outputs)
		unpooled_shape = outputs.size()
		outputs, indices = self.maxpool_with_argmax(outputs)
		return outputs, indices, unpooled_shape

# Used in the last Decoder when skip connection is not needed
class segnetUp1(nn.Module):
	def __init__(self, in_size, out_size):
		super(segnetUp1, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_shape):
		outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		return outputs

class segnetUp2(nn.Module):
	def __init__(self, in_size, out_size):
		super(segnetUp2, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
		self.conv1 = conv2DBatchNormRelu(in_size*1.5, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_shape, skip_con):
		outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
		outputs = torch.cat((outputs, skip_con), dim=1)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		return outputs


class segnetUp3(nn.Module):
	def __init__(self, in_size, out_size):
		super(segnetUp3, self).__init__()
		self.unpool = nn.MaxUnpool2d(2, 2)
# 		if out_size == 256:
# 			self.conv1 = conv2DBatchNormRelu(768, in_size, 3, 1, 1)
# 		else:
# 		if out_size ==512:
# 			self.conv1 = conv2DBatchNormRelu(in_size*2, in_size, 3, 1, 1)           
# 		else:
		self.conv1 = conv2DBatchNormRelu(in_size + out_size, in_size, 3, 1, 1)
		self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
		self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

	def forward(self, inputs, indices, output_shape, skip_con):
		outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
		outputs = torch.cat((outputs, skip_con), dim=1)
		outputs = self.conv1(outputs)
		outputs = self.conv2(outputs)
		outputs = self.conv3(outputs)
		return outputs
		
class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
		

class DenseSegWithSkipNet(nn.Module):

	def __init__(self, num_classes=15, img_channels=3, sizes=(1, 2, 3, 6), psp_size=512):
		super(DenseSegWithSkipNet, self).__init__()
		# self.computing_device = computing_device 
		self.num_classes = num_classes
		self.img_channels = img_channels

		self.down1 = segnetDown2(self.img_channels, 64)
		self.down2 = segnetDown2(64, 128)
		self.down3 = segnetDown3(128, 256)
		self.down4 = segnetDown3(256, 512)
		self.down5 = segnetDown3(512, 512)

		self.psp = PSPModule(psp_size, 512, sizes)


		self.up5 = segnetUp3(512, 512)
		self.up4 = segnetUp3(512, 256)
		self.up3 = segnetUp3(256, 128)
		self.up2 = segnetUp2(128, 64)
		self.up1 = segnetUp1(64, num_classes)

	def forward(self, inputs):

		down1, indices_1, unpool_shape1 = self.down1(inputs)
		down2, indices_2, unpool_shape2 = self.down2(down1)
		down3, indices_3, unpool_shape3 = self.down3(down2)
		down4, indices_4, unpool_shape4 = self.down4(down3)
		down5, indices_5, unpool_shape5 = self.down5(down4)
		print ("down5:", down5.shape)
		p = self.psp(down5)
		print ("p:", p.shape)

		up5 = self.up5(p, indices_5, unpool_shape5, down4)
		#up5 = self.up5(down5, indices_5, unpool_shape5, down4)
		up4 = self.up4(up5, indices_4, unpool_shape4, down3)
		up3 = self.up3(up4, indices_3, unpool_shape3, down2)
		up2 = self.up2(up3, indices_2, unpool_shape2, down1)
		up1 = self.up1(up2, indices_1, unpool_shape1)

		return up1

	def init_vgg16_params(self, vgg16):
		blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

		features = list(vgg16.features.children())

		vgg_layers = []
		for _layer in features:
			if isinstance(_layer, nn.Conv2d):
				vgg_layers.append(_layer)

		merged_layers = []
		for idx, conv_block in enumerate(blocks):
			if idx < 2:
				units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
			else:
				units = [
					conv_block.conv1.cbr_unit,
					conv_block.conv2.cbr_unit,
					conv_block.conv3.cbr_unit,
				]
			for _unit in units:
				for _layer in _unit:
					if isinstance(_layer, nn.Conv2d):
						merged_layers.append(_layer)

		assert len(vgg_layers) == len(merged_layers)

		for l1, l2 in zip(vgg_layers, merged_layers):
			if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
				assert l1.weight.size() == l2.weight.size()
				assert l1.bias.size() == l2.bias.size()
				l2.weight.data = l1.weight.data
				l2.bias.data = l1.bias.data