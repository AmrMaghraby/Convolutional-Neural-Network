function convolution_nn=initconvolution_nn(convolution_nn, SizeOfImage)

convolution_nn.input_image_height = SizeOfImage(1);
convolution_nn.input_image_width = SizeOfImage(2);
convolution_nn.no_of_input_channels=1;
if numel(SizeOfImage) == 3
  convolution_nn.no_of_input_channels=SizeOfImage(3);
end
convolution_nn.no_of_layers=1;
convolution_nn.layers{1} =struct('type', 'i', 'NoOfFeatureMaps', convolution_nn.no_of_input_channels);
convolution_nn.layers{1}.type = 'i'; 
convolution_nn.layers{1}.NoOfFeatureMaps = convolution_nn.no_of_input_channels;
convolution_nn.layers{1}.featuremap_width = convolution_nn.input_image_width;
convolution_nn.layers{1}.featuremap_height =convolution_nn.input_image_height ;
convolution_nn.layers{1}.prev_layer_NoOfFeatureMaps = 0;
convolution_nn.loss_func='auto'; 
convolution_nn.regularization_const = 0;
convolution_nn.learning_rate = 0.01;