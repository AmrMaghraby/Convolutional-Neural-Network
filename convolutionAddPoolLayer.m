function convolution_nn=convolutionAddPoolLayer(convolution_nn, SubsampleRate, SubsampleMethod)

convolution_nn.no_of_layers= convolution_nn.no_of_layers +1;
l=convolution_nn.no_of_layers;
convolution_nn.layers{l}.type = 'p';
convolution_nn.layers{l}.subsample_rate=SubsampleRate;
convolution_nn.layers{l}.subsample_method=SubsampleMethod;
convolution_nn.layers{l}.NoOfFeatureMaps = convolution_nn.layers{l-1}.NoOfFeatureMaps;
convolution_nn.layers{l}.featuremap_width = convolution_nn.layers{l-1}.featuremap_width/SubsampleRate;
convolution_nn.layers{l}.featuremap_height = convolution_nn.layers{l-1}.featuremap_height/SubsampleRate;

convolution_nn.layers{l}.act_func='none';