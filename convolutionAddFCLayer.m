function convolution_nn=convolutionAddFCLayer(convolution_nn,NoOfNodes, ActivationFuncName)

convolution_nn.no_of_layers= convolution_nn.no_of_layers +1;
l=convolution_nn.no_of_layers;
convolution_nn.layers{l}.type = 'f';
convolution_nn.layers{l}.NoOfNodes=NoOfNodes;
convolution_nn.layers{l}.act_func=ActivationFuncName;
PreviousLayerFeatureMapWidth=convolution_nn.input_image_width;
PreviousLayerFeatureMapHeight=convolution_nn.input_image_height;
PreviousLayerNoOfFeatureMaps = convolution_nn.no_of_input_channels;
convolution_nn.layers{l}.no_of_inputs = PreviousLayerNoOfFeatureMaps * PreviousLayerFeatureMapHeight *PreviousLayerFeatureMapWidth;
convolution_nn.layers{l}.convert_input_to_1D=1;
if l>1 & convolution_nn.layers{l-1}.type ~= 'f'
    PreviousLayerNoOfFeatureMaps = convolution_nn.layers{l-1}.NoOfFeatureMaps;
    PreviousLayerFeatureMapWidth = convolution_nn.layers{l-1}.featuremap_width;
    PreviousLayerFeatureMapHeight = convolution_nn.layers{l-1}.featuremap_height;
    convolution_nn.layers{l}.no_of_inputs = PreviousLayerNoOfFeatureMaps * PreviousLayerFeatureMapHeight *PreviousLayerFeatureMapWidth;
    convolution_nn.layers{l}.convert_input_to_1D=1;
elseif l>1 & convolution_nn.layers{l-1}.type == 'f'
    convolution_nn.layers{l}.no_of_inputs = convolution_nn.layers{l-1}.NoOfNodes;
    convolution_nn.layers{l}.convert_input_to_1D=0; 
end
convolution_nn.layers{l}.W =0.5*rand([NoOfNodes convolution_nn.layers{l}.no_of_inputs]) -0.25;
convolution_nn.layers{l}.b = 0.5*rand([NoOfNodes 1]) - 0.25;
    
