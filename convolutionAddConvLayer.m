function convolution_nn=convolutionAddConvLayer(convolution_nn,NoOfFeatureMaps,SizeOfKernels,ActivationFuncName)

convolution_nn.no_of_layers= convolution_nn.no_of_layers +1;
l=convolution_nn.no_of_layers;
convolution_nn.layers{l}.type = 'c';
convolution_nn.layers{l}.NoOfFeatureMaps = NoOfFeatureMaps;
convolution_nn.layers{l}.KernelWidth= SizeOfKernels(1);
convolution_nn.layers{l}.KernelHeight= SizeOfKernels(1);
if numel(SizeOfKernels)==2
    convolution_nn.layers{l}.KernelWidth= SizeOfKernels(2);
end
PreviousLayerFeatureMapWidth=convolution_nn.input_image_width;
PreviousLayerFeatureMapHeight=convolution_nn.input_image_height;
PreviousLayerNoOfFeatureMaps = convolution_nn.no_of_input_channels;
if l>1
    PreviousLayerNoOfFeatureMaps = convolution_nn.layers{l-1}.NoOfFeatureMaps;
    PreviousLayerFeatureMapWidth = convolution_nn.layers{l-1}.featuremap_width;
    PreviousLayerFeatureMapHeight = convolution_nn.layers{l-1}.featuremap_height;
end

convolution_nn.layers{l}.featuremap_width = PreviousLayerFeatureMapWidth - convolution_nn.layers{l}.KernelWidth +1;
convolution_nn.layers{l}.featuremap_height = PreviousLayerFeatureMapHeight - convolution_nn.layers{l}.KernelHeight +1;
convolution_nn.layers{l}.PreviousLayerNoOfFeatureMaps = PreviousLayerNoOfFeatureMaps;
k=0;
for i=1: NoOfFeatureMaps
    for j=1: PreviousLayerNoOfFeatureMaps
         k = k+1;
        convolution_nn.layers{l}.K(:,:,k)= 0.5*rand(convolution_nn.layers{l}.KernelHeight,convolution_nn.layers{l}.KernelWidth)-0.25;
    end
end
for j=1:NoOfFeatureMaps
     convolution_nn.layers{l}.b(j)=0;
end
convolution_nn.layers{l}.act_func=ActivationFuncName;
    