function err=testconvolution_nn(convolution_nn, teest_x, teest_y)
 convolution_nn = FullyConnect(convolution_nn, teest_x);
 
 if convolution_nn.layers{convolution_nn.no_of_layers}.type ~= 'f'
  zee=[];
  for k=1:convolution_nn.layers{convolution_nn.no_of_layers}.no_featuremaps
                   see =size(convolution_nn.layers{convolution_nn.no_of_layers}.featuremaps{k});
                   zee =[zee; reshape(convolution_nn.layers{convolution_nn.no_of_layers}.featuremaps{k}, see(1)*see(2), see(3))];
  end
   convolution_nn.layers{convolution_nn.no_of_layers}.outputs = zee;
 end

[~, l1]=max(convolution_nn.layers{convolution_nn.no_of_layers}.outputs, [],1);
[~, l2]=max(teest_y, [], 1);
idx = find(l1 ~= l2);
err = length(idx)/numel(l1);
err;