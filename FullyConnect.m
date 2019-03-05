function convolution_nn=FullyConnect(convolution_nn, xxx)
if convolution_nn.no_of_input_channels > 1
    for i=1:convolution_nn.no_of_input_channels 
        convolution_nn.layers{1}.featuremaps{i}=xxx(:,:,i,:);
    end
else
    convolution_nn.layers{1}.featuremaps{1}=xxx;
end
for i=2:convolution_nn.no_of_layers
    if convolution_nn.layers{i}.type == 'c'
        kii=0;
        zee=0;
        for j=1:convolution_nn.layers{i}.NoOfFeatureMaps
            z = 0; 
            for k=1:convolution_nn.layers{i-1}.NoOfFeatureMaps
                kii = kii +1;
                z = z + convn(convolution_nn.layers{i-1}.featuremaps{k},rot90(convolution_nn.layers{i}.K(:,:,kii),2),'valid');
            end
            if convolution_nn.layers{i}.act_func == 'soft'
                convolution_nn.layers{i}.featuremaps{j}= exp(z + convolution_nn.layers{i}.b(j));
                zee = zee + convolution_nn.layers{i}.featuremaps{j};
            else
                convolution_nn.layers{i}.featuremaps{j} = applyactfuncconvolution_nn(z+ convolution_nn.layers{i}.b(j),convolution_nn.layers{i}.act_func, 0);
            end
        end
        if convolution_nn.layers{i}.act_func == 'soft'
            for j=1:convolution_nn.layers{i}.NoOfFeatureMaps
                convolution_nn.layers{i}.featuremaps{j}= convolution_nn.layers{i}.featuremaps{j} ./ zee;
            end
        end
    elseif convolution_nn.layers{i}.type == 'p'  
            if convolution_nn.layers{i}.subsample_method == 'mean'
                h = ones([convolution_nn.layers{i}.subsample_rate convolution_nn.layers{i}.subsample_rate]);
                h=h./sum(h(:));
                for k=1:convolution_nn.layers{i-1}.NoOfFeatureMaps
                    zee = convn(convolution_nn.layers{i-1}.featuremaps{k}, h, 'valid'); %%'same'
                    convolution_nn.layers{i}.featuremaps{k} = zee(1:convolution_nn.layers{i}.subsample_rate:end, 1:convolution_nn.layers{i}.subsample_rate:end,:);
                end
            elseif convolution_nn.layers{i}.subsample_method == 'max '
                error 'max pooling not implemented'
            end
    elseif convolution_nn.layers{i}.type == 'f'
            zee=0;
            zee=[];
            if convolution_nn.layers{i-1}.type  ~= 'f'
                for k=1:convolution_nn.layers{i-1}.NoOfFeatureMaps
                   ss =size(convolution_nn.layers{i-1}.featuremaps{k});
                   ss(3) =size(convolution_nn.layers{i-1}.featuremaps{k},3);
                   if convolution_nn.input_image_width == 1
                       ss(3) =ss(2);
                       ss(2)=1;
                   end
                   zee =[zee; reshape(convolution_nn.layers{i-1}.featuremaps{k}, ss(1)*ss(2), ss(3))];
                   
                end
                convolution_nn.layers{i-1}.outputs = zee;
                convolution_nn.layers{i}.outputs = applyactfuncconvolution_nn(convolution_nn.layers{i}.W*zee + repmat(convolution_nn.layers{i}.b, 1, size(zee,2)), convolution_nn.layers{i}.act_func, 0); 
                
            else
                zee= convolution_nn.layers{i-1}.outputs;
                convolution_nn.layers{i}.outputs = applyactfuncconvolution_nn(convolution_nn.layers{i}.W*zee + repmat(convolution_nn.layers{i}.b, 1, size(zee,2)), convolution_nn.layers{i}.act_func, 0); 
            end
                
        
    end
    
end