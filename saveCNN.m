function saveCNN(cnn, filename)
%%%% save cnn weights and biases in a file
cnn1.namaste = 1;
cnn1.no_of_layers = cnn.no_of_layers;
cnn1.input_image_height = cnn.input_image_height;
cnn1.input_image_width = cnn.input_image_width; 
cnn1.no_of_input_channels = cnn.no_of_input_channels;
cnn1.layers{1}.type = cnn.layers{1}.type;
cnn1.layers{1}.NoOfFeatureMaps = cnn.layers{1}.NoOfFeatureMaps;
cnn1.layers{1}.featuremap_width = cnn.layers{1}.featuremap_width;
cnn1.layers{1}.featuremap_height =cnn.layers{1}.featuremap_height ;
cnn1.layers{1}.prev_layer_NoOfFeatureMaps = cnn.layers{1}.prev_layer_NoOfFeatureMaps;
%default parameters
cnn1.loss_func=cnn.loss_func;
cnn1.regularization_const = cnn.regularization_const;
cnn1.learning_rate =cnn.learning_rate;
for i=2:cnn.no_of_layers
    cnn1.layers{i}.type = cnn.layers{i}.type;
    if cnn.layers{i}.type == 'c'
        kk=0;
        for j=1:cnn.layers{i}.NoOfFeatureMaps
            for k=1:cnn.layers{i-1}.NoOfFeatureMaps
                kk = kk+1;
                cnn1.layers{i}.K(:,:,kk)= cnn.layers{i}.K(:,:,kk);
            end
            cnn1.layers{i}.b(j)= cnn.layers{i}.b(j);
        end
        cnn1.layers{i}.act_func= cnn.layers{i}.act_func;
        cnn1.layers{i}.NoOfFeatureMaps = cnn.layers{i}.NoOfFeatureMaps;
        cnn1.layers{i}.KernelWidth = cnn.layers{i}.KernelWidth;
        cnn1.layers{i}.KernelHeight = cnn.layers{i}.KernelHeight;
        cnn1.layers{i}.featuremap_width = cnn.layers{i}.featuremap_width;
        cnn1.layers{i}.featuremap_height = cnn.layers{i}.featuremap_height; 
    end
    if cnn.layers{i}.type == 'p'
        cnn1.layers{i}.subsample_rate = cnn.layers{i}.subsample_rate;
        cnn1.layers{i}.subsample_method = cnn.layers{i}.subsample_method;
        cnn1.layers{i}.NoOfFeatureMaps = cnn.layers{i}.NoOfFeatureMaps;
        cnn1.layers{i}.featuremap_width = cnn.layers{i}.featuremap_width;
        cnn1.layers{i}.featuremap_height = cnn.layers{i}.featuremap_height;
        cnn1.layers{i}.act_func = cnn.layers{i}.act_func; 
    end
     if cnn.layers{i}.type == 'f'
         cnn1.layers{i}.NoOfNodes = cnn.layers{i}.NoOfNodes;
         cnn1.layers{i}.act_func = cnn.layers{i}.act_func;
         cnn1.layers{i}.no_of_inputs = cnn.layers{i}.no_of_inputs;
         cnn1.layers{i}.convert_input_to_1D = cnn.layers{i}.convert_input_to_1D;
         cnn1.layers{i}.W = cnn.layers{i}.W;
         cnn1.layers{i}.b = cnn.layers{i}.b ;
     end
end
save(filename, 'cnn1');