function convolution_nn=trainconvolution_nn(convolution_nn,x,y, tx, ty, epochs,batchSize)
m=1;
m_index=1;
if size(x,4) > 1 
    m=size(x,4);
    m_index=4; 
else
    m=size(x,3);
    m_index=3;
end

no_of_batches = m/batchSize;
if rem(m, batchSize) ~=0
    error('no_of_batches should be integer');
end
%% This part is contributed by someone through Blogs
if convolution_nn.loss_func == 'auto'
   convolution_nn.loss_func = 'quad'; %quadtratic
   if convolution_nn.layers{convolution_nn.no_of_layers}.act_func == 'sigm'
       convolution_nn.loss_func = 'cros' ; %cross_entropy';
   elseif convolution_nn.layers{convolution_nn.no_of_layers}.act_func == 'tanh'
       convolution_nn.loss_func = 'quad'; 
   end
elseif strcmp(convolution_nn.loss_func, 'cros') == 1 & strcmp(convolution_nn.layers{convolution_nn.no_of_layers}.act_func, 'sigm') == 0
    display 'Not tested for gradient checking for cross entropy cost function other than sigm layer'
end
convolution_nn.CalcLastLayerActDerivative = 1;
if convolution_nn.loss_func == 'cros' 
    if convolution_nn.layers{convolution_nn.no_of_layers}.act_func == 'soft'
        convolution_nn.CalcLastLayerActDerivative =0;
    elseif convolution_nn.layers{convolution_nn.no_of_layers}.act_func == 'sigm'
        convolution_nn.CalcLastLayerActDerivative =0;
    end    
end
if convolution_nn.layers{convolution_nn.no_of_layers}.act_func == 'none'
    convolution_nn.CalcLastLayerActDerivative =0;
end
display 'training started...'
epoch_accuracy=[];
validation_accuracy=[];
t=0;
for i=1:epochs
    convolution_nn.loss_array=[];
    batch_error=[];
    tic
    for j=1:batchSize:m
        if m_index==4
            xx = x(:,:,:,j:j+batchSize-1);
        else
            xx = x(:,:,j:j+batchSize-1);
        end
        yy =y(:,j:j+batchSize-1);
        
        convolution_nn= FullyConnect(convolution_nn, xx);
        
        [~, l1]=max(convolution_nn.layers{convolution_nn.no_of_layers}.outputs, [],1);
        [~, l2]=max(yy, [], 1);
        idx = find(l1 ~= l2);
        err = length(idx)/numel(l1);
        batch_error=[batch_error err];
    
        convolution_nn = bpconvolution_nn(convolution_nn,yy);
        convolution_nn =gradientdescentconvolution_nn(convolution_nn);
        convolution_nn.loss_array = [convolution_nn.loss_array convolution_nn.loss];
    end
    toc
    t=t+toc;
    err=sum(batch_error);
    avg=(1-(err/no_of_batches))*100;
    i
    avg
    epoch_accuracy=[epoch_accuracy avg];
    convolution_nn = FullyConnect(convolution_nn, tx);
    [~, t1]=max(convolution_nn.layers{convolution_nn.no_of_layers}.outputs, [],1);
    [~, t2]=max(ty, [], 1);
    idx = find(t1 ~= t2);
    valid_acur = (1-(length(idx)/numel(t1)))*100;
    validation_accuracy=[validation_accuracy valid_acur];
end
t
epoch_accuracy
plot(1:epochs, epoch_accuracy, '-s', 1:epochs, validation_accuracy, '-s')
xlabel('Epoch')
legend({'Training','Validation'},'Location','southeast')
ylabel('Accuracy')
title('Relation Between Accuracy and # of Epochs')