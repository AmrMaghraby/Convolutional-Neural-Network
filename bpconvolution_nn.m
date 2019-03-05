function cnn=bpconvolution_nn(cnn,yy)

if cnn.layers{cnn.no_of_layers}.type ~= 'f'
  zz=[];
  for k=1:cnn.layers{cnn.no_of_layers}.NoOfFeatureMaps
                   ss =size(cnn.layers{cnn.no_of_layers}.featuremaps{k});
                   zz =[zz; reshape(cnn.layers{cnn.no_of_layers}.featuremaps{k}, ss(1)*ss(2), ss(3))];
  end
  cnn.layers{cnn.no_of_layers}.outputs = zz;
end
 er = ( cnn.layers{cnn.no_of_layers}.outputs - yy);
if cnn.loss_func == 'cros' %cross_entropy'
    if cnn.layers{cnn.no_of_layers}.act_func == 'sigm'
        er1 = -1.*sum((yy.*log(cnn.layers{cnn.no_of_layers}.outputs) + (1-yy).*log(1-cnn.layers{cnn.no_of_layers}.outputs)), 1);
    else
          error('cross entropy is implemented only when last layer is sigmoid');
    end
    cnn.loss = sum(er1(:))/size(er1,2); %loss over all examples
else
    er1 = er.^2;
    cnn.loss = sum(er1(:))/(2*size(er1,2)); %loss over all examples
     
end
if cnn.CalcLastLayerActDerivative ==1 
    er =applyactfuncconvolution_nn(cnn.layers{cnn.no_of_layers}.outputs,cnn.layers{cnn.no_of_layers}.act_func, 1, er);
end
%%%%%%%%%%%%%%first only last layer calculation
if cnn.layers{cnn.no_of_layers}.type == 'f'
       cnn.layers{cnn.no_of_layers}.er{1} =er; 
       cnn.layers{cnn.no_of_layers}.dW = cnn.layers{cnn.no_of_layers}.er{1} * ( cnn.layers{cnn.no_of_layers-1}.outputs)' / size(cnn.layers{cnn.no_of_layers}.er{1}, 2);
       cnn.layers{cnn.no_of_layers}.db = mean( cnn.layers{cnn.no_of_layers}.er{1}, 2);
elseif cnn.layers{cnn.no_of_layers}.type == 'p'
    sz2=0;
    for i=1:cnn.layers{cnn.no_of_layers}.NoOfFeatureMaps
        sz = size(cnn.layers{cnn.no_of_layers}.featuremaps{i});
        sz1 = sz(1)*sz(2);
        cnn.layers{cnn.no_of_layers}.er{i} = reshape(er(sz2+1 : sz2+sz1, : ), sz(1), sz(2), sz(3));
        sz2 = sz2+sz1;
    end
    %%upsample er for previous layer
    zz=cnn.layers{cnn.no_of_layers}.subsample_rate;
    if cnn.layers{cnn.no_of_layers}.subsample_method == 'mean'
 
        for i=1:cnn.layers{cnn.no_of_layers}.NoOfFeatureMaps
            sz = size(cnn.layers{cnn.no_of_layers}.featuremaps{i});
            ss1 = 1:sz(1); ss1 =kron(ss1, ones([1 zz]));
            ss2 = 1:sz(2); ss2 =kron(ss2, ones([1 zz]));
            sf{1}=ss1; sf{2}=ss2;
            er =cnn.layers{cnn.no_of_layers}.er{i};
            new_er = er(sf{:},:);
            cnn.layers{cnn.no_of_layers}.er{i} = new_er; % kron( cnn.layers{cnn.no_of_layers}.er{i}, ones([zz zz]));
        end
    else
        error 'this subsampling method not implemented';
    end
elseif cnn.layers{cnn.no_of_layers}.type == 'c'
     sz2=0;
    for i=1:cnn.layers{cnn.no_of_layers}.NoOfFeatureMaps
        sz = size(cnn.layers{cnn.no_of_layers}.featuremaps{i});
        sz1 = sz(1)*sz(2);
        cnn.layers{cnn.no_of_layers}.er{i} = reshape(er(sz2+1 : sz2+sz1, : ), sz(1), sz(2), sz(3));
        sz2 = sz2+sz1;
    end
    kk=0;
    for i=1:cnn.layers{cnn.no_of_layers}.NoOfFeatureMaps
        for j=1:cnn.layers{cnn.no_of_layers-1}.NoOfFeatureMaps
            zz= convn(cnn.layers{cnn.no_of_layers-1}.featuremaps{j}, rot90(cnn.layers{cnn.no_of_layers}.er{i},2), 'valid');
            kk = kk+1;
            cnn.layers{cnn.no_of_layers}.dK(:,:,kk) = mean(zz,3);
        end
        cnn.layers{cnn.no_of_layers}.db(i)= sum(cnn.layers{cnn.no_of_layers}.er{i}(:))/size(cnn.layers{cnn.no_of_layers}.er{i},3);
    end
    
end
for i=cnn.no_of_layers-1:-1:1
    
    if cnn.layers{i}.type == 'f'
       cnn.layers{i}.er{1} = ( (cnn.layers{i+1}.W)' * cnn.layers{i+1}.er{1} );
       cnn.layers{i}.er{1} = applyactfuncconvolution_nn(cnn.layers{i}.outputs,cnn.layers{i}.act_func, 1, cnn.layers{i}.er{1} ); 
       cnn.layers{i}.dW = cnn.layers{i}.er{1} * ( cnn.layers{i-1}.outputs)' / size(cnn.layers{i}.er{1}, 2);
       cnn.layers{i}.db = mean( cnn.layers{i}.er{1}, 2);
    elseif cnn.layers{i}.type == 'p'
           %er =cnn.layers{i+1}.er;
           cnn.layers{i}.er{1}= 0;
           if cnn.layers{i+1}.type == 'f'
                sz2=0;
                er = ( (cnn.layers{i+1}.W)' * cnn.layers{i+1}.er{1} );
                for j=1:cnn.layers{i}.NoOfFeatureMaps
                    sz = size(cnn.layers{i}.featuremaps{j});
                    sz1 = sz(1)*sz(2);
                    cnn.layers{i}.er{j} = reshape(er(sz2+1 : sz2+sz1, : ), sz(1), sz(2), sz(3));
                    sz2 = sz2+sz1;
                end
           elseif cnn.layers{i+1}.type == 'c'
               er =cnn.layers{i+1}.er;
               for k=1:cnn.layers{i}.NoOfFeatureMaps
                   cnn.layers{i}.er{k}=zeros(size(cnn.layers{i}.featuremaps{k}));
               end
               kk=0;
               for j=1:cnn.layers{i+1}.NoOfFeatureMaps
                   for k=1:cnn.layers{i}.NoOfFeatureMaps
                       kk = kk+1;
                      cnn.layers{i}.er{k} = cnn.layers{i}.er{k} + convn(er{j}, rot90(cnn.layers{i+1}.K(:,:,kk),2), 'full'); 
                   end
               end
           else
               er =cnn.layers{i+1}.er;
               cnn.layers{i}.er =er;
           end
           %upsample er for previous layer
           zz=cnn.layers{i}.subsample_rate;
            if cnn.layers{i}.subsample_method == 'mean'
                 for j=1:cnn.layers{i}.NoOfFeatureMaps
                    sz = size(cnn.layers{i}.featuremaps{j});
                    ss1 = 1:sz(1); ss1 =kron(ss1, ones([1 zz]));
                    ss2 = 1:sz(2); ss2 =kron(ss2, ones([1 zz]));
                    sf{1}=ss1; sf{2}=ss2;
                    er =cnn.layers{i}.er{j};
                    new_er = er(sf{:},:);
                    cnn.layers{i}.er{j} = new_er./(zz*zz); % kron( cnn.layers{cnn.no_of_layers}.er{i}, ones([zz zz]));
                 end
            else
                error 'this subsampling method not implemented';
            end            
    elseif cnn.layers{i}.type == 'c'
          er =0;
         cnn.layers{i}.er{1}= 0;
           if cnn.layers{i+1}.type == 'f'
                sz2=0;
                er1 = ( (cnn.layers{i+1}.W)' * cnn.layers{i+1}.er{1} );
                for j=1:cnn.layers{i}.NoOfFeatureMaps
                    sz = size(cnn.layers{i}.featuremaps{j});
                    sz1 = sz(1)*sz(2);
                    cnn.layers{i}.er{j} = reshape(er1(sz2+1 : sz2+sz1, : ), sz(1), sz(2), sz(3));
                    sz2 = sz2+sz1;
                end
                er = cnn.layers{i}.er;
           elseif cnn.layers{i+1}.type == 'c'
               error('not implemented yet- convolution layer with convolution layer. Instead, use Pooling layer with subsampling factor 1 between two conv layer');
           else
               er =cnn.layers{i+1}.er;
           end
           if cnn.layers{i}.act_func == 'soft'
%                error('softmax for backpropagation is not implemented yet');
                 err1 = zeros(size(er{1}));
                 for j=1:cnn.layers{i}.NoOfFeatureMaps
                     err1 = err1 + er{j}.*cnn.layers{i}.featuremaps{j};
                 end
                 for j=1:cnn.layers{i}.NoOfFeatureMaps
                     cnn.layers{i}.er{j} = cnn.layers{i}.featuremaps{j}.*(er{j} - err1);
                 end
           else
                   for j=1:cnn.layers{i}.NoOfFeatureMaps
                       cnn.layers{i}.er{j} =applyactfuncconvolution_nn(cnn.layers{i}.featuremaps{j},cnn.layers{i}.act_func, 1, er{j});
                   end
           end
           kk=0;
            for ii=1:cnn.layers{i}.NoOfFeatureMaps
                for j=1:cnn.layers{i-1}.NoOfFeatureMaps
                    zz= convn(cnn.layers{i-1}.featuremaps{j},flipdim(flipdim(flipdim(cnn.layers{i}.er{ii},1),2),3), 'valid'); %
                    kk = kk+1;
                    cnn.layers{i}.dK(:,:,kk) = zz./size(cnn.layers{i}.er{ii},3); %mean(zz,3);
                end
                cnn.layers{i}.db(ii)= sum(cnn.layers{i}.er{ii}(:))/size(cnn.layers{i}.er{ii},3);
            end
    end
end