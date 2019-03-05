clear;
clc;
disp 'Start-Building-CNN-From-Scratch'
Datapath = 'E:\AAST Portfolio\Semester 7\Project 1\MNIST\Extract\';
disp 'reading MNIST dataset...'
%% Copyrights -- This Section is copied from internet blogs "How To read MNist files in Matlab" 
f=fopen(fullfile(Datapath, 'train-images.idx3-ubyte'),'r', 'b'); 
if f < 0
    error('please load MNIST dataset, store it in a folder and check the path and name of the file');
end
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
h=fread(f,1,'int32');
w=fread(f,1,'int32');
train_x = uint8(fread(f,h*w*num,'uchar')); %load train images
train_x = permute(reshape(train_x, h, w,num), [2 1 3]);
train_x = double(train_x)./255;
fclose(f);

f=fopen(fullfile(Datapath, 't10k-images.idx3-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
h=fread(f,1,'int32');
w=fread(f,1,'int32');
test_x = uint8(fread(f,h*w*num,'uchar')); %load train images
test_x = permute(reshape(test_x, h, w,num), [2 1 3]);
test_x = double(test_x)./255;
fclose(f);

ff=fopen(fullfile(Datapath, 'train-labels.idx1-ubyte'),'r', 'b') ;
nn=fread(ff,1,'int32');
num=fread(ff,1,'int32');
y = double(fread(ff,num,'uint8'));   %load train labels
y = (y)'; %.
train_y = zeros([10 num]); % there are 10 labels in MNIST lables
for i=0:9 % labels are 0 - 9
    k = find(y==i);
    train_y(i+1,k)=1;
end
fclose(ff) ;
f=fopen(fullfile(Datapath, 't10k-labels.idx1-ubyte'),'r', 'b') ;
nn=fread(f,1,'int32');
num=fread(f,1,'int32');
y = double(fread(f,num,'uint8')); %load test labels
y = (y)' ;
test_y = zeros([10 num]); % there are 10 labels in MNIST lables
for i=0:9 % labels are 0 - 9
    k = find(y==i);
    test_y(i+1,k)=1;
end
fclose(f) ;
%% Ending of copied Section 
convolution_nn.namaste=1; 
convolution_nn=initcnn(convolution_nn,[h w]);

convolution_nn=convolutionAddConvLayer(convolution_nn, 10, [9 9], 'rect');
convolution_nn=convolutionAddPoolLayer(convolution_nn, 2, 'mean');
%%convolution_nn=convolutionAddConvLayer(convolution_nn, 20, [3 3], 'rect');
%%convolution_nn=convolutionAddPoolLayer(convolution_nn, 2, 'mean');
%%convolution_nn=convolutionAddConvLayer(convolution_nn, 40, [3 3], 'rect');
%%convolution_nn=convolutionAddPoolLayer(convolution_nn, 2, 'mean');
%%convolution_nn=convolutionAddFCLayer(convolution_nn,150, 'tanh' ); 
convolution_nn=convolutionAddFCLayer(convolution_nn,10, 'sigm' );
epochs = 500;
batchSize=50;
convolution_nn=trainconvolution_nn(convolution_nn,train_x,train_y,test_x,test_y,epochs,batchSize);
disp 'Training Finished...'
disp 'Testing Started...'
accuracypercent=(1-(testconvolution_nn(convolution_nn, test_x, test_y)))*100;
characc=num2str(accuracypercent);
Accuracy=strcat((characc), '%');
display (Accuracy);
display 'testing finished.'
