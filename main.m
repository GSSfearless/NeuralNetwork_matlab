%%%%%%%%%%%%%%%%%%%%%%%%%
% GanRunze Assignment 2
% 2021/2/15
%% Q1 Gradient Descent method
clear; close all; clc;
iter = 2000;
threshold = 0.01; 
eta = 0.001;
x0 = rand;
y0 = rand;
x = x0;
y = y0;
Xmin = -2;
Xmax = 2;
delta = 0.01;
Ymin = -2;
Ymax = 2;
X = [Xmin: delta: Xmax];
Y = [Ymin: delta: Ymax];
[xx ,yy] = meshgrid(X,Y);
F = (1-xx).^2 + 100*(yy-xx.^2).^2;
t1 = 0;
for i = 1:iter
    t1 = t1 + 1;
    gx = 2*(1-x) + 400*x*(y-x^2);
    gy = -200 * (y-x^2);
    x = x + eta*gx;
    y = y + eta*gy;
    P_s(1,t1) = x;
    P_s(2,t1) = y;
    G = plot(x,y,'LineWidth',1.5);
    set(get(get(G, 'Annotation'),'LegendInformation'),'IconDisplayStyle', 'off');
    E = (1-x)^2 + 100*(y-x^2)^2; 
    if abs(E) < threshold
        fprintf('Error < %6.5f\n', E);
        fprintf('Iteration: %d\n',t1);
        break
    end
    i
end
hold on;

% Newtons Methods
x = x0;
y = y0;
iter2 = 100;
threshold2 = 1e-8;
H = zeros(2,2);
g = zeros(2,1);
gradient = zeros(2,1);
t2 = 0;
for i = 1: iter2
    t2 = t2 + 1;
    g = [2*(1-x) + 400*x*(y-x^2);-200 * (y-x^2)];
    H = [(-2 + 400*y -800 * x^2),400*x;400*x,-200];
    gradient = -inv(H)*g;
    x = x + gradient(1);
    y = y + gradient(2);
    P2(1,t2) = x;
    P2(2,t2) = y;
     N = plot(x,y,'LineWidth',1.5);
    set(get(get(N,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
    E2 = (1-x)^2 + 100*(y-x^2)^2;
    if E2 < threshold2
        fprintf('Error is %6.5f',E2);
        fprintf('iteration',t2);
        break
    end
    i
end


origin= plot(1,1,'ro','MarkerSize',5,'MarkerFaceColor','r');
set(get(get(origin,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
text(1.05,1,'(1,1)','FontSize',10);
GD = plot(P_s(1,1:t1),P_s(2,1:t1),'r','LineWidth',1.5);
hold on;
Newton = plot(P2(1,1:t2),P2(2,1:t2),'b','LineWidth',1.5);
legend({'Gradient Descent','Newtons Method'});
hold on;
contour(xx, yy , F);

%% Q2 sequential mode training
clear; close all; clc;
b = 1.6;
x = -b:0.01:b;
y = 1.2*sin(pi*x) - cos(2.4*pi*x);
neuron_list = [1 2 5 8 10 20 50 100];

lr = 0.01;
epoch_s = 3000;
P_s = zeros(4,length(neuron_list));
Predict_s = zeros(length(y),length(neuron_list));
for i = 1:length(neuron_list)
    
    [P_s(1,i),P_s(2,i),P_s(3,i),P_s(4,i),Predict_s(:,i)] = seq_train(neuron_list(i),epoch_s,lr);

end
%% batch mode train using trainlm
clear; close all; clc;
b = 1.6;
x = -b:0.01:b;
y = 1.2*sin(pi*x) - cos(2.4*pi*x);
x2 = -b:0.05:b;
neuron_list = [1 2 5 8 10 20 50 100];

lr = 0.01;
epoch_b = 200;
P_b = zeros(4,length(neuron_list));
Predict_b = zeros(length(y),length(neuron_list));
for i = 1:length(neuron_list)

    [P_b(1,i),P_b(2,i),P_b(3,i),P_b(4,i),Predict_b(:,i)] = batch_trainlm(neuron_list(i),epoch_b,lr);
end

%% batch mode train using trainbr
clear; close all; clc;
b = 1.6;
x = -b:0.01:b;
y = 1.2*sin(pi*x) - cos(2.4*pi*x);
x2 = -b:0.05:b;
neuron_list = [1 2 5 8 10 20 50 100];
lr = 0.01;
epoch_b = 200;
P_b = zeros(4,length(neuron_list));
Predict_b = zeros(length(y),length(neuron_list));
for i = 1:length(neuron_list)

    [P_b(1,i),P_b(2,i),P_b(3,i),P_b(4,i),Predict_b(:,i)] = batch_trainbr(neuron_list(i),epoch_b,lr);
end


%% Q3 Image Classification Task
close all;clear;clc;

% transform of automobile_train 
path_auto = '/home/ryan/NNHomework/HW2/group_1/automobile_train/';
train_auto_list = dir(strcat(path_auto , '*.jpg'));
input_auto = zeros(32*32,length(train_auto_list));
input_auto_label = zeros(1,length(train_auto_list));
for i = 1:length(train_auto_list)
    image_auto = imread([path_auto, train_auto_list(i).name]);
    image_auto = rgb2gray(image_auto);
    input_auto(:,i) = image_auto(:);
    disp(['automobile_train',num2str(i)]);
end
save('/home/ryan/NNHomework/HW2/train_auto.mat');

% transform of automobile_test
path_auto_test = '/home/ryan/NNHomework/HW2/group_1/automobile_test/';
train_auto_test_list = dir(strcat(path_auto_test , '*.jpg'));
input_auto_test = zeros(32*32,length(train_auto_test_list));
input_auto_test_label = zeros(1,length(train_auto_test_list));
for i = 1:length(train_auto_test_list)
    image_auto_test = imread([path_auto_test, train_auto_test_list(i).name]);
    image_auto_test = rgb2gray(image_auto_test);
    input_auto_test(:,i) = image_auto_test(:);
    disp(['automobile_test',num2str(i)]);
end
save('/home/ryan/NNHomework/HW2/test_auto.mat');

% transform of dog_train
path_dog = '/home/ryan/NNHomework/HW2/group_1/dog_train/';
train_dog_list = dir(strcat(path_dog , '*.jpg'));
input_dog = zeros(32*32,length(train_dog_list));
input_dog_label = ones(1,length(train_dog_list));
for i = 1:length(train_dog_list)
    image_dog = imread([path_dog, train_dog_list(i).name]);
    image_dog = rgb2gray(image_dog);
    input_dog(:,i) = image_dog(:);
     disp(['dog_train',num2str(i)]);
end
save('/home/ryan/NNHomework/HW2/train_dog.mat','input_dog','input_dog_label');

% transform of dog_test
path_dog_test = '/home/ryan/NNHomework/HW2/group_1/dog_test/';
train_dog_test_list = dir(strcat(path_dog_test , '*.jpg'));
input_dog_test = zeros(32*32,length(train_dog_test_list));
input_dog_test_label = ones(1,length(train_dog_test_list));
for i = 1:length(train_dog_test_list)
    image_dog_test = imread([path_dog_test, train_dog_test_list(i).name]);
    image_dog_test = rgb2gray(image_dog_test);
    input_dog_test(:,i) = image_dog_test(:);
    disp(['dog_test',num2str(i)]);
end
save('/home/ryan/NNHomework/HW2/test_dog.mat');

train_data = [input_auto input_dog];    
% size of train_data is 1024*900
train_label = [input_auto_label input_dog_label];
% size of train_label is 1*900
test_data = [input_auto_test input_dog_test];
% test_data
test_label = [input_auto_test_label input_dog_test_label];
% test_label

epoch = 1000;
lr = 0.1;
input_train = [train_data; train_label]';
input_test = [test_data; test_label]';
[w,e,loss_train] = rosenblatt(input_train,epoch,lr);
acc_train = accuracy(input_train,w);
acc_test = accuracy(input_test,w);
fprintf('Training accuracy of Rosenblatt perceptron is: %6.5f\n',acc_train);
fprintf('Testing accuracy of Rosenblatt perceptron is: %6.5f\n',acc_test);

neuron = 20;
epoch = 1000;
lr = 0.1;
Fcn_list = string({'trainlm','trainbr','traingdx','trainbfg','traincgb'});% 5
rgl_list = [0 0.1 0.2 0.4 0.6 0.8 1.0];% 7
input_train = input_train';
input_test = input_test';
batch = 10;
% MLP batch

acc_batch = zeros(length(Fcn_list),length(rgl_list),2);
for i = 1: length(Fcn_list)
    for j = 1: length(rgl_list)
    [acc_batch(i,j,1),acc_batch(i,j,2)] = MLP_batch(batch,input_train,input_test, ...
        neuron,epoch,lr,rgl_list(j),Fcn_list(i));
    end
end

b = bar3(acc_batch(:,:,1),0.6);
title('MLP batch training set accuracy');
xticklabels({'0','0.1','0.2','0.4','0.6','0.8','1.0'});
xlabel('Regularization list');
yticklabels({'trainlm','trainbr','traingdx','trainbfg','traincgb'});
ylabel('TrainFcn list');
zlabel('Accuracy');

figure;
bar3(acc_batch(:,:,2));
title('MLP batch validation set accuracy');
xticklabels({'0','0.1','0.2','0.4','0.6','0.8','1.0'});
xlabel('Regularization list');
yticklabels({'trainlm','trainbr','traingdx','trainbfg','traincgb'});
ylabel('TrainFcn list');
zlabel('Accuracy');

figure;
plot(acc_batch(:,:,1));
title('MLP batch training set accuracy');
xticklabels({'trainlm','','trainbr','','traingdx', ...
    '','trainbfg','','traincgb'});
xlabel('TrainFcn list');
ylabel('Accuracy');
legend('0','0.1','0.2','0.4','0.6','0.8','1.0');

figure;
plot(acc_batch(:,:,2));
title('MLP batch validation set accuracy');
xticklabels({'trainlm','','trainbr','','traingdx', ...
    '','trainbfg','','traincgb'});
xlabel('TrainFcn list');
ylabel('Accuracy');
legend('0','0.1','0.2','0.4','0.6','0.8','1.0');

%%
% MLP Sequential

result = zeros(length(Fcn_list),length(rgl_list),2);
for i = 1: length(Fcn_list)
    for j = 1: length(rgl_list)
    [result(i,j,1),result(i,j,2)] = MLP_sequential(input_train,input_test, ...
        neuron,epoch,lr,rgl_list(j),Fcn_list(i));
    end
end


b = bar3(result(:,:,1),0.6);
title('MLP sequential training set accuracy');
xticklabels({'0','0.1','0.2','0.4','0.6','0.8','1.0'});
xlabel('Regularization list');
yticklabels({'trainlm','trainbr','traingdx','trainbfg','traincgb'});
ylabel('TrainFcn list');
zlabel('Accuracy');

figure;
bar3(result(:,:,2));
title('MLP sequential validation set accuracy');
xticklabels({'0','0.1','0.2','0.4','0.6','0.8','1.0'});
xlabel('Regularization list');
yticklabels({'trainlm','trainbr','traingdx','trainbfg','traincgb'});
ylabel('TrainFcn list');
zlabel('Accuracy');

figure;
plot(result(:,:,1));
title('MLP sequential training set accuracy');
xticklabels({'trainlm','','trainbr','','traingdx', ...
    '','trainbfg','','traincgb'});
xlabel('TrainFcn list');
ylabel('Accuracy');
legend('0','0.1','0.2','0.4','0.6','0.8','1.0');

figure;
plot(result(:,:,2));
title('MLP sequential validation set accuracy');
xticklabels({'trainlm','','trainbr','','traingdx', ...
    '','trainbfg','','traincgb'});
xlabel('TrainFcn list');
ylabel('Accuracy');
legend('0','0.1','0.2','0.4','0.6','0.8','1.0');


%%



