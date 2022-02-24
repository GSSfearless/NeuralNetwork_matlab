function [error_train, error_test, out_1, out_2, predict] = batch_train(n, epoch,lr) 
b = 1.6;
x = -b:0.05:b;              % training dataset
y = 1.2*sin(pi*x) - cos(2.4*pi*x);
net = newff(minmax(x),[1,n,1],{'tansig','tansig','purelin'},'trainlm');
net.trainParam.epochs = epoch;
net.trainParam.lr = lr;
net = train(net,x,y);
y2 = sim(net,x);

for i = 1:epoch
    display(['Epoch:',num2str(i)]);
    sum = 0;
    for j = 1:65
        e = abs(x(:,j)-y(:,j));
        sum = sum + e;
        sum = sum/65;
        error_train = mse(e);
        fprintf('Error_train:%6.5f\n',error_train);
    end
end

figure;
title('batch mode');
plot(x,y,'b','LineWidth',1.5);
hold on;
plot(x,y2,'r','LineWidth',1.5);
legend('Ground Truth Line','Approximate Function');
error_train = mse(error_train);
x2 = -1.6:0.01:1.6;                 % step length of testing set is 0.01
y2 = 1.2*sin(pi*x2) - cos(2.4*pi*x2);
predict = net(x2);
error_test = mse(predict - y2);
fprintf('The testing set mean square error is:%6.5f\n ',mse(error_test));
out_1 = net(3);
out_2 = net(-3);
fprintf('x =  3, output:%6.5f\n',out_1);
fprintf('x = -3, output:%6.5f\n',out_2);




