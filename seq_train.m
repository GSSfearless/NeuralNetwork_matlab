function [error_train, error_test, out_1, out_2, predict] = seq_train(n, epoch,lr)

x = -1.6:0.05:1.6;                % 0.05 is the training set step length
y = 1.2*sin(pi*x) - cos(2.4*pi*x);
net = newff(minmax(x),[1,n,1],{'tansig','tansig','purelin'},'traingd');
net.trainParam.epochs = epoch;
net.trainParam.lr = lr;
% x = num2cell(x);
% y = num2cell(y);
net = train(net,x,y);
y2=sim(net,x);

for i = 1:epoch
    display(['Epoch:', num2str(i)]);
    idx = randperm(65);
    error_train = abs(x(:,idx)-y(:,idx));
    fprintf('The training set mean square error is:%6.5f\n',mse(error_train));
    if mse(error_train) < 1e-4
        break
    end
end
figure;
title('sequential mode with BP');
xlabel('x');
ylabel('y');
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











