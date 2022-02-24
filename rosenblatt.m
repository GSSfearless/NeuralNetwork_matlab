function [w,error,loss]=rosenblatt(data,iter,lr)
s = size(data);
m = s(1);
n = s(2);
data = double([ones(m,1) data(:,1:s(2)-1)]);
label = double(data(:,n));
w = rand(n,1);
y = double(zeros(m,1));
for i = 1: iter
    error = 0;
    result = data * w;
    y(result>0) = 1;
    y(result<=0) = 0;
    e = double(abs(label - y));
    w = w + (lr * e' * data)';
    w_list(:,i) = w;
    error = error + 0;
    loss = 0.5* (error.^2);
    disp(['Rosenblatt epoch:',num2str(i)])
end
