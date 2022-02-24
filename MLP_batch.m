function [acc_train,acc_test] = MLP_batch(batch,input_train,input_test,n,epoch,lr,rgl,Fcn)
[m1,n1] = size(input_train);
[m2,n2] = size(input_test);

train_data = input_train(1:m1-1,:);
train_label = input_train(m1,:); 
test_data = input_test(1:m2-1,:);
test_label = input_test(m2,:);

test_data = repmat(test_data,1,9);
test_label = repmat(test_label,1,9);

train_data = num2cell(train_data,1);
train_label = num2cell(train_label,1);
test_data = num2cell(test_data,1);
test_label = num2cell(test_label,1);

net = patternnet(n);
net.trainParam.epochs = epoch;
net.trainParam.lr = lr;
net.performParam.regularization = rgl;
net.trainFcn = Fcn;
net.performFcn = 'crossentropy';
net.trainParam.showWindow = false;
net = configure(net,train_data,train_label); 

train_result = round(cell2mat(net(train_data)));
acc_train = 0;
for i = 1:length(input_train)
    accuracy_train = 1 - abs(mean(train_result - cell2mat(train_label)));
    acc_train = acc_train + accuracy_train;
end
acc_train = acc_train/length(input_train);



test_result = round(cell2mat(net(test_data)));
acc_test = 0;
for i = 1:length(input_test)
    accuracy_test = 1 - abs(mean(test_result - cell2mat(test_label)));
    acc_test = acc_test + accuracy_test;
end
acc_test = acc_test/length(input_test);





