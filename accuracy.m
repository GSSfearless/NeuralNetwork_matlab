function accuracy = accuracy(data, w)
s = size(data);
m = s(1);
n = s(2);
Data = double([ones(m,1) data(:,1:n-1)]);
label = double(data(:,n));
y = double(zeros(m,1));
result = Data * w;
y(result > 0) = 1;
y(result <= 0) = 0;
accuracy = mean(1 - abs(label - y));



