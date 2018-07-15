load cross_validation_data
target = data(:,11);
data = data(:,6); 
[M,N] = size(data);
k = 10; % when k = 1, it's called LOO
error = 0;
indices = crossvalind('Kfold',data(1:M,N),k);
for i = 1:10
    test = (indices == i); 
    train = ~test;
    train_data = data(train,:);
    train_target = target(train,:);
    test_data = data(test,:);
    test_target = target(test,:);
    pre_target = polyval(polyfit(train_data,train_target,2),test_data);
    error = error + sum(pre_target - test_target);
end
average_error = error / 100;