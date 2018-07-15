load cross_validation_data
target = data(:,11);
data = data(:,6); 
[M,N] = size(data);
k = 10; % when k = 1, it's called LOO
indices = crossvalind('Kfold',data(1:M,N),k);
see = 0;
for i = 1:10
    test = (indices == i); 
    train = ~test;
    train_data = data(train,:);
    train_target = target(train,:);
    test_data = data(test,:);
    test_target = target(test,:);
    yhat = polyval(polyfit(train_data,train_target,2),test_data);
    sse = see + sum(yhat - test_target);
end
CVerr = sse / 100;