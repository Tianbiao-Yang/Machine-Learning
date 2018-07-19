function [out] = bootstrap(data,B)
% data = 1:1:15;
% B = 5;
[M,N]=size(data);

if (exist('B')~=1), 
    B=N;
end;
 
out=zeros(M,N,B);
index=unidrnd(N,N,B)';
for i = 1:1:B
    c = ismember(data,index(i,:));
    train_data = index(i,:)
    j = find(c==0);
    test_data = data(j)
end
out=reshape(data(:,index),M,N,B);
 
end
