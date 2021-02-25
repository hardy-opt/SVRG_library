function data = ALLAML(seed)

A = importdata('ALLAML.mat');
A.Y(A.Y==2) = -1;

%id = [668 2497 2733 3170 3252 3561 5466 5565 6041 7128];
%A.X = A.X(:,id); %remove it
X = A.X(1:38,:);
Y = A.Y(1:38,:);

[n,d]=size(A.X);


rng(seed);
perm = randperm(38);


D = [X(perm,:)  ones(38,1)];
s = std(D);
s(s==0)=1;
m=mean(D);
D = (D-m)./s;
data.x_train = D'; % normalize(,'zscore','std')';

data.y_train = Y(perm,:)';

fprintf('This is ALLAML train data with n=%d, d=%d\n',size(data.x_train)');

Z = A.X(39:end,:);
W = A.Y(39:end,:);
rng('default');
per = randperm(34);

data.x_test = [Z(per,:) ones(34,1)]';
data.y_test = W(per,:)';

fprintf('This is ALLAML test data with n=%d, d=%d\n',size(data.x_test)');

rng(seed);
data.w_init = randn(d+1,1);%rand(7129,1);

end