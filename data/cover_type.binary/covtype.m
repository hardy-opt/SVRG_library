function data= covtype(seed)
   
    A = full(importdata('covtype_train.mat'))';
    
    [d,n] = size(A)
    
   
    B = importdata('covtype_label.mat')';
    
    [d,n] = size(B);
    
    
    C = A(:,406001:n);
    D = B(406001:n);
    %fprintf('n = %d, and d = %d\n',n,d);
    
%     C = [A, B]; 
%     
%     [n,d] = size(C);
%     
%     for i = 1:2
%     
%     %perm = randperm(n);
%     cv = cvpartition(size(C,1),'HoldOut', 0.3);
%     idx = cv.test;
%     
%     tr_in = C(~idx,:);
%     tst_in = C(idx,:);
%     
%     x_train{i} = tr_in(:,1:d-1);
%     x_label{i} = tr_in(:,d);
%     
%     save(['x_train' num2str(i) '.mat'],'i')
%     %save(sprintf('x_train{%d}',i), sprintf('x_train{%d}',i) );
%     %save('x_label{i}','x_label{i}');
%     end
    
    rng(seed);
    perm = randperm(406000);

    data.x_train = [A(:,perm) ;ones(1,406000)];
    
    
    
    data.y_train = B(:,perm);
    
    rng('default');
    perm=randperm(175012);
    
    data.x_test = [C(:,perm); ones(1,175012)];
    
    
    
    data.y_test = D(:,perm);
    
    
    [d,n] = size(data.x_train);
    
    %fprintf('n = %d, and d = %d\n',n,d);
    fprintf('This is Covtype train data with n=%d, d=%d\n',n,d-1);
    
    [d,n] = size(data.y_train);
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    
    [d1,n] = size(data.x_test);
    
    %fprintf('n = %d, and d = %d\n',n,d1);
    fprintf('This is Covtype test data with n=%d, d=%d\n',n,d1-1);
    
    [d,n] = size(data.y_test);
    
    %fprintf('n = %d, and d = %d\n',n,d);
    rng(seed);
    data.w_init = randn(d1,1);
    
end