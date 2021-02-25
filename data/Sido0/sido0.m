function data=sido0(seed)
   
    A = importdata('sido0_train.mat');
    
    [n,d] = size(A);
    
    
    B = importdata('sido0_train.targets');
    
    rng(seed);
    perm = randperm(n);

    data.x_train = [A(perm,:) ones(n,1)];
    
    
    [n,d1] = size(data.x_train);
    
    %fprintf('Train: n = %d, and d = %d\n',n,d1);
     fprintf('This is Sido0 train data with n=%d, d=%d\n',n,d);
    
    data.x_train = data.x_train';
    
    data.y_train = B(perm,:);
    
    data.y_train = data.y_train';
    
    
    %%%%%%%%%%%%%%%  TEST  %%%%%%%%%%%%%%%%
    
    E = importdata('sido0_test.mat');
    
    [n,d] = size(E);
    
    
    %%%%TEST data is not available
    F = importdata('ijcnn1_test_label.mat');
    
    rng('default');
    perm = randperm(n);

    data.x_test = [E(perm,:) ones(n,1)]';
    
    [n,d] = size(data.x_test');
    
   fprintf('This is Sido0 train data with n=%d, d=%d\n',n,d-1);
    
    data.y_test = F(perm,:);
    
    data.y_test = data.y_test';
    
    %fprintf('n = %d, and d = %d\n',n,d);
    rng(seed);
    data.w_init = randn(d1,1);
end
