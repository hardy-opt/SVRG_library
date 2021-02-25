function data=ijcnn1(seed)
   
    A = importdata('ijcnn1_train_inst.mat');
    
    [n,d] = size(A);
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    
    B = importdata('ijcnn1_train_label.mat');
    
 
    
    
    %data.x_train = A;
    
    %data.y_train = B;
    rng(seed);
    perm = randperm(n);

    data.x_train = [A(perm,:) ones(n,1)] ;
    
    
    [n,d1] = size(data.x_train);
    
    %fprintf('Train: n = %d, and d = %d\n',n,d1);
    
    x_train = data.x_train';
    
   % mie=min(min(x_train))
    %mxe=max(max(x_train))
    
    %fprintf('min element=%.3e and max element=%.3e',mie,mxe);
    
    %data.x_train = normalize(x_train,'range');
    
    data.x_train = x_train;
    
    fprintf('This is IJCNN1 train  data with n=%d, d=%d\n',size(data.x_train'));
    
    data.y_train = B(perm,:);
    
    [n,d] = size(data.y_train);
    
    data.y_train = data.y_train';
    
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    
   
    %%%%%%%%%%%%%% VAL  %%%%%%%%%%%%%%%%%%%
    
    
      C = importdata('ijcnn1_val_inst.mat');
    
    [n,d] = size(C);
    
    fprintf('Validation: n = %d, and d = %d\n',n,d);
    
    
    D = importdata('ijcnn1_val_label.mat');
    
    
    %data.x_test = C;
    
    %data.y_test = D;
    
    rng(seed);
    perm = randperm(n);

    data.x_val = C(perm(1:n),:);
    
    [n,d] = size(data.x_val);
    
    x_val = data.x_val';
    
    
    %data.x_val = normalize(x_val,'range');
    data.x_val = x_val;
    
   % fprintf('n = %d, and d = %d\n',n,d);
    
    
    data.y_val = D(perm(1:n),:);
    
    [n,d] = size(data.y_val);
    
    data.y_val = data.y_val';
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    
    
    
    
    
    %%%%%%%%%%%%%%%  TEST  %%%%%%%%%%%%%%%%
    
    E = importdata('ijcnn1_test_inst.mat');
    
    [n,d] = size(E);
    
    %fprintf('Test: n = %d, and d = %d\n',n,d);
    
    
    F = importdata('ijcnn1_test_label.mat');
    
    
    %data.x_test = C;
    
    %data.y_test = D;
    
    rng('default');
    perm = randperm(n);

    data.x_test = [E(perm,:) ones(n,1)];
    
    [n,d] = size(data.x_test);
    
    x_test = data.x_test';

 %   data.x_test = normalize(x_test,'range');
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    fprintf('Also, IJCNN1 test  data with n=%d, d=%d\n',size(data.x_test'));
    
    data.y_test = F(perm(1:n),:);
    
    [n,d] = size(data.y_test);
    
    data.y_test = data.y_test';
    
    %fprintf('n = %d, and d = %d\n',n,d);
    rng(seed);
    data.w_init = rand(d1,1);
end
