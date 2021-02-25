function data= w8a(seed)
   
    A = importdata('w8a_inst.mat');
    
    [n,d1] = size(A);
    
    fprintf('n = %d, and d = %d\n',n,d1);
    
    
    B = importdata('w8a_label.mat');
    
 
    
    
    %data.x_train = A;
    
    %data.y_train = B;
    rng(seed);
    perm = randperm(n);

    data.x_train = [A(perm,:) ones(n,1)];
    
    [n,d] = size(data.x_train);
   
    fprintf('This is W8A train data with n=%d, d=%d\n',n,d-1);
  %  fprintf('n = %d, and d = %d\n',n,d);
    
    x_train = data.x_train';
    
     %mie=min(min(x_train))
    %mxe=max(max(x_train)) No need to normalize
    
    data.x_train = x_train;
    
    %data.x_train = normalize(x_train,'range');
    
    data.y_train = B(perm,:);
    
    [n,d] = size(data.y_train);
    
    data.y_train = data.y_train';
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    
    %%%%%%%%%%%%%%%  TEST  %%%%%%%%%%%%%%%%
    
    C = importdata('w8a_tst_inst.mat');
    
    [n,d] = size(C);
    
    
    D = importdata('w8a_tst_label.mat');
    
    
    
    rng('default');
    perm = randperm(n);

    x_test =[ C(perm,:) ones(n,1)];
     
    data.x_test= x_test';
    
    [n,d] = size(data.x_test');
  
    data.y_test = D(perm,:);
    
    data.y_test = data.y_test';
    
    %fprintf('n = %d, and d = %d\n',n,d);
    
    fprintf('Also,  W8A test  data with n=%d, d=%d\n',n,d-1);
    rng(seed);
    data.w_init = randn(d1+1,1);
end
