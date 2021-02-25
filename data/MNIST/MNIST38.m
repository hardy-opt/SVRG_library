function data = MNIST38(seed)
%     T = readmatrix('mnist_test.csv'); % /train
%     [n,d] = size(T);
%     
%     fprintf('This is MNIST data with n=%d, d=%d\n',n,d-1);
%     
%     A = T(T(:,1)==3,:); %%%% For 3
%     
%     save('MNIST3_test','A');
%     
%     B = T(T(:,1)==8,:);    %%%% For 8
%     
%     save('MNIST8_test','B');
%     

    A = importdata('MNIST3.mat');
    
    [n1,d1] = size(A);  %%%% For 3
    
    B = importdata('MNIST8.mat');
    
    [n2,d2] = size(B);  %%%% For 8
    
    L3 = ones(n1,1);
    
    L8 = -1*ones(n2,1);
    
    A(:,1) = L3;  %%%%% For those whose labels are 3's, we have given it 1
    B(:,1) = L8;  %%%%% For those whose labels are 8's, we have given it -1
    
    Train = [A;B];
    
    [n3,d3] = size(Train);
    
    rng(seed);
    perm = randperm(n3);
    
    xtrain = Train(:,2:end);
    
    x_train = xtrain(perm,:)';
    
    %mie=min(min(x_train))
    
    mxe=max(max(x_train));
    
    %data.x_train =  normalize(x_train,'range');
    
    data.x_train = [x_train'./mxe ones(n3,1)]';
    
    
    
    fprintf('This is MNIST train 38 data with n=%d, d=%d\n',n3,d3-1);
    y_train = Train(:,1);
    
    data.y_train = y_train(perm)';
    
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %%%%%%%%%%%%%%%%              TEST                %%%%%%%%%%%%%%%%%%%%
   
    C = importdata('MNIST3_test.mat');
    
    [nt1,dt1] = size(C);  %%%% For 3
        
    D = importdata('MNIST8_test.mat');
    
    [nt2,dt2] = size(D);  %%%% For 8
    
    Lt3 = ones(nt1,1);
    
    Lt8 = -1*ones(nt2,1);
    
    C(:,1) = Lt3;  %%%%% For those whose labels are 3's, we have given it 1
    D(:,1) = Lt8;  %%%%% For those whose labels are 8's, we have given it -1
    
    Test = [C;D];
    
    [nt3,dt3] = size(Test);
    
    %fprintf('This is MNIST test data of 38 with n=%d, d=%d\n',nt3,dt3);
    rng('default');
    perm = randperm(nt3);
    
    xtest = Test(:,2:end);
    
    data.x_test = [xtest(perm,:) ones(nt3,1)]';
%     
%     maxe=max(max(x_test));
%     
%     %data.x_test = normalize(x_test,'range');
%     
%     data.x_test = [x_test./maxe;
%     
    fprintf('also, MNIST test 38 data with n=%d, d=%d.\n',nt3,dt3-1);
    
    y_test = Test(:,1);
    
    data.y_test = y_test(perm)';
    rng(seed);
    data.w_init = rand(d3,1);
end