function  demo()
% demonstration file for SGDLibrary.
%
% This file illustrates how to use this library in case of linear
% regression problem. This demonstrates SGD and SVRG algorithms.
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Oct. 24, 2016
% Modified by H.Kasai on Nov. 03, 2016


    clc;
    clear;
    close all;

    % generate synthetic data        
    % set number of dimensions
    %d = 3;
    % set number of samples    
    %n = 300;
    % generate data
      Tr_s1 = []; %Tr_s2 = []; Tr_s3 = [];
      C_s1 = []; %C_s2 = []; C_s3 = [];
      Vc_s1 = [];% Vc_s2 = []; Vc_s3 = [];
      Vl_s1 = []; %Vl_s2 = []; Vl_s3 = [];
      otime_s1 =[];% otime_s2 =[]; otime_s3 =[];
%or rho = [0 1000 100 10 1 0.1 0.01 0.001]

for reg=[1 0.1 0.01 0.001 0.0001 0.00001]
    
for step = [1 0.1 0.01 0.001 0.0001 0.00001]

for s=1:5
      if s==1
          seed=5.5;
      else
      seed=s-1;
      end
     isaninteger = @(x)isfinite(x) & x==floor(x);
   if isaninteger(seed)
       
   else
       seed = 'default';
   end
  
   fprintf('Loop number: %d \n',s);
   
    data =ALLAML(seed); %logistic_regression_data_generator(n, d);
    options.max_epoch=6500;    
    
    % define problem definitions
    problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
    
    
    % perform algorithms SGD and SVRG 
    options.w_init = data.w_init;   
    options.step_alg = 'decay-2';
    options.step_init = step; 
    options.verbose = 2;
    options.batch_size = 1;
    options.sub_mode='SVRG-LBFGS';
    %options.sub_mode= 'Lim-mem';
    
     
   [w_s1, info_s1] = slbfgs(problem, options);%,rho);
    
    
            Tr_s1 = [Tr_s1 info_s1.acc_tr];
            Vl_s1 = [Vl_s1 info_s1.acc_val];
            C_s1 = [C_s1 info_s1.cost'];
            Vc_s1 = [Vc_s1 info_s1.val_cost];
            otime_s1 = [otime_s1 info_s1.time'];


end

    info.epoch = info_s1.iter';    
    info.train_ac = (Tr_s1);  %info_s1.std = mean(Tr_s1);
    info.val_ac = (Vl_s1);  %info_s1.std = mean(Tr_s1);
    info.ocost = C_s1;
    info.vcost = Vc_s1;
    info.otime=mean(otime_s1')';
    
    S1=info;
    %Var = {'Epoch','Cost','Val_cost','Time','Train_acc','Val_acc'};
    %T = table(S1.iter',S1.ocost,S1.vcost,S1.otime,S1.train_ac,S1.val_ac,'VariableNames',Var);
    Name = sprintf('ALLAML_SLBFGS_S_%.1e_R_%.1e',options.step_init,reg);
    save(Name,'S1');%
    %if NRHO is zero then Nrho=frob_norm of Z;

end

end
    
    % display cost/optimality gap vs number of gradient evaluations
    %display_graph('time','cost', {'Other', 'Nystrom'}, {w_sgd, w_svrg}, {info_sgd, info_svrg});
    %I = info_svrg;

end


