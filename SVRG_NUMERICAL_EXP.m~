function  SVRG_NUMERICAL_EXP()
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
    
    options.max_epoch=100;    

    
    % generate synthetic data        
    % set number of dimensions
    %d = 3;
    % set number of samples    
    %n = 300;
    % generate data

%or rho = [0 1000 100 10 1 0.1 0.01 0.001]
for m=1:6

for reg=[1 0.1 0.01 0.001 0.0001 0.00001]
    
for step = [10 5 1 0.1 0.01 0.001 0.0001]

      Tr_s1 = []; %Tr_s2 = []; Tr_s3 = [];
      C_s1 = []; %C_s2 = []; C_s3 = [];
      Vc_s1 = [];% Vc_s2 = []; Vc_s3 = [];
      Vl_s1 = []; %Vl_s2 = []; Vl_s3 = [];
      otime_s1 =[];% otime_s2 =[]; otime_s3 =[];
      opt_g = [];
    
    
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
  
   fprintf('Loop number: S=%d, Step=%d, reg=%d \n',s,step,reg);
   
    data = MNIST38(seed); %logistic_regression_data_generator(n, d);
   % data = MNIST38(seed);
    options.max_epoch=100;    
    
    % define problem definitions
    problem = logistic_regression1(data.x_train, data.y_train, data.x_test, data.y_test,reg); 
    
    if s==1
    w_opt = problem.calc_solution(200);
    f_opt = problem.cost(w_opt); 
    fprintf('f_opt: %.24e\n', f_opt);  
    else
        f_opt = optm;
    end
    options.f_opt = f_opt;
    % perform algorithms SGD and SVRG 
    options.w_init = data.w_init;   
    options.step_alg = 'fix';
    options.step_init = step; 
    options.verbose = 2;
    options.batch_size = 1;
    %options.sub_mode='SVRG-LBFGS';
    %options.sub_mode= 'Lim-mem';
    
   
    if m==1
        [w_s1, info_s1] = svrg(problem, options);
    elseif m==2
        [w_s1, info_s1] = svrgdh(problem, options);
    elseif m==3
        [w_s1, info_s1] = svrg_bb(problem, options);
    elseif m==4
        [w_s1, info_s1] = svrgbb(problem, options);
    elseif m==5
        [w_s1, info_s1] = svrgbbb(problem, options);
    elseif m==6
        [w_s1, info_s1] = svrg2nd(problem, options);
    end
    
    if info_s1.iter < options.max_epoch
        break;
    end
    
            Tr_s1 = [Tr_s1 info_s1.acc_tr];
            Vl_s1 = [Vl_s1 info_s1.acc_val];
            C_s1 = [C_s1 info_s1.cost'];
            Vc_s1 = [Vc_s1 info_s1.val_cost];
            otime_s1 = [otime_s1 info_s1.time'];
            opt_g = [opt_g info_s1.optgap'];
            
            size(Tr_s1)
            
     optm = f_opt;

end

    info.epoch = info_s1.iter';    
    info.train_ac = (Tr_s1);  %info_s1.std = mean(Tr_s1);
    info.val_ac = (Vl_s1);  %info_s1.std = mean(Tr_s1);
    info.ocost = C_s1;
    info.vcost = Vc_s1;
    info.opt_gap = opt_g;
    info.otime=mean(otime_s1,2);
    info.grad_count = (info_s1.grad_calc_count)';
    
    S1=info;
    %Var = {'Epoch','Cost','Val_cost','Time','Train_acc','Val_acc'};
    %T = table(S1.iter',S1.ocost,S1.vcost,S1.otime,S1.train_ac,S1.val_ac,'VariableNames',Var);
    
    if m==1
    Name = sprintf('/home/optima/Desktop/SVRG_library/MNIST38/SVRG/svrg_%.1e_R_%.1e.mat',options.step_init,reg);
    save(Name,'S1');%
    elseif m==2
    Name = sprintf('/home/optima/Desktop/SVRG_library/MNIST38/SVRGDH/svrgdh_%.1e_R_%.1e.mat',options.step_init,reg);
    save(Name,'S1');%
    elseif m==3
    Name = sprintf('/home/optima/Desktop/SVRG_library/MNIST38/SVRG_BB/svrg_bb_%.1e_R_%.1e.mat',options.step_init,reg);
    save(Name,'S1');% SVRG_BB with BB step size
    elseif m==4
    Name = sprintf('/home/optima/Desktop/SVRG_library/MNIST38/SVRGBB/svrgbb_%.1e_R_%.1e.mat',options.step_init,reg);
    save(Name,'S1');% SVRG_BB with second order info
    elseif m==5
    Name = sprintf('/home/optima/Desktop/SVRG_library/MNIST38/SVRG_BBB/svrg_bbb_%.1e_R_%.1e.mat',options.step_init,reg);
    save(Name,'S1');% SVRG_BB with BB step size and BB in 2nd order info.
    elseif m==6
    Name = sprintf('/home/optima/Desktop/SVRG_library/MNIST38/SVRG_2nd/svrg_2nd_%.1e_R_%.1e.mat',options.step_init,reg);
    save(Name,'S1');%
    end
    %if NRHO is zero then Nrho=frob_norm of Z;

end

end
    
    % display cost/optimality gap vs number of gradient evaluations
    %display_graph('time','cost', {'Other', 'Nystrom'}, {w_sgd, w_svrg}, {info_sgd, info_svrg});
    %I = info_svrg;
end
end


