function quasi_Newton_test

clear;
clc;
close all;


%%%

     fprintf('###                  Select the Data set               ###\n'); 
    fprintf('##########################################################\n');
    fprintf('### Logistic REGRESSION                                ###\n');
    fprintf('###                                                    ###\n'); 
    fprintf('### 1) Arcene                                          ###\n');
    fprintf('### 2) Gisette                                         ###\n');
    fprintf('### 3) Madelon                                         ###\n'); 
    fprintf('### 4) RCV1_binary                                     ###\n'); 
    fprintf('### 5) COVtype                                         ###\n'); 
    fprintf('### 6) sido0                                           ###\n');
    fprintf('### 7) ALLAML                                          ###\n');
    fprintf('### 8) w8a                                             ###\n'); 
    fprintf('### 9) ijcnn1                                          ###\n'); 
    fprintf('### 10) MNIST38 (binary 3 = 1, 8 = -1                  ###\n'); 
    fprintf('### 11) Synthetic_logistic_data_generator              ###\n');
    fprintf('### 17) SMK_CAN                                        ###\n');
    fprintf('### 18) Prostate_GE                                    ###\n');
    fprintf('##########################################################\n');
    fprintf('###                                                    ###\n');

%%%
   
  in = input('Which data set you want to perfrom on?\nPlease enter the integer : ');
  
  
  Tr_s1 = []; Tr_s2 = []; Tr_s3 = [];
  C_s1 = []; C_s2 = []; C_s3 = [];
  Vc_s1 = []; Vc_s2 = []; Vc_s3 = [];
  Vl_s1 = []; Vl_s2 = []; Vl_s3 = [];
  otime_s1 =[]; otime_s2 =[]; otime_s3 =[];
  
 % seed = input('Give the random seed to select the random data samples:');
  
  for s=1:10
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
   
   %for q = 1:1
       
    
  data = data_input(in,seed);
  
  if in==7 %|| in==17 || in==18
      reg = 1;
      
  else
      reg = 0.01;
  end
  
  %problem = least_square(data.x_train,data.y_train,data.x_test,data.y_test, 0.0001);
  problem = logistic_regression1(data.x_train,data.y_train,data.x_test,data.y_test, reg);
    options.w_init = data.w_init;
    options.store_w = true;
    options.verbose = 1;
    options.stepsize_alg= @stepsize_alg;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    options.verbos = 1;
    
    
    options.f_opt = f_opt(in);
   
    fprintf('f_opt/Min_cost f(w*) = %.4e\n', options.f_opt);
    
    %options.g_opt = problem.full_grad(w_o);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    options.max_epoch=5000;
    c=1;
    step = {0.005*c, 1*10^-5*c, 0.0005*c};  
    alg = 'decay-2';
        
    options.step_alg = alg;
    options.method = 'normal';
    options.approximation = 'Nystrom';
    options.mem_size = 20;
    options.step_init = step{1};
    [w_s1,info_s1] = svrg_app(problem,options);
    
    fprintf('size of cost %dx%d\n',size(info_s1.cost));
    
     options.step_alg = alg;%'decay-2';
    options.step_init = step{2};
    options.approximation = 'Nystrom';
    options.method = 'normal';
    [w_s2,info_s2] = svrg_app(problem,options);
         
     options.step_alg = alg;%'decay-2';
     options.mem_size = 20;
    options.step_init = step{3};
    options.approximation = 'Nystrom';
    options.method = 'normal';
    [w_s3,info_s3] = svrg_app(problem,options); 
    
    
    
    Tr_s1 = [Tr_s1 info_s1.train_acc];
    Tr_s2 = [Tr_s2 info_s2.train_acc];
    Tr_s3 = [Tr_s3 info_s3.train_acc];
    
    
    Vl_s1 = [Vl_s1 info_s1.val_acc];
    Vl_s2 = [Vl_s2 info_s2.val_acc];
    Vl_s3 = [Vl_s3 info_s3.val_acc];
    
    
    
    C_s1 = [C_s1 info_s1.cost'];
    C_s2 = [C_s2 info_s2.cost'];
    C_s3 = [C_s3 info_s3.cost'];
    
    
    Vc_s1 = [Vc_s1 info_s1.test_cost'];
    Vc_s2 = [Vc_s2 info_s2.test_cost'];
    Vc_s3 = [Vc_s3 info_s3.test_cost'];
    
    
    otime_s1 = [otime_s1 info_s1.time'];
    otime_s2 = [otime_s2 info_s2.time'];
    otime_s3 = [otime_s3 info_s3.time'];
    
    
   end
%     
 

%%%%

    info_s1.train_ac = (Tr_s1);  %info_s1.std = mean(Tr_s1);
    info_s2.train_ac = (Tr_s2);  %info_s2.std = mean(Tr_s2);
    info_s3.train_ac = (Tr_s3);  %info_s3.std = mean(Tr_s3);
    
    
    info_s1.val_ac = (Vl_s1);  %info_s1.std = mean(Tr_s1);
    info_s2.val_ac = (Vl_s2);  %info_s2.std = mean(Tr_s2);
    info_s3.val_ac = (Vl_s3);  %info_s3.std = mean(Tr_s3);
    
    
    info_s1.ocost = C_s1;
    info_s2.ocost = C_s2;
    info_s3.ocost = C_s3;
    
    
    info_s1.vcost = Vc_s1;
    info_s2.vcost = Vc_s2;
    info_s3.vcost = Vc_s3;
    
    info_s1.otime=mean(otime_s1');
    info_s2.otime=mean(otime_s2');
    info_s3.otime=mean(otime_s3');
    
    S1=info_s1;
    S2=info_s2;
    S3=info_s3;
    
    if in==7
    Name = sprintf('ALLAML_%f.xlsx',0.1);
    elseif in==17
        Name = sprintf('SMK_CAN.xlsx');
    elseif in==18
        Name = sprintf('Prostate.xlsx');
    end
    Var = {'Epoch','Cost','Val_cost','Time','Train_acc','Val_acc'};
    
    T = table(S1.epoch',S1.ocost,S1.vcost,S1.otime',S1.train_ac,S1.val_ac,'VariableNames',Var);
    writetable(T,Name,'Sheet','SVRG-LBFGS-0.001');
    
    T2 = table(S2.epoch',S2.ocost,S2.vcost,S2.otime',S2.train_ac,S2.val_ac,'VariableNames',Var);
    writetable(T2,Name,'Sheet','Nystrom-5x1e-5');
    
    
    T3 = table(S3.epoch',S3.ocost,S3.vcost,S3.otime',S3.train_ac,S3.val_ac,'VariableNames',Var);
    writetable(T3,Name,'Sheet','SVRG-LBFGS-5x1e-5');
    
    %%%%
    
    algorithms_1 = {'SVRG-LBFGS','SVRG-Nystrom','SVRG-L-BFGS'};
    w_list_1 = {w_s1,w_s2,w_s3};
    info_list_1 = {info_s1,info_s2,info_s3};
    

%     
    C = [info_s1.cost(end), info_s2.cost(end),info_s3.cost(end)];
    
    [minv,d] = min(C);
    fprintf('Cost = %.4e\n',d);
    if d==1 
        m = 'SVRG -Newton';
    elseif d ==2
        m = 'SVRG -Nystrom';
    elseif d==3
        m = 'SVRG -LBFGS';
    end
    fprintf('Min value is %.22e attains by %s',minv,m);
    
    
   %end
    
    display_graph('epoch','accuracy', algorithms_1, w_list_1, info_list_1);
   % display_graph('time','optgap', algorithms_1, w_list_1, info_list_1);
   % display_graph('epoch','optgap', algorithms_1, w_list_1, info_list_1);    
    %display_graph('grad_calc_count','optgap',algorithms_1,w_list_1,info_list_1);
    display_graph('time','cost', algorithms_1, w_list_1, info_list_1);
    display_graph('epoch','cost', algorithms_1, w_list_1, info_list_1);
    display_graph('grad_calc_count','cost',algorithms_1,w_list_1,info_list_1);
    display_graph('epoch','Test_cost', algorithms_1, w_list_1, info_list_1);
   %display_graph('grad_calc_count','gnorm', algorithms_1, w_list_1, info_list_1);
   %costbb0.05=2.4563511861114180057086e+00 e100
  % 2.4563511861114162293518e+00 0.01 svrg 400

  
  %%%L-BFGS regularizer = 1; step size = 0.01 and memory = 20
  
end