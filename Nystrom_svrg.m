function [w, infos] = Nystrom_svrg(problem, in_options,rho)
% Stochastic limited-memory quasi-newton methods (Stochastic L-BFGS) algorithms.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% References:
%       sub_mode:   SQN:
%                   Byrd, R. H., Hansen, S. L., Nocedal, J., & Singer, Y. 
%                   "A stochastic quasi-Newton method for large-scale optimization," 
%                   SIAM Journal on Optimization, 26(2), 1008-1031, 2016.
%
%       sub_mode:   SVRG-SQN:
%                   Philipp Moritz, Robert Nishihara, Michael I. Jordan,
%                   "A Linearly-Convergent Stochastic L-BFGS Algorithm," 
%                   Artificial Intelligence and Statistics (AISTATS), 2016.
%
%       sub_mode:   SVRG LBFGS:
%                   R. Kolte, M. Erdogdu and A. Ozgur, 
%                   "Accelerating SVRG via second-order information," 
%                   OPT2015, 2015.
%
%                   
% Created by H.Kasai on Oct. 15, 2016
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();
    

    % set local options 
    local_options.sub_mode = 'SQN';  % SQN or SVRG-SQN or SVRG-LBFGS
    local_options.mem_size = 20;    
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options);      
    


    
    %%%%%%%%%%%%%%%
    acc_tr = 0;
    acc_val = 0;
    %var = 0;
    %vr=0;
    val_cost=0;
    %step=1;
    %%%%%%%%%%%%%%%%


    % set paramters
    if options.batch_size > n
        options.batch_size = n;
    end   
    
    if ~isfield(in_options, 'batch_hess_size')
        options.batch_hess_size = 20 * options.batch_size;
    end    

    if options.batch_hess_size > n
        options.batch_hess_size = n;
    end    
    
        
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size)*2;     
        

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);  
    
    % set start time
    start_time = tic();
    
    % display infos
    if options.verbose > 0
        fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', options.sub_mode, epoch, f_val, optgap);
    end     

    % main loop
    %while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    while (epoch < options.max_epoch)
        % permute samples
      %  if options.permute_on
      %      perm_idx = randperm(n);
      %  else
            perm_idx = [1:n, 1:n];
      %  end

    %    if strcmp(options.sub_mode, 'SVRG-SQN') || strcmp(options.sub_mode, 'SVRG-LBFGS')
            % compute full gradient
            %full_grad_new = problem.grad(w,1:n);
            full_grad_new = problem.full_grad(w);
            % count gradient evaluations
            grad_calc_count = grad_calc_count + n; 
     %   end          


%        if strcmp(options.sub_mode, 'SVRG-SQN') || strcmp(options.sub_mode, 'SVRG-LBFGS')
            % store w for SVRG
            w0 = w;
            full_grad = full_grad_new;
%        end          
      
        
        for j = 1 : num_of_bachces

               if j==1 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     GN = abs(full_grad);
%                     set = zeros(1,10);
%                       for p = 1:10
%                      [a,b] = min(GN);
%         
%                         set(p) = b;    %[set b];
%         
%                        GN(b) = GN(b) + Inf;
%                      % fprintf('max(G)=%d, indices=%d\n',a,b);
%                        GN1 = GN;
%                         GN = GN1;
%         
%                  
                    rng(j);
                    set = randperm(d,10);
                   % E = set(1:7);
                   % F = set(1:5);
                   
                    %sam=randperm(n,100);

                   % fprintf('Size of G = %d\n',length(G));
                    [aha,fn1,apta] = problem.app_hess(w0,1:n,set,0);
                

                lam = 1e-3;%norm(full_grad); % Norm of full_gradient
                lk = length(set); % k: colums
               % HI = inv(H); % Hessian Inverse
                
                if rho==0 %;%Nystrom regularized
                    rho = norm(aha,'fro');
                    nfg = 1/rho;
                else
                nfg = 1/rho;
                end
                Ey = eye(lk);
                M = aha*inv(Ey+nfg*(aha'*aha));
                end
            % update step-size
            step = options.stepsizefun(total_iter, options);                
         
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad = problem.grad(w, indice_j);
            
            % calculate variance reduced gradient
 %           if strcmp(options.sub_mode, 'SVRG-SQN') || strcmp(options.sub_mode, 'SVRG-LBFGS')
                grad_w0 = problem.grad(w0,indice_j);
                grad = full_grad + grad - grad_w0;    
 %           end 
            

                     %NI =  nfg*eye(d) - (nfg)*aha/(Ey+nfg*(aha'*aha))*aha';
                     vect = aha'*grad;
                    
                     Mv = M*vect;
                     NI = nfg*(grad - nfg*Mv); 
                     v = step*NI;

                    w = w - v;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end              
           
                       
            
            total_iter = total_iter + 1;
        end
        
             
        %vr = norm(step*v-step*problem.grad(w,1:n))^2;

        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + j * options.batch_size;        
        epoch = epoch + 1;



          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        test_cost = problem.test_cost(w);
        val_cost = [val_cost test_cost];
        
       % var = [var vr];
        
        p_tr = problem.prediction(w,'Tr');
        acctr = problem.accuracy(p_tr,'Tr');
        acc_tr = [acc_tr acctr]; 
        
        
        p_vl = problem.prediction(w,'Vl');
        accvl = problem.accuracy(p_vl,'Vl');
        acc_val = [acc_val accvl]; 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        
        % store infos
        [infos, f_val, optgap] = store_infos(problem, w, options, infos, epoch, grad_calc_count, elapsed_time);            

        % display infos
        if options.verbose > 0
            fprintf('%s: Epoch = %03d, cost = %.16e, optgap = %.4e\n', 'Nystrom', epoch, f_val, optgap);
        end
    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epochr = %g\n', options.max_epoch);
    end
      
    infos.acc_tr = acc_tr';
    infos.acc_val = acc_val';
    %infos.var = var';
    infos.val_cost = val_cost';
    
end

