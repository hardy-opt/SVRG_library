function [w, infos] = sgd(problem, in_options)
% Stochastic gradient descent (SGD) algorithm.
%
% Inputs:
%       problem     function (cost/grad/hess)
%       in_options  options
% Output:
%       w           solution of w
%       infos       information
%
% This file is part of SGDLibrary.
%
% Created by H.Kasai on Feb. 15, 2016
% Modified by H.Kasai on Mar. 25, 2018


    % set dimensions and samples
    d = problem.dim();
    n = problem.samples();  

    % set local options 
    local_options = [];
    
    % merge options
    options = mergeOptions(get_default_options(d), local_options);   
    options = mergeOptions(options, in_options); 
 

    %%%%%%%%%%%%%%%
    acc_tr = 0;
    acc_val = 0;
    var = 0;
    vr=0;
    val_cost=0;
    step=1;
    %%%%%%%%%%%%%%%%
    
    % initialize
    total_iter = 0;
    epoch = 0;
    grad_calc_count = 0;
    w = options.w_init;
    num_of_bachces = floor(n / options.batch_size); 

    % store first infos
    clear infos;    
    [infos, f_val, optgap] = store_infos(problem, w, options, [], epoch, grad_calc_count, 0);
    
    % display infos
    if options.verbose > 0
        fprintf('SGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
    end    

    % set start time
    start_time = tic();

    % main loop
%    while (optgap > options.tol_optgap) && (epoch < options.max_epoch)
    while  (epoch < options.max_epoch)

        % permute samples
        if options.permute_on
            perm_idx = randperm(n);
        else
            perm_idx = 1:n;
        end

        for j = 1 : num_of_bachces
            
            % update step-size
            step = options.stepsizefun(total_iter, options);
            
            % calculate gradient
            start_index = (j-1) * options.batch_size + 1;
            indice_j = perm_idx(start_index:start_index+options.batch_size-1);
            grad =  problem.grad(w, indice_j);

            % update w
            v= grad;
            w = w - step * grad;
            
            % proximal operator
            if ismethod(problem, 'prox')
                w = problem.prox(w, step);
            end  
            
            total_iter = total_iter + 1;
            
        end
        
        vr = norm(step*v-step*problem.grad(w,1:n))^2;

        % measure elapsed time
        elapsed_time = toc(start_time);
        
        % count gradient evaluations
        grad_calc_count = grad_calc_count + num_of_bachces * options.batch_size;        
        epoch = epoch + 1;

    
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        test_cost = problem.test_cost(w);
        val_cost = [val_cost test_cost];
        
        var = [var vr];
        
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
            fprintf('SGD: Epoch = %03d, cost = %.16e, optgap = %.4e\n', epoch, f_val, optgap);
        end

    end
    
    if optgap < options.tol_optgap
        fprintf('Optimality gap tolerance reached: tol_optgap = %g\n', options.tol_optgap);
    elseif epoch == options.max_epoch
        fprintf('Max epoch reached: max_epoch = %g\n', options.max_epoch);
    end


    infos.acc_tr = acc_tr';
    infos.acc_val = acc_val';
    infos.var = var';
    infos.val_cost = val_cost';
    
end