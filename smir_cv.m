function [SMG, CVLog]  = smir_cv( X, Yc, options)
%
% Function to choose the best parameters from the specified lists of
% candidates by cross validation for SMIR.
% 
% - X: input data matrix of d x n 
%   where d is dimension, and n is the sample size.
% 
% - Yc: output values in {1,...,c} of size 1 x l 
%   where c is the number of classes. 
%   l is the number of labeled points. So, the first l data points in
%   X are labeled, and the rest n-l points are unlabeled. 
%   It is essential that the values be in {1,...,c}. For example, in the case
%   of binary classification, {1,2} must be used i.e., not {-1, 1}.
% 
% - options: struct to specify options. 
%   Possible options are:
% 
%    - options.gaussianwidthlist to specify an array of Gaussian width
%    candidates 
%    - options.gammalist to specify an array of regularization parameter
%    candidates
%    - options.lambdafactorlist to specify lambdafactor candidates. 
%    Each lambdafactor in this list is used to calculate a lambda candidate
%    lambda = \gamma*c/n + lambdafactor
%    - options.fold to specify the number of folds in cross validation
% 
% See also demo_smir_msd_cv.m for a demonstration.
% 
if nargin < 3
    options = [];
end

if length(unique(Yc)) < 2
    error('%s: There should be at least 2 labeled points from 2 different classes.', mfilename);
end

smirfunc = @smir;


med = meddistance(X);
% Gaussian width candidates to try
gaussWidths = myProcessOptions(options, 'gaussianwidthlist', ...
    sqrt(2)*med*(2.^(-1:1)));

% array of gamma's to try
gammaList = myProcessOptions(options, 'gammalist', 10.^[-3, 1, 3] );

% array of lambdafactor's to try
lambdafactorList = myProcessOptions(options, 'lambdafactorlist', 10.^[-2, -4, -6, -8. -10]);

% Number of folds in cross validation used to select gamma, lambda and kernel
% function
fold = myProcessOptions(options, 'fold', 2 );

% seed for data stratification
seed = myProcessOptions(options, 'seed', 1);

% Begin CV
CVLog = struct();

% Grid search to choose best (kerFunc, gamma, lambda)
I = strapart(Yc, fold, seed);

% Matrix of classification errors (kerFunc's x gamma's x lambda's)
CErr = zeros(length(gaussWidths), length(gammaList), length(lambdafactorList));
for ki=1:length(gaussWidths)
    gaussianwidth = gaussWidths(ki);
    options.gaussianwidth= gaussianwidth;
    for gi=1:length(gammaList)
        gamma = gammaList(gi);
        options.gamma = gamma;
        
        for li=1:length(lambdafactorList)
            lambdafactor = lambdafactorList(li);
            options.lambdafactor = lambdafactor;
            
            foldErr = zeros(fold, 1);
            for fi=1:fold
                % Make test index
                teI = I(fi, :);
                % Make training index
                trI = ~teI;
                Ytr = Yc(trI);
                % Make training data
                
                % semi-supervised CV
                Xtr = [X(:,trI), X(:, (length(Yc)+1):end)];
                                
                % Make test data
                Yte = Yc(teI);

                SM = smirfunc(Xtr, Ytr, options);
    %             plot2dlabels(Xtr, Ytr, false);
                q = SM.q;
                eYh = q(X(:,teI));
                [V, eYc] = max(eYh, [],1); %estimated Yc
                % Compute classification error (0-1 loss)
                cerr = mean(Yte ~= eYc);
                foldErr(fi) = cerr;
            end
            mfe = mean(foldErr);
            CErr(ki, gi, li) = mfe;
            fprintf('%s: Gauss. width: %.3g, gamma: %.3g, lambf: %.3g => cerr: %.3g\n', ...
                func2str(smirfunc), gaussianwidth, gamma, lambdafactor, mfe);
        end
    end
end
CVLog.CVError = CErr;

% Choose the best kerfunc, gamma and epsilon
[minerr, ind] = min(CErr(:));
[b_ki, b_gi, b_li] = ind2sub(size(CErr), ind);

bestgaussianwidth = gaussWidths(b_ki);
bestgamma = gammaList(b_gi);
bestlambdafactor = lambdafactorList(b_li);

% Train model with full Y using the best chosen parameters
SMG.bestgaussianwidth = bestgaussianwidth;
SMG.bestgamma = bestgamma;
SMG.bestlambdafactor = bestlambdafactor;
SMG.mincverr = minerr;

options.gaussianwidth = bestgaussianwidth;
options.gamma = bestgamma;
options.lambdafactor = bestlambdafactor;

% Measure the training time for the best parameters
tic();
t0 = cputime();
SM = smirfunc(X, Yc, options);
SM.timecpu = cputime() - t0;
SM.timetictoc = toc();

SMG = dealstruct(SMG, SM);
SMG = orderfields(SMG);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

