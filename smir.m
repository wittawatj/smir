function SM = smir(X, Yc, options)
%
% Squared-loss Mutual Information Regularization (SMIR)
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
% - options: struct to specify options. This argument is optional. 
%   If options are not specified, default values will be used.
%   Possible options are:
% 
%    - options.gaussianwidth to specify Gaussian width to use
%    - options.gamma to specify regularization parameter
%    - options.lambdafactor to specify lambdafactor used to calculate
%     lambda = \gamma*c/n + lambdafactor
%    - options.paramnormalize to specify normalization to use
%       - If 1, then A is normalized so that prior of Y is uniform . 
%       - If 2, A is normalized so that prior of Y follows the estimates from
%           the labeled data (default).
%       - If 3, no normalization.
% 
% Reference: Squared-loss Mutual Information Regularization: A Novel
% Information-theoretic Approach to Semi-supervised Learning (ICML2013)
% - Gang Niu (gang@sg.cs.titech.ac.jp), Tokyo Institute of Technology, 
% - Wittawat Jitkrittum (wittawatj@gmail.com), Tokyo Institute of Technology,
% - Bo Dai (bdai6@gatech.edu),  Georgia Institute of Technology,
% - Hirotaka Hachiya (hacchan@gmail.com), Tokyo Institute of Technology,
% - Masashi Sugiyama (sugi@cs.titech.ac.jp), Tokyo Institute of Technology
% 

if nargin < 3
    options = [];
end

c = length(unique(Yc));
if c <= 1
    error('Number of classes should be at least 2.');
end
n = size(X,2);
l = length(Yc);

% Change Yc to Y (indicator matrix) (c x l)
Y = sparse(Yc, 1:l, 1, c, l);

% Kernel function used to model class conditional probability 
% q(y|x;alpha) in the paper
gaussianWidth = myProcessOptions(options, 'gaussianwidth', ...
    sqrt(2)*meddistance(X)/3);

kerFunc = @(a,b)(kerGaussian(a,b, gaussianWidth)) ;
% gamma: regularization parameter
% trade-off between supervised and unsupervised part
gamma = myProcessOptions(options, 'gamma', 0.1);

% loss function. 
% 1 for mean absolution deviation (mad).
% 2 for mean square deviation (msd).
loss = myProcessOptions(options, 'loss', 2);

% function to parameterize lambda. f: struct(c,gamma,n),lambdafactor -> lambda
lambf_func = @lambf_plus; 

% regularization of A. 
% lambdafactor
if isfield(options, 'lambdafactor')
	lambdafactor = options.lambdafactor;
	lambda = lambf_func(struct('c', c, 'gamma', gamma, 'n', n), lambdafactor);
else
	lambda = c*gamma/n + 1e-10;
end

if lambda <= c*gamma/n
    error('%s: Necessary condition: lambda > c*gamma/n', mfilename);
end

% Laplace smoothing constant. Used for smoothing the probability so that it
% does not result in 0/0
lapconst = 1e-8;

% If 1, then A is normalized so that prior of Y is uniform 
% If 2, A is normalized so that prior of Y follows the estimates from
% the labeled data (default).
% If 3, no normalization.
paramnormalize = myProcessOptions(options, 'paramnormalize', 2);

% Begin SMIT-MSD

K = full(kerFunc(X,X));
[EV ED] = eig(K);
Dn12 = sparse(1:n, 1:n, sum(K,1).^(-0.5), n, n);
% KB = K(:, 1:l); % K*B (n x l)

Kp12 = EV*sparse(1:n, 1:n, real(diag(ED).^0.5), n, n)*EV'; %K^0.5; 
Kn12 = EV*sparse(1:n, 1:n, real(diag(ED).^(-0.5)), n, n)*EV'; %K^-0.5; 
Kp12B = Kp12(:, 1:l);

T = Dn12*Kp12B;
if loss == 1
    Q = lambda*n*l*eye(n) - gamma*l*c*Dn12*K*Dn12;
elseif loss == 2
    Q = n*(T*T') + lambda*n*l*eye(n) - gamma*l*c*Dn12*K*Dn12;
else
    error('%s: unknown loss function %d', mfilename, loss);
end
A = Q \ (n*T*Y');

%%% Normalize A %%%

if paramnormalize == 1
    % 1/c normalize (uniform class prior assumption)
    Beta = bsxfun(@rdivide, (n/c)*Kn12*Dn12*A, sum(Kp12*Dn12*A,1));
elseif paramnormalize ==2
    % normalize with prior estimated from Yc
    Pi = mean(Y,2); % c x 1
    Beta = bsxfun(@rdivide, n*Kn12*Dn12*A*diag(Pi), sum(Kp12*Dn12*A,1));
elseif paramnormalize == 3
    % no normalization
    Beta = Kn12*Dn12*A;
end

% Return results in struct
SM.A = A;
SM.Beta = Beta;

SM.Options = options;

% Construct the class conditional probability function handle
SM.q = @(newX)(q(newX,X, kerFunc, Beta, lapconst)); 
SM = orderfields(SM);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function Yh = q(newX, X, kerFunc, Beta, lapconst)
    c = size(Beta,2);    
    nK = kerFunc(X, newX);
    unYh = Beta'*nK; % c x ..
    munYh = max(unYh, 0);
    % Estimate Yhat (soft indicator matrix)
    Yh = bsxfun(@rdivide, munYh + lapconst, sum(munYh,1) + c*lapconst);
    Yh = full(Yh);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


function lambda = lambf_plus(S, lambdafactor)
%
% function to parameterize lambda. f: struct(c,gamma,n),lambdafactor -> lambda
%
c = S.c;
gamma = S.gamma;
n = S.n;
lambda = c*gamma/n + lambdafactor;

end
