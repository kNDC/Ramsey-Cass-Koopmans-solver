%% Description
% Solver for the optimal policy (consumption) function in the stochastic
% version of the Ramsey-Cass-Koopmans model with CIES (CRRA) preferences,
% a CES production technology and a gBM stochastic process for
% productivity. Solves for c by applying the Newton-Raphson algorithm to
% the Euler equation. The deterministic version of the model emerges as the
% particular case with zero volatility.

%% Parameter summary
% H/h preferences
% rho: discount factor
% theta: inverse CIES
% 
% Technology
% nu: capital share
% gamma: elasticity of substitution b/w K and L
% delta: capital depreciation rate
% 
% Growth rates
% g: intrinsic productivity growth rate
% n: population growth rate
% 
% Volatility
% sigma: productivity volatility rate
% 
% Productivity scale factors
% uK: capital
% uL:labour

%% Solver function
function[c, V, errCode] = RCK_solve(k, ...
                                 rho, theta, ...
                                 nu, gamma, delta, ...
                                 g, n, ...
                                 varargin)
    % Parsing optional parameters
    sigma = sparse(1,1); uK = 1; uL = 1;
    
    errCode = 0;
    errTol = eps^0.5;
    iMax = 5e2;

    if (nargin < 8) errCode = 1; return; end % too few arguments
    if (nargin >= 9) sigma = varargin{1}; end
    if (nargin >= 10) uK = varargin{2}; end
    if (nargin >= 11) uL = varargin{3}; end
    if (nargin >= 12) errTol = varargin{4}; end
    if (nargin >= 13) iMax = varargin{5}; end

    % Functions
    syms ks;
    if (gamma == 1) fs = (uK*ks).^nu*uL.^(1-nu);
    else fs = (nu*(uK*ks).^(1-1/gamma) + (1-nu)*uL.^(1-1/gamma)).^(gamma/(gamma-1));
    end
    
    f_p1s = diff(fs, ks);
    f_p2s = diff(f_p1s, ks);
    
    f = matlabFunction(fs, "vars", ks);
    f_p1 = matlabFunction(f_p1s, "vars", ks);
    f_p2 = matlabFunction(f_p2s, "vars", ks);
    
    if (theta == 1) u = @(c) log(c);
    else u = @(c) c.^(1-theta)/(1-theta);
    end

    % Finite difference derivative matrices
    Ik = length(k); dk = (k(Ik) - k(1))/Ik;
    d1Mx = spdiags(repmat([-1 sparse(1,1) 1],Ik,1),-1:1,Ik,Ik)/2;
    d1Mx(1,1)=-1; d1Mx(1,2)=1; d1Mx(Ik,Ik-1)=-1; d1Mx(Ik,Ik)=1;
    d1Mx = d1Mx/dk;
    
    d2Mx = spdiags(repmat([1 -2 1],Ik,1),-1:1,Ik,Ik);
    d2Mx(1,1) = -1; d2Mx(Ik,Ik) = -1;
    d2Mx = d2Mx/dk^2;

    % Main loop - Euler equation + Newton-Raphson (modified relaxation method)
    % Initialisation
    metric=1; i=0;
    c = k;
    
    % Static (c-independent) component of the Jacobian
    Js = theta*spdiag(f(k) - (delta+n+g - theta*sigma^2)*k)*d1Mx - ...
                 spdiag(f_p1(k)-delta - rho - theta*g + theta^2*sigma^2/2) + ...
                 sigma^2/2*spdiag(k.^2)*d1Mx^2;
    Jd = sparse(Ik,Ik);
    
    tic
    while 1
        i = i + 1;
    
        % Merit function/Error vector (from the HJB equation)
        err = theta*spdiag(f(k) - (delta+n+g - theta*sigma^2)*k - c)*d1Mx*c - ...
                  (f_p1(k)-delta - rho - theta*g + theta^2*sigma^2/2).*c - ...
                  theta*(theta+1)*sigma^2/2.*(k.*d1Mx*c).^2./c + ...
                  sigma^2/2*k.^2 .* (d1Mx^2*c);
        metric = max(abs(err));

        % Loop exit
        fprintf('Iteration = %d; max error = %f;\n', i, metric);
        if ((metric <= errTol) || (i >= iMax)) break;
        end
    
        % Dynamic (variable-dependent) component of the Jacobian
        Jd = -theta * (spdiag(d1Mx*c)+spdiag(c)*d1Mx + ...
                 (theta+1)*sigma^2/2*spdiag(k.^2) * spdiag(2*(d1Mx^2*c)./c - (d1Mx*c./c).^2));
    
        c = c - (Js+Jd)\err;
    end

    if (isnan(metric) || metric > errTol)
        errCode = 2;
    end
    
    if (metric > errTol)
        fprintf('--------\n');
        fprintf('The algorithm failed to converge in %d iterations!\n', i);
        toc
    else
        fprintf('--------\n');
        fprintf('The algorithm converged in %d iterations.\n', i);
        toc
    end

    % Extracting lifetime utility/value function V
    V = cumsum(c.^(-theta)*dk);
    err = (u(c) + (f(k) - (delta+n+g + sigma^2/2)*k + theta*sigma^2*k - c).*c.^(-theta) + ...
        (sigma^2*k.^2/2).*d1Mx^2*V)/(rho - n - (1-theta)*g - (1-theta)^2*sigma^2/2) - V;
    err = median(err);
    V = V + err;
end

%% Simple sparse diagonalisation
function[x] = spdiag(v)
    [rv,cv] = size(v);
    
    if (cv==1)
        x = spdiags(v,0,rv,rv);
        return;
    elseif (rv==1)
        x = spdiags(v',0,cv,cv);
        return;
    else x=v; return;
    end
end