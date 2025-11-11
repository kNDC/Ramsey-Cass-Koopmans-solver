% Solver for paths of a diffusion stochastic process of the standard form
% dx = mu(x)*dt + sigma(x)*dB. Uses a version of the relaxation algorithm.

function[out] = diffusion(mx,sx,mx_p1,sx_p1,x0,dt,dB)
    if(isequal(class(mx),'function_handle') && ...
            isequal(class(mx),'griddedInterpolant'))
        return;
    end
    if(isequal(class(sx),'function_handle') && ...
            isequal(class(sx),'griddedInterpolant'))
        return;
    end

    [T,N] = size(dB);
    dB_ = dB'; dB_ = dB_(:);

    % Adjustment and differentiation operators
    adjMx = kron(spdiags(ones(T,1),0,T,T+1), speye(N)); % Itô
    % adjMx = kron(spdiags(ones(T+1,2),0:1,T,T+1)/2, speye(N)); % Stratonovich
    % adjMx = kron(spdiags(ones(T,1),1,T,T+1), speye(N)); % Hänggi-Klimontovich

    d1Mx = kron(spdiags(repmat([-1 1],T+1,1), 0:1, T, T+1), speye(N));

    % Initialisation
    metric=1; i=0;
    x_ = kron(ones(T+1,1),x0);

    Js = [speye(N), sparse(N,N*T); d1Mx];
    Jd = sparse(N*(T+1),N*(T+1));

    while (1)
        i=i+1;

        err = [x_(1:N) - x0;
               d1Mx*x_ - mx(adjMx*x_)*dt - spdiag(dB_)*sx(adjMx*x_)];
        metric = max(abs(err));

        if (metric <= eps^0.4 || i > 1e3) break;
        end

        Jd = [sparse(N,N*(T+1));
              -spdiag(mx_p1(adjMx*x_))*adjMx*dt - ...
              spdiag(dB_)*spdiag(sx_p1(adjMx*x_))*adjMx];

        x_ = x_ - (Js+Jd)\err;
    end
    fprintf('%i diffusion trajectories constructed in %i iterations!\n', N, i);
    out = reshape(x_, N, T+1)';
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