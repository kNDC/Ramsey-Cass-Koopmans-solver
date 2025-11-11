clear all; close all; clc;

%% Parameters
% H/h preferences
rho = 0.03;
theta = 0.5;

% Technology
nu = 0.35;
gamma = 1; % Elasticity of substitution b/w K and L
delta = 0.04;

% Growth rates
g = 0.02;
n = 0.01;

% Volatility
sigma = 0.05;

% Shocks
uK = 1;
uL = 1;

%% Functions
syms ks;
if (gamma == 1) fs = (uK*ks).^nu*uL.^(1-nu);
else fs = (nu*(uK*ks).^(1-1/gamma) + (1-nu)*uL.^(1-1/gamma)).^(gamma/(gamma-1));
end

% Symbolic derivatives of the production function
f_p1s = diff(fs, ks);
f_p2s = diff(f_p1s, ks);

% Numeric derivatives
f = matlabFunction(fs, "vars", ks);
f_p1 = matlabFunction(f_p1s, "vars", ks);
f_p2 = matlabFunction(f_p2s, "vars", ks);

% Symbolic derivatives of the utility function
if (theta == 1) u = @(c) log(c);
else u = @(c) c.^(1-theta)/(1-theta);
end

% Numeric derivatives
u_p1 = @(c) c.^(-theta);
u_p2 = @(c) -theta*c.^(-theta-1);

%% Steady state
% SS for determ. Ramsey, GR for Solow
% Numerical solution using unidimensional Newton-Raphson

fprintf("Solving for the steady state...\n");

if (rho <= n + (1-theta)*g - (1-theta)^2*sigma^2/2)
    fprintf("The parameterisation admits no solution!\n\n")
    return
end

metric=1; i=0; kSS=1;
while (metric >= eps)
    err = f_p1(kSS)-delta - rho - theta*(g+sigma^2/2);
    kSS = kSS - err/f_p2(kSS);
    metric = abs(err); i=i+1;
end
ySS = f(kSS);
cSS = ySS - (delta+n+g - sigma^2/2)*kSS;
iSS = ySS - cSS;

fprintf("Deterministic values: kSS=%f; cSS=%f; iSS=%f;\n", kSS, cSS, iSS);

metric=1; i=0; kGR=1;
while (metric >= eps)
    kGR = kGR - (f_p1(kGR) - (delta+g+n - sigma^2/2))/f_p2(kGR);
    metric = abs(f_p1(kGR) - (delta+g+n - sigma^2/2)); i=i+1;
end
yGR = f(kGR);
cGR = yGR - (delta+n+g - sigma^2/2)*kGR;
iGR = yGR - cGR;

fprintf("Golden rule values: kSS=%f; cSS=%f; iSS=%f;\n\n", kGR, cGR, iGR);

%% Grids, differentiation operators
Ik = 2000; kmin = 1e-3*kSS; kmax = 4*kSS;
k = linspace(kmin, kmax, Ik)';
dk = (kmax-kmin)/Ik;

d1Mx = spdiags(repmat([-1 sparse(1, 1) 1], Ik, 1), -1:1, Ik, Ik)/2;
d1Mx(1, 1)=-1; d1Mx(1, 2)=1; d1Mx(Ik, Ik-1)=-1; d1Mx(Ik, Ik)=1;
d1Mx = d1Mx/dk;

d2Mx = spdiags(repmat([1 -2 1], Ik, 1), -1:1, Ik, Ik);
d2Mx(1, 1) = -1; d2Mx(Ik, Ik) = -1;
d2Mx = d2Mx/dk^2;

%% Solution - see notes in the solver
[cpD, VD, ~] = RCK_solver(k, rho, theta, nu, gamma, delta, g + sigma^2/2, n, 0); % deterministic case
fprintf("\n");

[cp, V, ~] = RCK_solver(k, rho, theta, nu, gamma, delta, g, n, sigma); % stochastic case
fprintf("\n");

% Investment policies
ipD = f(k) - cpD;
ip = f(k) - cp;

%% Optimal policy functions
figure; 
tiledlayout(2, 2, "Padding", "tight", "TileSpacing", "tight");

nexttile; hold on;
plot(k, cpD, "linewidth", 1.2, "color", [0.3 0 1]);
plot(k, cp, "linewidth", 1.2, "color", [1 0.3 0]);
xlim([0 kmax]); ylim([0 Inf]);
hold off;

lgd = legend("Deterministic", "Stochastic"); legend boxoff;
set(lgd, "interpreter", "latex", "Location", "south");

xlabel("$\tilde{k}$", "interpreter", "latex");
ylabel("$\tilde{c}^*\!\!\left(\tilde{k}\right)$", "interpreter", "latex");
title("Optimal consumption function~$\tilde{c}^*\!\!\left(\tilde{k}\right)$", ...
    "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(k, ipD, "linewidth", 1.2, "color", [0.3 0 1]);
plot(k, ip, "linewidth", 1.2, "color", [1 0.3 0]);
xlim([0 kmax]);
hold off;

lgd = legend("Deterministic", "Stochastic"); legend boxoff;
set(lgd, "interpreter", "latex", "Location", "south");

xlabel("$\tilde{k}$", "interpreter", "latex");
ylabel("$\tilde\imath^*\!\!\left(\tilde{k}\right)$", "interpreter", "latex");
title("Optimal investment function~$\tilde{i}^*\!\!\left(\tilde{k}\right)$", ...
    "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(k, VD, "linewidth", 1.2, "color", [0.3 0 1]);
plot(k, V, "linewidth", 1.2, "color", [1 0.3 0]);
xlim([0 kmax]); ylim([min([0 VD(1) V(1)]) Inf]);
hold off;

lgd = legend("Deterministic", "Stochastic"); legend boxoff;
set(lgd, "interpreter", "latex", "Location", "south");

xlabel("$\tilde{k}$", "interpreter", "latex");
ylabel("$V\!\!\left(\tilde{k}\right)$", "interpreter", "latex");
title("Value function~$V\!\!\left(\tilde{k}\right)$", ...
    "interpreter", "latex", "fontsize", 12);

%% Stationary distribution diagrams
% Capital
dtk = f(k) - (delta+n+g - sigma^2/2)*k - cp;
dBk = -sigma*k;
fk = ones(Ik, 1);

% Stationary distribution does not change over time => is zeroed by the
% respective Kolmogorov's forward operator Kf; in addition, the density's
% total mass is unity => (seemingly) overidentified linear system to pin 
% the density down.  The system becomes exactly identified, as Kf's 
% determinant is 0.
Kfk = [(d1Mx*k*dk)'; (spdiag(dtk)*d1Mx + spdiag(dBk.^2)/2*d1Mx^2)'];
err = Kfk * fk - [1; zeros(Ik, 1)];
fk = fk - Kfk\err;

mdk = med(k, fk); % median
Ek = k'*(fk.*gradient(k)); % mean
SDk = (k.^2'*(fk.*gradient(k)) - Ek^2)^0.5; % std deviation
k_zoomed = linspace(max(Ek-4*SDk, k(1)), Ek+4*SDk, 400); % k zoomed in

figure;
tiledlayout(2, 2, "Padding", "tight", "TileSpacing", "tight");

nexttile; hold on;
line(Ek*[1 1], [0 makima(k, fk, Ek)], "linestyle", "--", "color", [1 0.3 0]);
line(mdk*[1 1], [0 makima(k, fk, mdk)], "linestyle", ":", "color", [1 0.3 0]);
plot(k_zoomed, makima(k, fk, k_zoomed), "linewidth", 1.2, "color", [1 0.3 0]);
xlim(max(Ek + 4*SDk*[-1 1], k(1))); ylim([0 Inf]);
hold off;

xlabel("$\tilde{k}$", "interpreter", "latex");
ylabel("$f_{\tilde{k}}\!\left(\tilde{k}\right)$", "interpreter", "latex");
title("Capital per effective capita~$\tilde{k}$", ...
    "interpreter", "latex", "fontsize", 12);

% Output
[y, fy] = density(k, fk, f(k)); % density using the change of variable formula
mdy = med(y, fy);
Ey = y'*(fy.*gradient(y));
SDy = (y.^2'*(fy.*gradient(y)) - Ey^2)^0.5;
y_zoomed = linspace(max(Ey-4*SDy, y(1)), Ey+4*SDy, 400); % Zoomed in

nexttile; hold on;
line(Ey*[1 1], [0 makima(y, fy, Ey)], "linestyle", "--", "color", [1 0.3 0]);
line(mdy*[1 1], [0 makima(y, fy, mdy)], "linestyle", ":", "color", [1 0.3 0]);
plot(y_zoomed, makima(y, fy, y_zoomed), "linewidth", 1.2, "color", [1 0.3 0]);
xlim([y_zoomed(1) y_zoomed(end)]); ylim([0 Inf]);
hold off;

xlabel("$\tilde{y}$", "interpreter", "latex");
ylabel("$f_{\tilde{y}}\left(\tilde{y}\right)$", "interpreter", "latex");
title("Output per effective capita~$\tilde{y}$", ...
    "interpreter", "latex", "fontsize", 12);

% Consumption
[c, fc] = density(k, fk, cp);
mdc = med(c, fc);
Ec = c'*(fc.*gradient(c));
SDc = (c.^2'*(fc.*gradient(c)) - Ec^2)^0.5;
c_zoomed = linspace(max(Ec-4*SDc, cp(1)), Ec+4*SDc, 400);

nexttile; hold on;
line(Ec*[1 1], [0 makima(c, fc, Ec)], "linestyle", "--", "color", [1 0.3 0]);
line(mdc*[1 1], [0 makima(c, fc, mdc)], "linestyle", ":", "color", [1 0.3 0]);
plot(c_zoomed, makima(c, fc, c_zoomed), "linewidth", 1.2, "color", [1 0.3 0]);
xlim([c_zoomed(1) c_zoomed(end)]); ylim([0 Inf]);
hold off;

xlabel("$\tilde{c}$", "interpreter", "latex");
ylabel("$f_{\tilde{c}}\!\left(\tilde{c}\right)$", "interpreter", "latex");
title("Consumption per effective capita~$\tilde{c}$", ...
      "interpreter", "latex", "fontsize", 12);

% Investment
[i, fi] = density(k, fk, ip);

mdi = med(i, fi);
Ei = gradient(i)'*(i.*fi);
SDi = (abs(gradient(i))'*(i.^2.*fi) - Ei^2)^0.5;
iZ = linspace(Ei-4*SDi, Ei+4*SDi, 400);

nexttile; hold on;
line(Ei*[1 1], [0 makima(i, fi, Ei)], "linestyle", "--", "color", [1 0.3 0]);
line(mdi*[1 1], [0 makima(i, fi, mdi)], "linestyle", ":", "color", [1 0.3 0]);
plot(iZ, makima(i, fi, iZ), "linewidth", 1.2, "color", [1 0.3 0]);
xlim(Ei + 4*SDi*[-1 1]); ylim([0 Inf]);
hold off;

xlabel("$\tilde{i}$", "interpreter", "latex");
ylabel("$f_{\tilde{i}}\!\left(\tilde{i}\right)$", "interpreter", "latex");
title("Investment per effective capita~$\tilde{i}$", ...
    "interpreter", "latex", "fontsize", 12);

fprintf("Means: k: %f; y: %f; c: %f; i: %f;\n", ...
    Ek, Ey, Ec, Ei);
fprintf("SDs: k: %f; y: %f; c: %f; i: %f;\n", ...
    SDk, SDy, SDc, SDi);
% fprintf("Relative: k: %f; y: %f; c: %f; i: %f;\n", ...
%     sk/kSS, ...
%     sigma*f_p1(kSS)*kSS/f(kSS), ...
%     sigma*eSS, ...
%     sigma*abs(f_p1(kSS)*kSS-eSS*cSS)/iSS);
fprintf("\n");

clear k_zoomed y_zoomed c_zoomed iZ;

%% Time path diagrams
tmax = 1000; t = (0:1e-2:tmax)'; N = 10;
tmin = -floor(tmax/5);
dt = t(2)-t(1);

rng(123);
dBt = dt^0.5*randn(length(t)-1, N); Bt = [zeros(1, N); cumsum(dBt)];

m = @(x) (g+sigma^2/2)*x; m_p1 = @(x) g+sigma^2/2;
s = @(x) sigma*x; s_p1 = @(x) sigma;
At = diffusion(m, s, m_p1, s_p1, ones(N, 1), dt, dBt);

mk = griddedInterpolant(k, f(k) - (delta+n+g - sigma^2/2)*k - cp, "makima");
mk_p1 = griddedInterpolant(k, f_p1(k) - (delta+n+g - sigma^2/2) - d1Mx*cp, "makima");
sk = @(k) -sigma*k; sk_p1 = @(k) -sigma;

% Initial values of k are generated from its stationary distribution =>
% numeric generation using the inverse cumulative distribution with a
% slight padding Fk + eps^0.7*k; Fk = cumsum(fk.*(d1Mx*k)*dk);
kt = diffusion(mk, sk, mk_p1, sk_p1, ...
    makima(cumsum(fk.*(d1Mx*k)*dk) + eps^0.5*k, k, rand(N, 1)), dt, dBt);
yt = f(kt);
ct = makima(k, cp, kt);
it = makima(k, ip, kt);

figure;
hold on; histogram(kt, "Normalization", "pdf");
plot(k, fk, "linewidth", 2);
hold off;

figure;
tiledlayout(2, 3, "Padding", "tight", "TileSpacing", "tight");

nexttile; hold on;
plot(t, kt, "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{k}\!\left(t\right)$", "interpreter", "latex");
title("Capital path(s)~$\tilde{k}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, yt, "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{y}\!\left(t\right)$", "interpreter", "latex");
title("Output path(s)~$\tilde{y}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, ct, "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{c}\!\left(t\right)$", "interpreter", "latex");
title("Consumption path(s)~$\tilde{c}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, it, "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{i}\!\left(t\right)$", "interpreter", "latex");
title("Investment path(s)~$\tilde{i}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, sigma*Bt, "linewidth", 0.8, "color", [0.4 0 1]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\sigma \tilde{B}\!\left(t\right)$", "interpreter", "latex");
title("Shock path~$\sigma \tilde{B}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, At, "linewidth", 0.8, "color", [0.4 0 1]);
plot(t, exp((g+sigma^2/2)*t), "--", "linewidth", 1.2, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$A\!\left(t\right)$", "interpreter", "latex");
title("Productivity path~$A\!\left(t\right)$", ...
    "interpreter", "latex", "fontsize", 12);

%% Detrended paths
figure;
tiledlayout(2, 3, "Padding", "tight", "TileSpacing", "tight");

nexttile; hold on;
plot(t, log(kt.*exp(sigma*Bt)./mean(kt.*exp(sigma*Bt))), "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{k}\!\left(t\right)$", "interpreter", "latex");
title("Capital path(s)~$\tilde{k}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, log(yt.*exp(sigma*Bt)./mean(yt.*exp(sigma*Bt))), "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{y}\!\left(t\right)$", "interpreter", "latex");
title("Output path(s)~$\tilde{y}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, log(ct.*exp(sigma*Bt)./mean(ct.*exp(sigma*Bt))), "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{c}\!\left(t\right)$", "interpreter", "latex");
title("Consumption path(s)~$\tilde{c}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, log(it.*exp(sigma*Bt)./mean(it.*exp(sigma*Bt))), "linewidth", 0.8, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\tilde{i}\!\left(t\right)$", "interpreter", "latex");
title("Investment path(s)~$\tilde{k}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, sigma*Bt, "linewidth", 0.8, "color", [0.4 0 1]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$\sigma \tilde{B}\!\left(t\right)$", "interpreter", "latex");
title("Shock path(s)~$\sigma \tilde{B}\!\left(t\right)$", ...
      "interpreter", "latex", "fontsize", 12);

nexttile; hold on;
plot(t, At, "linewidth", 0.8, "color", [0.4 0 1]);
plot(t, exp((g+sigma^2/2)*t), "--", "linewidth", 1.2, "color", [1 0.3 0]);
xlim([0 tmax]); hold off;

xlabel("$t$", "interpreter", "latex");
ylabel("$A\!\left(t\right)$", "interpreter", "latex");
title("Productivity path(s)~$A\!\left(t\right)$", ...
    "interpreter", "latex", "fontsize", 12);

%% Simple sparse diagonalisation
function[x] = spdiag(v)
    [rv, cv] = size(v);
    
    if (cv == 1) x = spdiags(v, 0, rv, rv);
    elseif (rv == 1) x = spdiags(v', 0, cv, cv);
    else x = v;
    end
end

%% Median calculation based on the density
function[out] = med(x, fx)
    Fx = cumsum(gradient(x).*fx);
    mdIx = find(Fx > 0.5, 1);
    out = interp1(Fx(mdIx+(-1:0)), x(mdIx+(-1:0)), 0.5, "linear");
end

%% Numeric density of a function of a random variable
function[out_grid, out] = density(x, fx, y) % y = y(x), fx - distribution of X
    % Lengths of x, fx and y are assumed to be equal!

    out_grid = linspace(min(y), max(y), length(y))'; % grid
    y_p = gradient(y, x); % y'(x) for finding the indices of extrema
    extrema = zeros(length(y_p), 1); % speaking of which...
    
    i = 1;
    n_extrema = 0;
    while (i < length(y_p))
        if (y_p(i) * y_p(i + 1) < 0)
            n_extrema = n_extrema + 1;
            extrema(n_extrema) = i;

            % Captures that the central derivative can change its sign
            % before the extremum.
            if (y_p(i) > 0 && y(i + 1) > y(i)) 
                extrema(n_extrema) = extrema(n_extrema) + 1;
            elseif (y_p(i) < 0 && y(i + 1) < y(i))
                extrema(n_extrema) = extrema(n_extrema) + 1;
            end
        end
        i = i + 1;
    end

    extrema = extrema(1:n_extrema);
    clear i n_extrema;

    if (isempty(extrema))
        % No extrema? Calculate the density for y's whole range
        intervals = [1;
            length(y_p)];
    else
        % Using the indices, split y's range into monotone intervals
        intervals = [1;
            kron(speye(length(extrema)), [1;1]) * extrema + ...
                kron(ones(length(extrema),1), [0;1]);
            length(y_p)];
        intervals = full(intervals);
    end
    clear extrema;
    
    out = zeros(length(y), 1);
    
    i = 1;
    while (i < length(intervals))
        % Build the inverse for each interval
        interval = (intervals(i):intervals(i+1))';

        % Numeric inverse of y = y(x), extrapolation is shut off
        y_inv = interp1(y(interval), x(interval), out_grid, "makima", NaN);
        
        % Standard change of variable formula (numeric flavour)
        out_part = makima(x, fx, y_inv) .* ...
            abs(gradient(y_inv, out_grid));
        out_part(isnan(out_part)) = 0;
        out = out + out_part;

        i = i + 2;
    end
end