close; clear; clc;
fun = @(x) 1./(pi*(1+x.^2)).*exp(-1j*x);
int_ex = 1/exp(1);
tol = 1e-3;
J = 1e4;
[res, fc] = quadl(fun,-J,J, tol);
err = abs(res-int_ex)
err_r = abs(res-int_ex)/abs(int_ex)
