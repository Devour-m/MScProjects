function eta_new = eta_adaptation(eta_old,rate_difference,h)

if rate_difference >=0
    eta_new = (eta_old-1)*exp(-h*rate_difference) + 1;
else
    eta_new = (1-eta_old)*exp(h*rate_difference) + (2*eta_old-1);
end

end