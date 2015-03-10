function [P] = plot_mean_var( T, Y, p )
    m = mean(T);
    s = std(T);
    %T = (T-m)./s;
    vals = unique(T);
    size(vals);
    M = zeros(size(vals));
    E = zeros(size(vals));
    for i = 1:size(vals,1)
        v = vals(i);
        % ERROR TERROR
        M(i) = mean(Y(T(:,1) == v));
        E(i) = std(Y(T(:,1) == v));
    end
    errorbar(vals,M,E, 'b-');
    hold on;
    P = polyfit(T,Y,p);
    plot(vals,polyval(P,vals), 'r-');
    f = fit(vals,M,'fourier4');
    w = f.w
    2*pi/w
    %plot(vals,f(vals),'g-');
    hold off;
end

