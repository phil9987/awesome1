function [ vals, M, E ] = plot_mean_var( T, Y, p )
    vals = unique(T);
    size(vals)
    M = zeros(size(vals));
    E = zeros(size(vals));
    for i = 1:size(vals,1)
        v = vals(i);
        M(i) = mean(Y(T(:,1) == v));
        E(i) = std(Y(T(:,1) == v));
    end
    errorbar(vals,M,E, 'b-');
    hold on;
    P = polyfit(vals,M,p)
    plot(vals,polyval(P,vals));
    hold off;
end

