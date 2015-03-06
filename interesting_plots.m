plot_mean_var(T(:,2),Y,3)         %month
title('month')
figure
title('hour')
plot_mean_var(T(:,4),Y,15)         %hour
figure
title('W1')
plot_mean_var(round(W1*10)/10,Y,9)
figure
title('W3')
plot_mean_var(W3,Y,3)
figure
title('W6')
plot_mean_var(W6,Y,3)