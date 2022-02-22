%% P1-3-b: nearest neighbor
x = 0:1:4;
y = [0,1,0,-1,0];
interpolate_point = 0.5:1:4;
figure
result = interp1(x,y,interpolate_point,'nearest');
plot(x,y,'o',interpolate_point,result,':.');
xlim([0 4]);
title('(Default) Linear Interpolation');


%% P1-3-b: perfect sinc interpolation
y = [0,1,0,-1,0];
u = 0:1:4; 
up = 0:0.5:4; 
x = linspace(1,length(y),length(y));
for i=1:length(up)
    yp(i) = sum(y.*sinc(up(i) - u));           
end
figure(1)
plot(u,y,'-ob'); hold on; plot(up,yp,'-*r');

%% close all;


old = [1,2,3;4,5,6;7,8,9];
old(:,:,2) = [1,2,3;4,5,6;7,8,9];
old(:,:,3) = [1,2,3;4,5,6;7,8,9];

new=[0,3,3;4,5,6;7,8,9];
new(:,:,2) = [1,2,3;4,5,6;7,8,9];
new(:,:,3) = [1,2,3;4,5,6;7,8,9];
h=3, w=3;

% method 1: immse function
result = immse(old, new);
% method 2: by hand    
MSE=sum(sum((old-new).^2)); 
MSE=sum(MSE(:))/(3*h*w);

