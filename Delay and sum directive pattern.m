% x = audioread('');

% [Xnf,w,t] = spectrogram(x,window,noverlap,w)
[X,fs] = audioread('mic1.wav');
X=X';

for i=2:6
   [x,~] = audioread(['mic',num2str(i),'.wav']);
   X = [X; x']; 
end

X = X';

%%
theta_s = 2*pi/3;
r = 4.611e-2;
c = 343;

kj = [cos(theta_s) sin(theta_s)];

phis = 2*pi*[120 60 0 -60 -120 180]/(360);

mj = r*[cos(phis); sin(phis)];
m1 = repmat(mj(:,4), [1,6]);

deltaM = mj - m1;

delta = kj*deltaM/c;

delta_amostras = -round(delta*fs);

for i=1:6
    if(delta_amostras(i) ~= 0)
        if(delta_amostras(i) <= 0)
            X(:, i) = [zeros(abs(delta_amostras(i)),1); X(1:end-abs(delta_amostras(i)), i)];
        else
            X(:, i) = [X(delta_amostras(i):end-1, i); zeros(delta_amostras(i),1)];
        end
    end
end

final = mean(X,2);


%%

x1 = abs(fft(X(:,1)));
x_filt = abs(fft(final));

% x1_pw = sqrt(abs(pwelch(X(:,1), 10000)));
% x_filt_pw = sqrt(abs(pwelch(final, 1000)));

N = length(x1);
f = fs*[0:1:N-1]/N;

subplot(2,1,1)
plot(f,20*log10(fs*x1/N))
xlim([0,6000])

subplot(2,1,2)
plot(f,20*log10(fs*x_filt/N))
xlim([0,6000])

%%

f = 0;

w = exp(-j*2*pi*delta*f)/6;

theta_s = 2*pi*[0:1:1023]'/1024;
r = 4.611e-2;
c = 343;

kj = [cos(theta_s) sin(theta_s)];

mj = r*[cos(phis); sin(phis)];
m1 = repmat(mj(:,1), [1,6]);

deltaM2 = mj - m1;
delta2 = kj*deltaM2/c;

a_p = exp(-j*2*pi*delta2*f)';
power = abs(conj(w)*a_p).^2;

%totPower = sum(power)*(theta_s(2));
% power_dB = 20*log10(power/totPower);

figure
polarplot(theta_s,power) 



