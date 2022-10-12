filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/';
typename = '_proj';
load([filepath,'ch_power',typename,'.mat']);

%%
targetCh = {'FZ','FCZ','CZ'};
freq_feature = 3:7;
ch_idx = ismember(chan30, targetCh);
features = cell2mat(cellfun(@(x) reshape(x(ch_idx,freq_feature+1),[],1),ch_psd_lib,'uniformoutput',0));

% train/ test split
X = features';
Y = [ones(21,1); zeros(71,1)];
% weight LDA according to number of samples
weights = [ones(21,1)*size(X,1)/21;...
    ones(71,1)*size(X,1)/71;];
clf_lda = fitcdiscr(X,Y,'weights',weights);
pred = predict(clf_lda,X);
figure
cmTrain = confusionchart(Y, pred);
cmTrain.ColumnSummary = 'column-normalized';
cmTrain.RowSummary = 'row-normalized';


coeffs = clf_lda.Coeffs(2,1).Linear;
const = clf_lda.Coeffs(2,1).Const;

save([filepath,'lda_result',typename,'.mat'],'features','const','coeffs','targetCh','freq_feature','clf_lda','sess_list','label_increase','label_normal')

%% sanity check FFT, Spectopo and ERSP
data = EEG.data(5,:);
[ersp,~,powbase,times,freqs_ersp,~,~,tfdata] = newtimef(EEG.data(ch_idx,:),EEG.pnts, [EEG.xmin EEG.xmax]*1000, EEG.srate,0,'baseline',NaN);
bins = discretize(freqs_ersp,0:51);
tf_cut = zeros(50,200);
for i = 1:50
    tf_cut(i,:) = sum(abs(tfdata(bins==i,:)/diff(times(1:2))).^2,1);
end
[spec,freqs_spec] = spectopo(EEG.data(ch_idx,:),0,EEG.srate,'plot','off');

fft_data = fft(data);
P2 = abs(fft_data/length(fft_data));
P1 = P2(1:length(fft_data)/2+1);
P1(2:end-1) = 2*P1(2:end-1);
plt_f = EEG.srate*(0:length(fft_data)/2)/length(fft_data);
fft_cut = zeros(1,50);
bins = discretize(plt_f,0.5:50.5);
for i = 1:50
    fft_cut(i) = sum(P1(bins==i).^2);
end

% check Parserval's theorem
fprintf('var(data) = sum(PSD)\n %f = %f\n',var(data),sum(10.^(spec/10)))
disp('Interpretation: average power in signal = sum of PSD across frequencies.')
fprintf('sum(data.^2) = mean(abs(fft_data).^2)\n %f = %f\n',sum(data.^2), mean(abs(fft_data).^2))
fprintf('sum(data.^2) = mean(P1*length(data).^2)/4 = sum(P2*length(data).^2)/length(data)\n %f = %f\n)',sum(data.^2), mean((P1*length(data)).^2)/4)
fprintf('sum(data.^2) = sum(fft_cut)*length(data)/2\n %f = %f\n',sum(data.^2), sum(fft_cut)*length(data)/2)
disp('Since fft_cut only includes power from 1 to 50 Hz, the summation will be smaller.')


%% window wise calculate PSD
sess_list = [label_increase',label_normal'];
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/';
savepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/LOO_win_lda/stepsize/';
targetCh = {'FZ','FCZ','CZ'};
f_feature = 3:7;
% winlen = 30; 
feature_lib = cell(1,length(sess_list));
spec_all_lib = cell(1,length(sess_list));

for winlen = [1,2,3,5,10,20,30,40,60,90,120,150,180,240,300] % sec; window length for calculating PSD
%         for overlap = [0.25, 0.5, 0.75] % overlapping rate 
    for stepsize_sec = [1,10,15]
        parfor j = 1:length(sess_list)
            i = sess_list(j);
            filename = sprintf('%d.set',i);
            EEG = pop_loadset([filepath,filename]);
            binsize = winlen*EEG.srate;
            stepsize = stepsize_sec*EEG.srate;
%             stepsize = floor(binsize*(1-overlap));
            if EEG.pnts < binsize
                nTrial = 1;
            else
                nTrial = ceil((EEG.pnts-binsize)/stepsize)+1;
            end
            chdata = zeros(length(targetCh),EEG.pnts);
            for ch_i = 1:length(targetCh)
                tarCh = targetCh{ch_i};
                ch_idx = ismember({EEG.chanlocs.labels},tarCh);
                chdata(ch_i,:) = EEG.data(ch_idx,:);
            end
            % calculate PSD
            tarf = 50;
            % remove last trial if it is shorter than 1 second
            if length((nTrial-1)*stepsize+1:min((nTrial-1)*stepsize+binsize,EEG.pnts))<EEG.srate
                nTrial = nTrial-1;
            end
            spec_lib = zeros(length(targetCh),tarf,nTrial); % channel by freq by trials;
            for t_i = 1:nTrial
                tmpdata = chdata(:,(t_i-1)*stepsize+1:min((t_i-1)*stepsize+binsize,EEG.pnts));
                [tmpspec,~] = spectopo(tmpdata,0,EEG.srate,'plot','off');
                spec_lib(:,:,t_i) = tmpspec(:,1:tarf);
            end

            feature_lib{j} = reshape(spec_lib(:,f_feature+1,:),15,nTrial); % feature by trial
            spec_all_lib{j} = spec_lib;
        end
        disp('Done')
    %     save([savepath,'feature_lib',typename,sprintf('_win%d_overlap%d',winlen,overlap*100),'.mat'],'feature_lib','spec_all_lib','targetCh','f_feature','sess_list','label_increase','label_normal','winlen');
        save([savepath,'feature_lib',typename,sprintf('_win%d_stepsec%d',winlen,stepsize_sec),'.mat'],'feature_lib','spec_all_lib','targetCh','f_feature','sess_list','label_increase','label_normal','winlen');
    end
end
disp('All Done.')

%%
bin_check = zeros(1,92);
pnt_check = zeros(1,92);
parfor j = 1:length(sess_list)
    i = sess_list(j);
    filename = sprintf('%d.set',i);
    EEG = pop_loadset([filepath,filename]);
    binsize = winlen*EEG.srate;
    bin_check(j) = binsize;
    pnt_check(j) = EEG.pnts;
end
disp('Done')

%% get weird subjects
thres_time = 10; % min
figure
bar(pnt_check/1000/60,'DisplayName','Rest duration');
hold on
grid on
abnormal_idx = pnt_check>thres_time*60*1000;
plot(find(abnormal_idx), pnt_check(abnormal_idx)/1000/60, 'x','DisplayName',num2str(sess_list(abnormal_idx)),'linewidth',3,'markersize',15);
legend
xlabel('Subject ID')
ylabel('Rest Duration')


%% LOO LDA classifier
% train LDA on the full-length recording without the testing recording
% savepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/LOO_win_lda/';
targetCh = {'FZ','FCZ','CZ'};
freq_feature = 3:7;
ch_idx = ismember(chan30, targetCh);
features = cell2mat(cellfun(@(x) reshape(x(ch_idx,freq_feature+1),[],1),ch_psd_lib,'uniformoutput',0));
lda_lib = cell(1,length(sess_list));
lda_const_lib = zeros(1,length(sess_list));
lda_coeffs_lib = zeros(size(features));

parfor j = 1:length(sess_list)
    if j <=21
        nb_inc = 20;
        nb_norm = 71;
    else
        nb_inc = 21;
        nb_norm = 70;
    end
    trainX = features;
    trainX(:,j) = [];
    % train/ test split
    X = trainX';
    Y = [ones(nb_inc,1); zeros(nb_norm,1)];
    % weight LDA according to number of samples
    weights = [ones(nb_inc,1)*size(X,1)/nb_inc;...
        ones(nb_norm,1)*size(X,1)/nb_norm;];
    clf_lda = fitcdiscr(X,Y,'weights',weights);
%     pred = predict(clf_lda,X);
%     figure
%     cmTrain = confusionchart(Y, pred);
%     cmTrain.ColumnSummary = 'column-normalized';
%     cmTrain.RowSummary = 'row-normalized';


    lda_coeffs_lib(:,j) = clf_lda.Coeffs(2,1).Linear;
    lda_const_lib(j) = clf_lda.Coeffs(2,1).Const;
    lda_lib{j} = clf_lda;
end
save([filepath,'loo_lda_lib',typename,'.mat'],'lda_lib','lda_const_lib','lda_coeffs_lib','targetCh','freq_feature','sess_list','label_increase','label_normal')

%% correct rate of Window-wise prediction
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/';
load([filepath,'loo_lda_lib_proj.mat']);
winlen_list = [1,2,3,5,10,20,30,40,60,90,120,150,180,240,300];
overlap_list = [0, 25, 50, 75];
win_acc_lib = zeros(length(winlen_list),length(overlap_list),length(sess_list));

for w_i = 1:length(winlen_list)
    winlen = winlen_list(w_i);
    for o_i = 1:length(overlap_list)
        overlap = overlap_list(o_i);
        stepsize = floor(winlen*(1-overlap/100));
        if overlap==0
            load([filepath,sprintf('feature_lib_proj_win%d.mat',winlen)]);
        else
            load([filepath,sprintf('LOO_win_lda/feature_lib_proj_win%d_overlap%d.mat',winlen, overlap)]);
        end
        nbwin = floor((300-winlen)/stepsize)+1;

        for j = 1:length(sess_list)
            loc_lda = lda_lib{j};
            test_X = feature_lib{j};
            test_X = test_X(:,1:min(nbwin, size(test_X,2)))';
            tmp_acc = predict(loc_lda,test_X);
            if j > 21
                tmp_acc = ~tmp_acc;
            end
            win_acc_lib(w_i,o_i,j) = sum(tmp_acc)/length(tmp_acc);
        end
    end
end
disp('Done')

%% Majority vote
o_i = 4; % overlap
overall_acc = mean(win_acc_lib>0.5,3)*100;
inc_acc = mean(win_acc_lib(:,:,1:21)>0.5,3)*100;
norm_acc = mean(win_acc_lib(:,:,22:end)>0.5,3)*100;
balanced_acc = (inc_acc+norm_acc)/2;

figure
plot(overall_acc(:,o_i),'g--','DisplayName','Overall Acc.','linewidth',3)
hold on
grid on
plot(inc_acc(:,o_i),'r-','DisplayName','Increase Acc.','linewidth',3)
plot(norm_acc(:,o_i),'b-','DisplayName','Normal Acc.','linewidth',3)
plot(balanced_acc(:,o_i),'k-','DisplayName','Balanced Acc.','linewidth',3)
xlabel('Window length (sec)')
ylabel('Acc.')
set(gca,'xTick',1:length(winlen_list))
% set(gca,'xTick',winlen_list)
set(gca,'xTickLabel',winlen_list)
legend 
set(gca,'fontsize',20)
title(sprintf('Majority Vote with different Window Size (overlap = %d%%)', overlap_list(o_i)))
set(gcf,'color','w')

%% Comparing effect of overlap
inc_acc = mean(win_acc_lib(:,:,1:21)>0.5,3)*100;
norm_acc = mean(win_acc_lib(:,:,22:end)>0.5,3)*100;
balanced_acc = (inc_acc+norm_acc)/2;
[max_acc, m_i] = max(balanced_acc);
cmap = cool(4);

figure
plt_f = 1:size(balanced_acc,1);
hold on
grid on
for p_i = 1:4
plot(balanced_acc(:,p_i),'-','Color',cmap(p_i,:),'DisplayName',sprintf('Overlap = %d%%, Max Acc. = %.2f%%', overlap_list(p_i), max_acc(p_i)),'linewidth',3)
plot(plt_f(m_i(p_i)),max_acc(p_i),'x','Color',cmap(p_i,:),'linewidth',3,'markersize',15)
end

xlabel('Window length (sec)')
ylabel('Balanced Acc.')
set(gca,'xTick',1:length(winlen_list))
set(gca,'xTickLabel',winlen_list)
legend 
set(gca,'fontsize',20)
title('Balanced Acc with different Window Size and Overlapping Rate')
set(gcf,'color','w')

%% correct rate of Window-wise prediction (full length)
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/';
load([filepath,'loo_lda_lib_proj.mat']);
winlen_list = [1,2,3,5,10,20,30,40,60,90,120,150,180,240,300];
win_acc_lib = zeros(length(winlen_list),length(sess_list));

for w_i = 1:length(winlen_list)
    winlen = winlen_list(w_i);
    load([filepath,sprintf('feature_lib_proj_win%d.mat',winlen)]);
%     nbwin = floor(300/winlen);
    
    for j = 1:length(sess_list)
        loc_lda = lda_lib{j};
        test_X = feature_lib{j};
%         test_X = test_X(:,1:min(nbwin, size(test_X,2)))';
        tmp_acc = predict(loc_lda,test_X');
        if j > 21
            tmp_acc = ~tmp_acc;
        end
        win_acc_lib(w_i,j) = sum(tmp_acc)/length(tmp_acc);
    end
end

%% correct rate of Window-wise prediction using LDA trainned on all recordings
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/';
load([filepath,'lda_result_proj.mat']);
winlen_list = [1,2,3,5,10,20,30,40,60,90,120,150,180,240,300];
win_acc_lib = zeros(length(winlen_list),length(sess_list));
loc_lda = clf_lda;

for w_i = 1:length(winlen_list)
    winlen = winlen_list(w_i);
    load([filepath,sprintf('feature_lib_proj_win%d.mat',winlen)]);
    nbwin = floor(300/winlen);
    test_X = feature_lib{j};
    test_X = test_X(:,1:min(nbwin, size(test_X,2)))';
    tmp_acc = predict(loc_lda,test_X);
    if j > 21
        tmp_acc = ~tmp_acc;
    end
    win_acc_lib(w_i,j) = sum(tmp_acc)/length(tmp_acc);
end


%% correct rate of Window-wise prediction with fixed stepsize
filepath = '/home/yuan/Documents/2019 stress in classroom/resting collection/bp_refA_rmBadCh/subsetCh/ASR/rmEye/';
load([filepath,'loo_lda_lib_proj.mat']);
winlen_list = [1,2,3,5,10,20,30,40,60,90,120,150,180,240,300];
stepsize_list = [1, 10, 15];
win_acc_lib = zeros(length(winlen_list),length(stepsize_list),length(sess_list));

for w_i = 1:length(winlen_list)
    winlen = winlen_list(w_i);
    for o_i = 1:length(stepsize_list)
        stepsize = stepsize_list(o_i);
        

        load([filepath,sprintf('LOO_win_lda/stepsize/feature_lib_proj_win%d_stepsec%d.mat',winlen, stepsize)]);
        
        nbwin = floor((300-winlen)/stepsize)+1;

        for j = 1:length(sess_list)
            loc_lda = lda_lib{j};
            test_X = feature_lib{j};
            test_X = test_X(:,1:min(nbwin, size(test_X,2)))';
            tmp_acc = predict(loc_lda,test_X);
            if j > 21
                tmp_acc = ~tmp_acc;
            end
            win_acc_lib(w_i,o_i,j) = sum(tmp_acc)/length(tmp_acc);
        end
    end
end
disp('Done')


% Majority vote
o_i = 3; % overlap
overall_acc = mean(win_acc_lib>0.5,3)*100;
inc_acc = mean(win_acc_lib(:,:,1:21)>0.5,3)*100;
norm_acc = mean(win_acc_lib(:,:,22:end)>0.5,3)*100;
balanced_acc = (inc_acc+norm_acc)/2;

figure
plot(overall_acc(:,o_i),'g--','DisplayName','Overall Acc.','linewidth',3)
hold on
grid on
plot(inc_acc(:,o_i),'r-','DisplayName','Increase Acc.','linewidth',3)
plot(norm_acc(:,o_i),'b-','DisplayName','Normal Acc.','linewidth',3)
plot(balanced_acc(:,o_i),'k-','DisplayName','Balanced Acc.','linewidth',3)
xlabel('Window length (sec)')
ylabel('Acc.')
set(gca,'xTick',1:length(winlen_list))
% set(gca,'xTick',winlen_list)
set(gca,'xTickLabel',winlen_list)
legend 
set(gca,'fontsize',20)
title(sprintf('Majority Vote with different Window Size (stepsize = %d sec)', stepsize_list(o_i)))
set(gcf,'color','w')

% Comparing effect of overlap
inc_acc = mean(win_acc_lib(:,:,1:21)>0.5,3)*100;
norm_acc = mean(win_acc_lib(:,:,22:end)>0.5,3)*100;
balanced_acc = (inc_acc+norm_acc)/2;
[max_acc, m_i] = max(balanced_acc);
cmap = cool(3);

figure
plt_f = 1:size(balanced_acc,1);
hold on
grid on
for p_i = 1:3
plot(balanced_acc(:,p_i),'-','Color',cmap(p_i,:),'DisplayName',sprintf('Stepsize = %d sec, Max Acc. = %.2f%%', stepsize_list(p_i), max_acc(p_i)),'linewidth',3)
plot(plt_f(m_i(p_i)),max_acc(p_i),'x','Color',cmap(p_i,:),'linewidth',3,'markersize',15)
end

xlabel('Window length (sec)')
ylabel('Balanced Acc.')
set(gca,'xTick',1:length(winlen_list))
set(gca,'xTickLabel',winlen_list)
legend 
set(gca,'fontsize',20)
title('Balanced Acc with different Window Size and Stepsize')
set(gcf,'color','w')