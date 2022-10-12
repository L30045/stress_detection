% plot psd
tarCh = {'FZ','FCZ','CZ'};
load('\\hoarding/yuan/Documents/2019 stress in classroom/resting collection/raw/ch_power_raw.mat');
psd_lib_raw = ch_psd_lib;
% load('\\hoarding\yuan\Documents\2019 stress in classroom\resting collection\bp_refA_rmBadCh\subsetCh\ASR\rmEye\ch_power_proj.mat');
load('\\hoarding\yuan\Documents\2019 stress in classroom\resting collection\bp_refA_rmBadCh\ASR\ICrej\ch_power.mat');
psd_lib_icrej = ch_psd_lib;

figure

for t_i = 1:3
    plt_p = cell2mat(cellfun(@(x) x(ismember(chan30,tarCh{t_i}),:), psd_lib_raw','uniformoutput',0));
    raw_inc = plt_p(1:21,:);
    raw_norm = plt_p(22:end,:);
    plt_p = cell2mat(cellfun(@(x) x(ismember(chan30,tarCh{t_i}),:), psd_lib_icrej','uniformoutput',0));
    ic_inc = plt_p(1:21,:);
    ic_norm = plt_p(22:end,:);
    
    % significant
%     data = {raw_inc',raw_norm'};
%     [~, ~, pval] = statcond(data,'method','bootstrap','naccu',1000);
%     sig_raw = pval <= 0.05;
%     data = {ic_inc',ic_norm'};
%     [~, ~, pval] = statcond(data,'method','bootstrap','naccu',1000);
%     sig_ic = pval <= 0.05;
    sig_raw = ttest2(raw_inc,raw_norm);
    sig_ic = ttest2(ic_inc,ic_norm);
    sig_raw = logical(sig_raw);
    sig_ic = logical(sig_ic);
    
    subplot(2,3,t_i)
    xline(7,'k--','linewidth',2,'DisplayName','Feature range (7Hz)')
    hold on
    grid on
    xline(3,'k--','linewidth',2,'DisplayName','Feature range (3Hz)')
    h = shadedErrorBar(freqs, raw_inc, {@mean @std}, 'lineprops',{'linewidth',3,'color','r','DisplayName','Increase'});
    h = shadedErrorBar(freqs, raw_norm, {@mean @std}, 'lineprops',{'linewidth',3,'color','b','DisplayName','Normal'});
    plot(freqs(sig_raw),mean(raw_inc(:,sig_raw)),'kx','markersize',15,'linewidth',3)
    if t_i == 3
        legend(findobj(gca,'-regexp','DisplayName','[^'']'))
    end
    xlabel('Frequency (Hz)')
    ylabel('Power (\muV^2)')
    set(gca,'fontsize',20)
    set(gcf,'color','w')
    title(tarCh{t_i})
    xlim([0 20])
    
    subplot(2,3,t_i+3)
    xline(7,'k--','linewidth',2,'DisplayName','Feature Frequencies range (7Hz)')
    hold on
    grid on
    xline(3,'k--','linewidth',2,'DisplayName','Feature Frequencies range (3Hz)')
    h = shadedErrorBar(freqs, ic_inc, {@mean @std}, 'lineprops',{'linewidth',3,'color','r','DisplayName','Increase'});
    h = shadedErrorBar(freqs, ic_norm, {@mean @std}, 'lineprops',{'linewidth',3,'color','b','DisplayName','Normal'});
    plot(freqs(sig_ic),mean(ic_inc(:,sig_ic)),'kx','markersize',15,'linewidth',3)
%     legend(findobj(gca,'-regexp','DisplayName','[^'']'))
    xlabel('Frequency (Hz)')
    ylabel('Power (\muV^2)')
    set(gca,'fontsize',20)
    set(gcf,'color','w')
    xlim([0 20])
end