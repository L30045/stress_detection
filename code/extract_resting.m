% extract resting section from first sememster
%% file location and EEG channel labels
filepath = 'E:\NCTU_RWN-SFC\EEG+ECG\';
summary_list = readtable([filepath,'summary_NCTU_RWN-SFC.xls']);
summary_subj = lower(summary_list.Var6(1:110));

%% channl location
chan_NuAmps = readtable([filepath, '30ch_loc_NuAmps.xls']);
chan_SynAmps = readtable([filepath, '30ch_loc_SynAmps2.xls']);

%%
savepath = 'E:\NCTU_RWN-SFC\EEG+ECG\resting collection\first semester\';
if ~exist(savepath,'dir')
    mkdir(savepath)
end
error_EEG = [];
rmCh_lib = cell(size(summary_subj));

%% EEG preprocessing
for i = 1:114
    fprintf('>>>>>>> subj %d\n\n', i);
    loc_path = [filepath, 'first semester\', summary_list.Var4{i}];
    loc_f = [loc_path,filesep,summary_subj{i},'.set'];   
    savename = sprintf('%d.set',i);
    % channl location
    chan_path = summary_list.Var17(i);
    switch chan_path{:}
        case '30ch_loc_NuAmps.xls'
            chan_lib = chan_NuAmps;
        case '30ch_loc_SynAmps2.xls'
            chan_lib = chan_SynAmps;
    end
    tar_ch = cellfun(@(x) x(2:end-1),chan_lib.label,'uniformoutput',0);
    % load EEG
    EEG = pop_loadset(loc_f);
    % extract resting section
    rest_start = EEG.event(1).latency;
    rest_end = EEG.event(2).latency;
    
    if (rest_end - rest_start)/EEG.srate/60 < 1
        error_EEG = [error_EEG; i];
    else
        EEG = pop_select(EEG,'point',[rest_start, rest_end]);
        % preprocessing
        % remove irrelevant chs
        preserveCh = ismember({EEG.chanlocs.labels},tar_ch);
        EEG = pop_select(EEG, 'channel', {EEG.chanlocs(preserveCh).labels});
        % channel location lookup
        for ch_i = 1:length(EEG.chanlocs)
            t_i = ismember(tar_ch,EEG.chanlocs(ch_i).labels);
            EEG.chanlocs(ch_i).theta = chan_lib.theta(t_i);
            EEG.chanlocs(ch_i).radius= chan_lib.radius(t_i);
            EEG.chanlocs(ch_i).X = chan_lib.X(t_i);
            EEG.chanlocs(ch_i).Y = chan_lib.Y(t_i);
            EEG.chanlocs(ch_i).Z = chan_lib.Z(t_i);
            EEG.chanlocs(ch_i).sph_theta = chan_lib.sph_theta(t_i);
            EEG.chanlocs(ch_i).sph_phi = chan_lib.sph_phi(t_i);
            EEG.chanlocs(ch_i).sph_radius = chan_lib.sph_radius(t_i);
        end
        % bandpass
        EEG = pop_eegfiltnew(EEG,0.5,50);
        % reref
        ref_ch = [];
        EEG = pop_reref(EEG, ref_ch);
        % bad channel removal + ASR
        EEG = clean_rawdata(EEG,5,-1,0.7,4,20,-1);
        rmCh_lib{i} = setdiff(tar_ch,{EEG.chanlocs.labels});
        % save resting data
        pop_saveset(EEG,savename,savepath);
    end
end
save([savepath,'errorEEG.txt'], 'error_EEG', '-ASCII');
save([savepath,'rmCh_lib.mat'], 'rmCh_lib');
disp('Done')

