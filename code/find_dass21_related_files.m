filepath = 'E:\NCTU_RWN-SFC\EEG+ECG\';
file_list = readtable([filepath(1:end-8), 'DSS+DASS21/DASS21/SFC_DASS21.xlsx']);
filepath_lib = cellfun(@(x) x(2:end-1),file_list.session,'uniformoutput',0);
subj_list = file_list.subject;
filepath_lib = filepath_lib(1:find(file_list.semester==1,1,'last'));
subj_list = subj_list(1:find(file_list.semester==1,1,'last'));
summary_list = readtable([filepath,'summary_NCTU_RWN-SFC.xls']);
summary_file = summary_list.Var4(1:110);
summary_subj = lower(summary_list.Var5(1:110));

%%
target_idx = zeros(size(subj_list));
for i = 1:length(subj_list)
    target_idx(i) = find(ismember(summary_file, filepath_lib{i})...
        & ismember(summary_subj, sprintf('s%02d',subj_list(i))));
end

% removed trial
rm_trials = find(diff(target_idx)>1)+1;