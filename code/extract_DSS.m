%% extract DSS information
addpath('../data/classroom/')
addpath(genpath('../data/classroom/DSS'))
dss_path = 'D:\Research\stress_detection\data\classroom\DSS\DSS\first semester\';
load('ch_power_raw.mat')
summary_table = readtable('firstsem_summary.csv');

%% extract date
subj_inc = summary_table.SubjectNumber(label_increase);
subj_norm = summary_table.SubjectNumber(label_normal);
date_inc = cellfun(@(x) x(regexp(x,'201*','once'):regexp(x,'201*','once')+7), summary_table.Dataset(label_increase),'uniformoutput',0);
date_norm= cellfun(@(x) x(regexp(x,'201*','once'):regexp(x,'201*','once')+7), summary_table.Dataset(label_normal),'uniformoutput',0);
dss_struct = struct('stress',[],'fatigue',[],'KSS',[],'Readiband',[],'PSD','');
dss_lib_inc = cell(1,length(subj_inc));
dss_lib_norm = cell(1,length(subj_norm));
field_name = fieldnames(dss_struct);

for d_i = 1:length(subj_inc)
    loc_path = [dss_path,sprintf('%02d/daily/',subj_inc(d_i))];
    filename = [loc_path,sprintf('day%s.txt',date_inc{d_i})];
    if exist(filename,'file')
        loc_dss = dss_struct;
        fid = fopen(filename);
        tline = fgetl(fid);
        k = 1;
        while ischar(tline) && k<=5
            if ~isempty(tline)
                if k==5
                    loc_dss.(field_name{k}) = tline(regexp(tline,':','once')+2:end);
                else
                    loc_dss.(field_name{k}) = str2double(tline(regexp(tline,'\d')));
                end
                k = k+1;
            end
            tline = fgetl(fid);
        end
        fclose(fid);
        dss_lib_inc{d_i} = loc_dss;
    end
end

for d_i = 1:length(subj_norm)
    loc_path = [dss_path,sprintf('%02d/daily/',subj_norm(d_i))];
    filename = [loc_path,sprintf('day%s.txt',date_norm{d_i})];
    if exist(filename,'file')
        loc_dss = dss_struct;
        fid = fopen(filename);
        tline = fgetl(fid);
        k = 1;
        while ischar(tline) && k<=5
            if ~isempty(tline)
                if k==5
                    loc_dss.(field_name{k}) = tline(regexp(tline,':','once')+2:end);
                else
                    loc_dss.(field_name{k}) = str2double(tline(regexp(tline,'\d')));
                end
                k = k+1;
            end
            tline = fgetl(fid);
        end
        fclose(fid);
        dss_lib_norm{d_i} = loc_dss;
    end
end

% print missing rate
fprintf('DSS missng rate - Increase group: %d%%\n',...
    round(sum(cellfun(@isempty, dss_lib_inc))/length(dss_lib_inc)*100));
fprintf('DSS missng rate - Normal group: %d%%\n',...
    round(sum(cellfun(@isempty, dss_lib_norm))/length(dss_lib_norm)*100));

save('dss_lib_inc.mat','dss_lib_inc','label_increase','subj_inc','date_inc');
save('dss_lib_norm.mat','dss_lib_norm','label_normal','subj_norm','date_norm');