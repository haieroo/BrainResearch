%% 多因素组内组间方差分析---判断偏好颜色偏好形状偏好纹理的神经元
clc;clear;close all
addpath(genpath('F:\peng\Data_picture_fixed_Analyse_picture\Analyse_chengxu\Spikeanalyse\spktools'));
%%
RootFile = 'F:\peng\Data_picture_fixed_Analyse_picture\Pre_data\all\picture1\';
[NAMEFile0,PATHFile0,FILEIndex0] = uigetfile('*.mat','Select the date file',RootFile);
FullFilename = strcat(PATHFile0,NAMEFile0);%读取数据路径  P_value_
load(FullFilename)
  window_on = [550,850];  %ENTO
%window_on = [570,870];  %MVL
window_off = [200,500]; %黑屏时间段-0.3-0

%%
bin_spike_anova=[];
for i= 1:length(bin_spikeall)
data=bin_spikeall(i).bin_spike_all;
respones_onall = sum(data(window_on(1):window_on(2)-1,:,:,:),1)*1000/(window_on(2)-window_on(1)); %计算窗口内的平均发放率
respones_offall = sum(data(window_off(1):window_off(2)-1,:,:,:),1)*1000/(window_off(2)-window_off(1)); %计算窗口内的平均发放率

[final_feature_matrix, final_eta2_matrix, all_S_width, all_d_prime,all_sparseness] = two_way_ANOVA(respones_onall, respones_offall);
[neuron_labels, class_id] = panduan_ANOVA(final_feature_matrix);

name=bin_spikeall(i).name;

[summary_table, per_neuron_table] = compute_feature_metrics_by_class( ...
    respones_onall, respones_offall, class_id);  
bin_spike_anova0=struct('bin_spike_all',data,'name',name,'class_id',class_id, ...
    'ANOVA_Pvalue',final_feature_matrix,'eta2_matrix', ...
    final_eta2_matrix,'summary_table',summary_table,'per_neuron_table',per_neuron_table);
bin_spike_anova=[bin_spike_anova,bin_spike_anova0];
end

%% 提取 ENTO MVL脑区稀疏度 eta2 矩阵 d'数据
bin_spike_anova_ENTO=bin_spike_anova;
bin_spike_anova_MVL=bin_spike_anova;

plot_all_eta2_sparseness_shared_axes(bin_spike_anova_ENTO, bin_spike_anova_MVL);