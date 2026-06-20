%% spike信号分析
clc;clear;close all
addpath(genpath('F:\peng\Data_picture_fixed_Analyse_picture\Analyse_chengxu\Spikeanalyse\spktools'));
%% 画出分类颜色形状纹理的 平均响应 以及raster曲线   单个神经元（或单个 MUA 记录位点）
RootFile = 'F:\peng\Data_picture_fixed_Analyse_picture\Pre_data\all\bin_all\picture1\';
[NAMEFile0,PATHFile0,FILEIndex0] = uigetfile('*.mat','Select the date file',RootFile);
FullFilename = strcat(PATHFile0,NAMEFile0);%读取数据路径
load(FullFilename)
window_on = [550,850];  %ENTO
% window_on = [570,870];  %MVL
window_off = [200,500]; %黑屏时间段-0.3-0
% 计算 Bar Chart 需要的净放电率 (Net MUA) % bin_spike_sct=time*trials*channel*stimulus
 for i=1:length(bin_spikeall)
     data=bin_spikeall(i).bin_spike_all;
     [color_data, shape_data,texture_data] = data_color_shape_texture(data);
     
     respones_onall = sum(data(window_on(1):window_on(2)-1,:,:,:),1)*1000/(window_on(2)-window_on(1)); %计算窗口内的平均发放率
     respones_offall = sum(data(window_off(1):window_off(2)-1,:,:,:),1)*1000/(window_off(2)-window_off(1)); %计算窗口内的平均发放率

    for ch=1:size(data,3)  
  
     resp=squeeze(respones_onall(:,:,ch,:)-respones_offall(:,:,ch,:));
     nTrials = size(resp, 1);  % 获取真实的试次数 (Trial number)
     net_mua = mean(resp, 1); % 计算跨试次的均值 (明确指定维度 1)
     bar_sem = std(resp, 0, 1) / sqrt(nTrials);

     F_color_bar1(ch, net_mua, bar_sem,color_data,window_off);  % 画出Bar Chart color PSTH  raster
     F_shape_bar1(ch, net_mua, bar_sem, shape_data, window_off);
     F_texture_bar1(ch, net_mua, bar_sem, texture_data, window_off);

     F_color_bar(ch, net_mua, bar_sem,color_data,window_off);  % 画出Bar Chart color PSTH raster
     F_shape_bar(ch, net_mua, bar_sem, shape_data, window_off);
     F_texture_bar(ch, net_mua, bar_sem, texture_data, window_off);
     end
 end