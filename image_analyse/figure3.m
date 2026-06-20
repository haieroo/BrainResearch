% =========================================================================
% 计算并绘制神经信号的表征相异性矩阵 (Neural RDMs)
clear; clc; close all;
%% 1. 全局图表高级设置
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontName', 'Arial');
%% 2. 载入 MATLAB 神经信号数据与标签
data_path='F:\peng\Data_picture_fixed_Analyse_picture\data\';
load(strcat(data_path));
size(X_ENTO, 1), size(X_ENTO, 2), size(X_MVL, 1), size(X_MVL, 2));
%% 2. 载入 MATLAB 神经信号数据与标签
shapes = {'S1', 'S2', 'S3'};
textures = {'T1', 'T2'};
colors = {'Blue', 'Green', 'Red', 'Yellow'};
stimulus_labels = cell(24, 1);
shape_list = cell(24, 1);
texture_list = cell(24, 1);
color_list = cell(24, 1);
idx = 1;
for s = 1:length(shapes)
    for t = 1:length(textures)
        for c = 1:length(colors)
            stimulus_labels{idx} = sprintf('%s_%s_%s', shapes{s}, textures{t}, colors{c});
            shape_list{idx} = shapes{s};       % 记录这 24 个 trial 对应的形状
            texture_list{idx} = textures{t};   % 记录这 24 个 trial 对应的纹理
            color_list{idx} = colors{c};       % 记录这 24 个 trial 对应的颜色
            idx = idx + 1;
        end
    end
end
%% 3. 计算 RDM (基于 1 - Pearson r)
fprintf('正在计算 ENTO 和 MVL 的表征相异性矩阵 (RDM)...\n');
rdm_ento = squareform(pdist(X_ENTO, 'correlation'));
rdm_mvl  = squareform(pdist(X_MVL, 'correlation'));
%% 4. 绘制对比图
figure('Color', 'w', 'Position', [100, 100, 1300, 600]);

titles = {'ENTO RDM', 'MVL RDM'};
rdms = {rdm_ento, rdm_mvl};
num_stim = length(stimulus_labels);

for i = 1:2
    ax = subplot(1, 2, i);
    imagesc(rdms{i});
    colormap(ax, 'parula'); 
    colorbar;
    title(titles{i}, 'FontSize', 20);
    axis square;
    set(ax, 'XTick', 1:num_stim, 'XTickLabel', stimulus_labels, ...
            'YTick', 1:num_stim, 'YTickLabel', stimulus_labels, ...
            'FontSize', 10, 'TickLength', [0 0],'TickLabelInterpreter', 'none');
    xtickangle(90);
    hold on;
    for k = 1.5 : (num_stim - 0.5)
        xline(k, 'Color', [1 1 1 0.6], 'LineWidth', 0.5);
        yline(k, 'Color', [1 1 1 0.6], 'LineWidth', 0.5);
    end
    hold off;
    xlim([0.5, num_stim + 0.5]);
    ylim([0.5, num_stim + 0.5]);
end
%% 画Clustering聚类图,data_path
F_DMS(rdm_ento,rdm_mvl);
%% RDM 类内 vs. 类间 距离量化与可视化 (包含控制变量的纹理分析)
F_RDM_class_distance_EM_perm(rdm_ento, rdm_mvl, 5000);