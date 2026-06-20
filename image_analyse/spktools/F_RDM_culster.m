% =========================================================================
% 计算并绘制神经信号的表征相异性矩阵 (Neural RDMs)
% 基于 1 - Pearson 相关系数 (1 - r)
% =========================================================================
clear; clc; close all;
% X_MVL=X_MVLall(:,16:190);
% X_ENTO=X_ENTOall(:,150:357);
% 
% X_MVL0 = reshape(X_MVL, 3, 24, size(X_MVL,2));
% X_MVL = squeeze(mean(X_MVL0,1));
% X_ENTO0 = reshape(X_ENTO, 3, 24, size(X_ENTO,2));
% X_ENTO = squeeze(mean(X_ENTO0,1));
%% 1. 全局图表高级设置
% 设置默认字体为 Arial (顶刊标配)
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontName', 'Arial');
%% 2. 载入 MATLAB 神经信号数据与标签
    % 载入标签
    % load(label_path, 'stimulus_labels');
data_path='F:\peng\Data_picture_fixed_Analyse_picture\fenxi\picture\data\';
    % 载入神经数据
dataname='acc1';
load(strcat(data_path,dataname));
fprintf('✅ 成功载入数据！X_ENTO 维度: [%d, %d], X_MVL 维度: [%d, %d]\n', ...
size(X_ENTO, 1), size(X_ENTO, 2), size(X_MVL, 1), size(X_MVL, 2));
%% 2. 载入 MATLAB 神经信号数据与标签
shapes = {'S1', 'S2', 'S3'};
textures = {'T1', 'T2'};
colors = {'Blue', 'Green', 'Red', 'Yellow'};
% 初始化用于存储 24 个标签的元胞数组
stimulus_labels = cell(24, 1);
% ★ 新增：初始化用于记录每个刺激具体属性的 24 元素数组，解决维度不兼容问题
shape_list = cell(24, 1);
texture_list = cell(24, 1);
color_list = cell(24, 1);

idx = 1;
% 利用三层嵌套循环，自动生成全排列标签及对应属性
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
% MATLAB 的 pdist(X, 'correlation') 默认计算的正是 1 - Pearson相关系数  cosine spearman
rdm_ento = squareform(pdist(X_ENTO, 'cosine'));
rdm_mvl  = squareform(pdist(X_MVL, 'cosine'));
%% 4. 绘制顶刊级双面板对比图
figure('Color', 'w', 'Position', [100, 100, 1300, 600]);

titles = {'ENTO RDM', 'MVL RDM'};
rdms = {rdm_ento, rdm_mvl};
num_stim = length(stimulus_labels);

for i = 1:2
    ax = subplot(1, 2, i);
    
    % 绘制热力图 (使用 imagesc 具有最高的灵活性)
    imagesc(rdms{i});
    colormap(ax, 'parula'); % 使用与 Seaborn 相同的 viridis 翠绿色系
    colorbar;
    
    % 设置标题与字体  , 'FontWeight', 'bold'
    title(titles{i}, 'FontSize', 20);
    
    % 设置坐标轴与标签
    axis square;
    set(ax, 'XTick', 1:num_stim, 'XTickLabel', stimulus_labels, ...
            'YTick', 1:num_stim, 'YTickLabel', stimulus_labels, ...
            'FontSize', 10, 'TickLength', [0 0],'TickLabelInterpreter', 'none');
    xtickangle(90); % X轴标签旋转90度
    % =======================================================
    % ★ 添加极其细微的白色网格线，划分每一个刺激格，提升高级感
    % =======================================================
    hold on;
    for k = 1.5 : (num_stim - 0.5)
        % 画垂直白线
        xline(k, 'Color', [1 1 1 0.6], 'LineWidth', 0.5);
        % 画水平白线
        yline(k, 'Color', [1 1 1 0.6], 'LineWidth', 0.5);
    end
    hold off;
    
    % 清理图表边界
    xlim([0.5, num_stim + 0.5]);
    ylim([0.5, num_stim + 0.5]);
end
%% 5. 保存极高精度的矢量图与数据矩阵
% 添加总标题
% sgtitle('Neural Representational Dissimilarity Matrices (1 - Pearson r)', 'FontSize', 22, 'FontWeight', 'bold');
% 保存矢量 PDF
fig_save_path = 'Neural_RDMs_MATLAB.pdf';
exportgraphics(gcf, strcat(data_path,fig_save_path), 'ContentType', 'vector');
fprintf('✅ 神经信号 RDM 绘制完成，高清图片已保存至: %s\n', fig_save_path);
% 保存 RDM 数据供后续的 MDS 降维和 RSA 使用
mat_save_path = 'Neural_RDMs.mat';
save(strcat(data_path,mat_save_path), 'rdm_ento', 'rdm_mvl');
fprintf('✅ ENTO 和 MVL 距离矩阵已成功打包存入: %s\n', mat_save_path);
%% 画Clustering聚类图,data_path
F_DMS(rdm_ento,rdm_mvl);
% % F_DMS1(rdm_ento,rdm_mvl,data_path);
% F_MDS_Silhouette_Compare(rdm_ento, rdm_mvl);
%% RDM 类内 vs. 类间 距离量化与可视化 (包含控制变量的纹理分析)
F_RDM_class_distance_EM(rdm_ento,rdm_mvl);
F_RDM_class_distance_ScatterEM(rdm_ento,rdm_mvl);