% =========================================================================
% 双脑区标准单试次解码 (Single-trial Standard Decoding)
% 目标: 解码颜色 (Color)  形状 (Shape)或6种Morph
% 包含: 随神经元数量的动态解码曲线 + 全量细胞混淆矩阵 包括各种类型位点计算
% =========================================================================
clc; close all;

% === 1. 参数与基础设置 ===
% 🚀 目标特征开关：填 'Color' 或 'Shape'或 Morph
Target_Feature = 'Morph';   

nIter = 100;                   % 随机重抽样次数
nStim = 24;                   % 刺激总数
Standard_Trial_Num = 8;       % 统一截取最小试次
Total_Trials = nStim * Standard_Trial_Num; 

Total_Neurons_ENTO = size(X_ENTO, 2);
Total_Neurons_MVL  = size(X_MVL, 2);
%% === 3. 构建单试次标签矩阵 ===
shape_labels = kron([1, 2, 3]', ones(8, 1));        
color_labels = repmat([1, 2, 3, 4]', 6, 1);         

Y_shape = reshape(repmat(shape_labels', Standard_Trial_Num, 1), [], 1);
Y_color = reshape(repmat(color_labels', Standard_Trial_Num, 1), [], 1);

% 6个形态 主要是跨颜色泛化
morph_labels = kron((1:6)', ones(4, 1));  
Y_morph = reshape(repmat(morph_labels', Standard_Trial_Num, 1), [], 1);

if strcmp(Target_Feature, 'Color')
    Y_target = Y_color;
    chance_level = 0.25;
    class_names = {'Blue', 'Green', 'Red', 'Yellow'}; 
elseif strcmp(Target_Feature, 'Shape')
    Y_target = Y_shape;
    chance_level = 1/3;
    class_names = {'Shape 1', 'Shape 2', 'Shape 3'};
elseif strcmp(Target_Feature, 'Morph')
    Y_target = Y_morph;     % 目标：解码 6 种 morph
    chance_level = 1/6;     % 6 类 morph，随机水平 16.7%
    class_names = {'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'};
end

%% === 4. 执行 Neuron Dropping 动态解码 (5折交叉验证) ===
max_plot_neurons = min([Total_Neurons_ENTO, Total_Neurons_MVL]); 
neuron_steps = [1, 2, 5, 10:10:max_plot_neurons]; %群体

%neuron_steps = [1, 2, 4, 8:4:max_plot_neurons]; %主效应位点划分群体
num_steps = length(neuron_steps);

acc_ento = nan(num_steps, nIter);
acc_mvl  = nan(num_steps, nIter);

t = templateSVM('Standardize', true, 'KernelFunction', 'linear'); 

disp(['🚀 开始执行单试次标准解码: ', Target_Feature, ' ...']);

for i = 1:num_steps
    n = neuron_steps(i);
    for iter = 1:nIter
        % 随机抽样细胞
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        cv = cvpartition(Y_target, 'KFold', 5);

        mdl_ento = fitcecoc(sub_X_ento, Y_target, ...
            'Learners', t, ...
            'CVPartition', cv);
        
        mdl_mvl = fitcecoc(sub_X_mvl, Y_target, ...
            'Learners', t, ...
            'CVPartition', cv);
        % 记录准确率
        acc_ento(i, iter) = 1 - kfoldLoss(mdl_ento);
        acc_mvl(i, iter)  = 1 - kfoldLoss(mdl_mvl);
    end
    fprintf('完成抽样维度: %d 神经元...\n', n);
end

%% === 5.1 严谨版：各节点动态置换检验 (Permutation Test) ===
disp('🚀 正在执行严谨的各节点置换检验 (这可能需要几分钟时间)...');
nPerm = 500; 
null_acc_ento = zeros(nPerm, num_steps);
null_acc_mvl  = zeros(nPerm, num_steps);

% 先确定每个 stimulus 的目标标签
if strcmp(Target_Feature, 'Color')
    stim_target_labels = color_labels;  
elseif strcmp(Target_Feature, 'Shape')
    stim_target_labels = shape_labels;  
elseif strcmp(Target_Feature, 'Morph')
    stim_target_labels = morph_labels;  
end

for p = 1:nPerm
    shuffled_Y = Y_target(randperm(length(Y_target)));
    for i = 1:num_steps
        n = neuron_steps(i);    
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        cv = cvpartition(Y_target, 'KFold', 5);
        mdl_ento_null = fitcecoc(sub_X_ento, shuffled_Y, 'Learners', t, 'CVPartition', cv);
        mdl_mvl_null  = fitcecoc(sub_X_mvl,  shuffled_Y, 'Learners', t, 'CVPartition', cv);
        
        null_acc_ento(p, i) = 1 - kfoldLoss(mdl_ento_null);
        null_acc_mvl(p, i)  = 1 - kfoldLoss(mdl_mvl_null);
    end
    
    if mod(p, 10) == 0
        fprintf('   已完成 %d / %d 次置换...\n', p, nPerm);
    end
end

real_mean_ento = mean(acc_ento, 2)';
real_mean_mvl  = mean(acc_mvl, 2)';

p_vals_ento = ones(1, num_steps);
p_vals_mvl  = ones(1, num_steps);
alpha_level = 0.05; 

for i = 1:num_steps
    p_vals_ento(i) = (sum(null_acc_ento(:, i) >= real_mean_ento(i)) + 1) / (nPerm + 1);
    p_vals_mvl(i)  = (sum(null_acc_mvl(:, i) >= real_mean_mvl(i)) + 1) / (nPerm + 1);
end

fprintf('✅ 严谨版置换检验全部完成！\n');
%% === 5. 绘制随神经元数量的解码曲线 ===   
m_ento = mean(acc_ento, 2)' * 100; s_ento = (std(acc_ento, 0, 2)' / sqrt(nIter)) * 100;
m_mvl  = mean(acc_mvl, 2)' * 100;  s_mvl  = (std(acc_mvl, 0, 2)' / sqrt(nIter)) * 100;

figure('Position', [100, 100, 650, 500], 'Color', 'w');
set(gcf, 'DefaultAxesFontName', 'Arial');
hold on;
yline(chance_level * 100, '--k', sprintf('Chance (%.1f%%)', chance_level*100), 'LineWidth', 2, 'FontSize', 12, 'LabelHorizontalAlignment', 'left');

X_patch = [neuron_steps, fliplr(neuron_steps)];
fill(X_patch, [m_ento + s_ento, fliplr(m_ento - s_ento)], [0.2 0.4 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill(X_patch, [m_mvl + s_mvl, fliplr(m_mvl - s_mvl)], [0.8 0.2 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

p1 = plot(neuron_steps, m_ento, '-o', 'Color', [0.2 0.4 0.8], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.2 0.4 0.8]);
p2 = plot(neuron_steps, m_mvl, '-s', 'Color', [0.8 0.2 0.2], 'LineWidth',2, 'MarkerSize', 5, 'MarkerFaceColor', [0.8 0.2 0.2]);

set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.5, 'FontSize', 14);
xlabel('Number of recording sites', 'FontSize', 18);
ylabel('Accuracy (%)',  'FontSize', 18);
title([strrep(Target_Feature, '_', ' '), ' population decoding'], 'FontSize', 20);

ylim([15, 80]); 

% ylim([10, 70]); 
% 把图例挪到左上角 (NorthWest)，平衡画面重心
legend([p1, p2], {'ENTO', 'MVL'}, 'Location', 'northwest', 'FontSize', 14, 'Box', 'off');
% --- 绘制底部的显著性彩色水平线段 (替代满天星) ---
% 1. 定义线段的 Y 轴高度 (放在图表底部，X 轴上方一点)
y_pos_ento = 18; 
y_pos_mvl  = 16.5;

% y_pos_ento = 13; 
% y_pos_mvl  = 11.5;
% 2. 核心技巧 (NaN 断点法)：将 p 值大于 0.05 的节点设为 NaN
% 这样 plot 函数画线时，遇到不显著的地方就会自动断开
y_line_ento = repmat(y_pos_ento, 1, num_steps);
y_line_ento(p_vals_ento >= alpha_level) = NaN;

y_line_mvl = repmat(y_pos_mvl, 1, num_steps);
y_line_mvl(p_vals_mvl >= alpha_level) = NaN;

% 3. 画显著性水平粗线
plot(neuron_steps, y_line_ento, '-', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(neuron_steps, y_line_mvl, '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'HandleVisibility', 'off');

% 4. 把文字挪到左下角，靠近 Y 轴的地方 (X=2, Y=20)，避免和右边的线段末端挤在一起
text(max(neuron_steps)*0.6, 20, 'Horizontal lines: p < 0.05', ...
     'FontSize', 12, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');
