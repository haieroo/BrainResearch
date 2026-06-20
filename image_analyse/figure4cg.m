% =========================================================================
% 双脑区细粒度纹理解码 (Within-Shape Fine-grained Texture Decoding)
% 包含：随神经元数量的动态解码曲线 + 最终混淆矩阵绘制
% =========================================================================
clc; close all;

% === 1. 参数与基础设置 ===
Target_Feature = 'Texture';   % 目标特征：纹理 (2分类)
nIter = 100;                   % 随机重抽样次数
nStim = 24;                   % 刺激总数
Standard_Trial_Num = 8;       % 统一截取最小试次
Total_Trials = nStim * Standard_Trial_Num; 
chance_level = 0.50;          % 纹理为2分类，随机几率 50%
%% 
Total_Neurons_ENTO = size(X_ENTO, 2);
Total_Neurons_MVL  = size(X_MVL, 2);

shape_labels   = kron([1, 2, 3]', ones(8, 1));        
texture_labels = repmat(kron([1, 2]', ones(4, 1)), 3, 1); 

Y_shape   = reshape(repmat(shape_labels', Standard_Trial_Num, 1), [], 1);
Y_texture = reshape(repmat(texture_labels', Standard_Trial_Num, 1), [], 1);

%% === 3. 执行动态抽样 (Neuron Dropping) 解码 ===
% 设定测试阶梯
max_neurons_to_test = min([Total_Neurons_ENTO, Total_Neurons_MVL]); 
% neuron_steps = [1, 2, 5, 10:10:127]; % 前期密一些，后期10个一加
neuron_steps = [1, 2, 4, 6:2:max_neurons_to_test]; 
num_steps = length(neuron_steps);

acc_ento = nan(num_steps, nIter);
acc_mvl  = nan(num_steps, nIter);
t = templateSVM('Standardize', true, 'KernelFunction', 'linear'); 

disp('🚀 开始执行控制形状的细粒度纹理解码...');

for i = 1:num_steps
    n = neuron_steps(i);
    for iter = 1:nIter
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        shape_acc_ento = zeros(3, 1);
        shape_acc_mvl  = zeros(3, 1);
        
        % 分别在 3 个形状内部独立解码纹理
        for s_idx = 1:3
            curr_idx = (Y_shape == s_idx);
            
            % KFold=5 进行 5折交叉验证
            mdl_ento = fitcecoc(sub_X_ento(curr_idx, :), Y_texture(curr_idx), 'Learners', t, 'KFold', 5);
            mdl_mvl  = fitcecoc(sub_X_mvl(curr_idx, :), Y_texture(curr_idx), 'Learners', t, 'KFold', 5);
            
            shape_acc_ento(s_idx) = 1 - kfoldLoss(mdl_ento);
            shape_acc_mvl(s_idx)  = 1 - kfoldLoss(mdl_mvl);
        end
        % 将3个形状的准确率求平均
        acc_ento(i, iter) = mean(shape_acc_ento);
        acc_mvl(i, iter)  = mean(shape_acc_mvl);
    end
    fprintf('完成抽样维度: %d 神经元...\n', n);
end
%% === 5.1 严谨版：各节点动态纹理置换检验 ( Permutation Test) ===
disp('🚀 正在执行严谨的各节点纹理置换检验 (这可能需要几分钟时间)...');
nPerm = 500; 
null_acc_ento = zeros(nPerm, num_steps);
null_acc_mvl  = zeros(nPerm, num_steps);

for p = 1:nPerm

    shuffled_Y = zeros(size(Y_texture));
    for s_idx = 1:3
        curr_idx = (Y_shape == s_idx);
        curr_Y = Y_texture(curr_idx);
        shuffled_Y(curr_idx) = curr_Y(randperm(length(curr_Y)));
    end
    
    for i = 1:num_steps
        n = neuron_steps(i);
        
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        shape_acc_ento_null = zeros(3, 1);
        shape_acc_mvl_null  = zeros(3, 1);
        
        for s_idx = 1:3
            curr_idx = (Y_shape == s_idx);
            mdl_ento_null = fitcecoc(sub_X_ento(curr_idx, :), shuffled_Y(curr_idx), 'Learners', t, 'KFold', 5);
            mdl_mvl_null  = fitcecoc(sub_X_mvl(curr_idx, :), shuffled_Y(curr_idx), 'Learners', t, 'KFold', 5);
            
            shape_acc_ento_null(s_idx) = 1 - kfoldLoss(mdl_ento_null);
            shape_acc_mvl_null(s_idx)  = 1 - kfoldLoss(mdl_mvl_null);
        end
        
        null_acc_ento(p, i) = mean(shape_acc_ento_null);
        null_acc_mvl(p, i)  = mean(shape_acc_mvl_null);
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

fprintf('✅ 严谨版各节点纹理置换检验全部完成！\n');
%% === 5. 绘制带有阴影误差带的动态解码曲线 ===

m_ento = mean(acc_ento, 2)' * 100; s_ento = (std(acc_ento, 0, 2)' / sqrt(nIter)) * 100;
m_mvl  = mean(acc_mvl, 2)' * 100;  s_mvl  = (std(acc_mvl, 0, 2)' / sqrt(nIter)) * 100;

figure('Position', [100, 100, 650, 500], 'Color', 'w');
set(gcf, 'DefaultAxesFontName', 'Arial');

hold on;

yline(chance_level * 100, '--k', sprintf('Chance (%.1f%%)', chance_level*100), ...
    'LineWidth', 2, 'FontSize', 12, 'LabelHorizontalAlignment', 'left');

% 绘制阴影带 (FaceAlpha 控制透明度)
X_patch = [neuron_steps, fliplr(neuron_steps)];
fill(X_patch, [m_ento + s_ento, fliplr(m_ento - s_ento)], [0.2 0.4 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill(X_patch, [m_mvl + s_mvl, fliplr(m_mvl - s_mvl)], [0.8 0.2 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% 绘制主线
p1 = plot(neuron_steps, m_ento, '-o', 'Color', [0.2 0.4 0.8], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.2 0.4 0.8]);
p2 = plot(neuron_steps, m_mvl, '-s', 'Color', [0.8 0.2 0.2], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.8 0.2 0.2]);

set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.5, 'FontSize', 14);
xlabel('Number of recording sites', 'FontSize', 18);
ylabel('Accuracy (%)',  'FontSize', 18);
title([strrep(Target_Feature, '_', ' '), ' population decoding'], 'FontSize', 20);

ylim([15, 80]); 
legend([p1, p2], {'ENTO', 'MVL'}, 'Location', 'northwest', 'FontSize', 14, 'Box', 'off');

% 1. 统一定义线段的高度 (放在画面最底部，Y=18 和 Y=16.5)
y_pos_ento = 18; 
y_pos_mvl  = 16.5;

% 2. 生成 NaN 断点
y_line_ento = repmat(y_pos_ento, 1, num_steps);
y_line_ento(p_vals_ento >= alpha_level) = NaN;

y_line_mvl = repmat(y_pos_mvl, 1, num_steps);
y_line_mvl(p_vals_mvl >= alpha_level) = NaN;


plot(neuron_steps, y_line_ento, '-', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(neuron_steps, y_line_mvl, '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'HandleVisibility', 'off');

text(max(neuron_steps)*0.6, 20, 'Horizontal lines: p < 0.05', ...
     'FontSize', 12, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');

