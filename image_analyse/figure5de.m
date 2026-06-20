% =========================================================================
% 双脑区全集群跨条件泛化解码 (CCGP) - 嵌套模式独立脚本
% 任务：纹理跨颜色泛化
% 逻辑：在同一形状内部，用 3 种颜色的纹理训练，预测未见过的第 4 种颜色的纹理。
%       分别在 3 个形状下计算，最终求取平均泛化准确率。 
% =========================================================================
clc; close all;

% === 1. 参数与基础设置 ===
Target_Feature = 'Decoding texture generalizing across color';   
nIter = 100;                   % 随机重抽样次数 
nStim = 24;                   % 刺激总数
Standard_Trial_Num = 8;       % 统一截取最小试次 
Total_Trials = nStim * Standard_Trial_Num; 
chance_level = 0.50;          % 纹理2分类，Chance = 50%

Total_Neurons_ENTO = size(X_ENTO, 2);
Total_Neurons_MVL  = size(X_MVL, 2);
fprintf('✅ 数据就绪: ENTO 有 %d 个细胞, MVL 有 %d 个细胞。\n', Total_Neurons_ENTO, Total_Neurons_MVL);

%% === 3. 构建对应的试次标签 ===
shape_labels   = kron([1, 2, 3]', ones(8, 1));        
texture_labels = repmat(kron([1, 2]', ones(4, 1)), 3, 1); 
color_labels   = repmat([1, 2, 3, 4]', 6, 1);         

Y_shape   = reshape(repmat(shape_labels', Standard_Trial_Num, 1), [], 1);
Y_texture = reshape(repmat(texture_labels', Standard_Trial_Num, 1), [], 1);
Y_color   = reshape(repmat(color_labels', Standard_Trial_Num, 1), [], 1);

%% === 4. 执行动态抽样 CCGP 解码 (嵌套逻辑) ===
max_plot_neurons = min([127, Total_Neurons_ENTO, Total_Neurons_MVL]); 
neuron_steps = [1, 2, 5, 10:10:max_plot_neurons]; 
num_steps = length(neuron_steps);

acc_ento = nan(num_steps, nIter);
acc_mvl  = nan(num_steps, nIter);

t = templateSVM('Standardize', true, 'KernelFunction', 'linear'); 
disp(['🚀 开始执行 嵌套模式 的 CCGP 动态解码: ', Target_Feature, ' ...']);

for i = 1:num_steps
    n = neuron_steps(i);
    for iter = 1:nIter
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        shape_acc_ento = zeros(3, 1);
        shape_acc_mvl  = zeros(3, 1);
        
        % 🌟 核心嵌套：在 3 个形状内部独立执行跨颜色泛化
        for s_idx = 1:3
            curr_shape_idx = (Y_shape == s_idx);
            
            folds_acc_ento = zeros(4, 1); % 4 种颜色轮换
            folds_acc_mvl  = zeros(4, 1);
            
            for c_idx = 1:4
                % 训练集：当前形状下，非目标颜色的试次
                train_idx = (Y_color ~= c_idx) & curr_shape_idx; 
                % 测试集：当前形状下，目标颜色的试次
                test_idx  = (Y_color == c_idx) & curr_shape_idx; 
                
                mdl_ento = fitcecoc(sub_X_ento(train_idx, :), Y_texture(train_idx), 'Learners', t);
                mdl_mvl  = fitcecoc(sub_X_mvl(train_idx, :),  Y_texture(train_idx), 'Learners', t);
                
                folds_acc_ento(c_idx) = mean(predict(mdl_ento, sub_X_ento(test_idx, :)) == Y_texture(test_idx));
                folds_acc_mvl(c_idx)  = mean(predict(mdl_mvl, sub_X_mvl(test_idx, :))  == Y_texture(test_idx));
            end
            % 当前形状的跨颜色平均准确率
            shape_acc_ento(s_idx) = mean(folds_acc_ento);
            shape_acc_mvl(s_idx)  = mean(folds_acc_mvl);
        end
        % 全局（3个形状）的平均准确率
        acc_ento(i, iter) = mean(shape_acc_ento);
        acc_mvl(i, iter)  = mean(shape_acc_mvl);
    end
    fprintf('完成抽样维度: %d 神经元...\n', n);
end
%% === 5.1 严谨版：各节点动态 CCGP 嵌套置换检验 ( Permutation Test) ===
disp('🚀 正在执行 嵌套模式 的各节点 CCGP 置换检验 (这可能需要几分钟时间)...');
nPerm = 500; 
null_acc_ento = zeros(nPerm, num_steps);
null_acc_mvl  = zeros(nPerm, num_steps);

for p = 1:nPerm
    shuffled_Y_texture = Y_texture;
    for s_idx = 1:3
        curr_shape_idx = find(Y_shape == s_idx);
        shuffled_Y_texture(curr_shape_idx) = Y_texture(curr_shape_idx(randperm(length(curr_shape_idx))));
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
            curr_shape_idx = (Y_shape == s_idx);
            folds_acc_ento_null = zeros(4, 1);
            folds_acc_mvl_null  = zeros(4, 1);
            
            for c_idx = 1:4
                train_idx = (Y_color ~= c_idx) & curr_shape_idx; 
                test_idx  = (Y_color == c_idx) & curr_shape_idx; 
                
                mdl_ento_null = fitcecoc(sub_X_ento(train_idx, :), shuffled_Y_texture(train_idx), 'Learners', t);
                mdl_mvl_null  = fitcecoc(sub_X_mvl(train_idx, :),  shuffled_Y_texture(train_idx), 'Learners', t);
                
                folds_acc_ento_null(c_idx) = mean(predict(mdl_ento_null, sub_X_ento(test_idx, :)) == shuffled_Y_texture(test_idx));
                folds_acc_mvl_null(c_idx)  = mean(predict(mdl_mvl_null, sub_X_mvl(test_idx, :))  == shuffled_Y_texture(test_idx));
            end
            shape_acc_ento_null(s_idx) = mean(folds_acc_ento_null);
            shape_acc_mvl_null(s_idx)  = mean(folds_acc_mvl_null);
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

fprintf('✅ 严谨版嵌套 CCGP 置换检验全部完成！\n');
%% === 5.2 绘制泛化解码曲线 ===
m_ento = mean(acc_ento, 2)' * 100; s_ento = (std(acc_ento, 0, 2)' / sqrt(nIter)) * 100;
m_mvl  = mean(acc_mvl, 2)' * 100;  s_mvl  = (std(acc_mvl, 0, 2)' / sqrt(nIter)) * 100;
%
figure('Position', [100, 100, 650, 500], 'Color', 'w');
set(gcf, 'DefaultAxesFontName', 'Arial');
hold on;

yline(chance_level * 100, '--k', sprintf('Chance (%.1f%%)', chance_level*100), 'LineWidth', 2, 'FontSize', 12, 'LabelHorizontalAlignment', 'left');

X_patch = [neuron_steps, fliplr(neuron_steps)];
fill(X_patch, [m_ento + s_ento, fliplr(m_ento - s_ento)], [0.2 0.4 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill(X_patch, [m_mvl + s_mvl, fliplr(m_mvl - s_mvl)], [0.8 0.2 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

p1 = plot(neuron_steps, m_ento, '-o', 'Color', [0.2 0.4 0.8], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.2 0.4 0.8]);
p2 = plot(neuron_steps, m_mvl, '-s', 'Color', [0.8 0.2 0.2], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.8 0.2 0.2]);

set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.5, 'FontSize', 14);
xlabel('Number of neurons', 'FontSize', 18);
ylabel('Accuracy (%)', 'FontSize', 18);
title('Decoding texture generalizing across color', 'FontSize', 20);
legend([p1, p2], {'ENTO', 'MVL'}, 'Location', 'northwest', 'FontSize', 14, 'Box', 'off');
ylim([20, 80]); 
% === 调整坐标系与图例空间 (绝对统一标尺) ===
% ylim([15, 80]); % 和 Color, Shape 统一高度，给下方留出大量负空间
% legend([p1, p2], {'ENTO', 'MVL'}, 'Location', 'northwest', 'FontSize', 14, 'Box', 'off');

% === 绘制底部的显著性彩色水平线段 (NaN 断点法) ===
% 1. 统一定义线段的高度 (和所有图完全相同的位置，Y=18 和 Y=16.5)
y_pos_ento = 18; 
y_pos_mvl  = 16.5;

% 2. 将不显著的节点设为 NaN
y_line_ento = repmat(y_pos_ento, 1, num_steps);
y_line_ento(p_vals_ento >= alpha_level) = NaN;

y_line_mvl = repmat(y_pos_mvl, 1, num_steps);
y_line_mvl(p_vals_mvl >= alpha_level) = NaN;

% 3. 画显著性水平粗线
plot(neuron_steps, y_line_ento, '-', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(neuron_steps, y_line_mvl, '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'HandleVisibility', 'off');

% 4. 专业的斜体注记，统一放在左下角负空间 (Y=20)
text(max(neuron_steps)*0.6, 20, 'Horizontal lines: p < 0.05', ...
     'FontSize', 12, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');