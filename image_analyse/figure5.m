% =========================================================================
% 双脑区全集群跨条件泛化解码 (CCGP) & 泛化混淆矩阵
clc; close all;
% 可选参数：
% 1. 'Decoding color generalizing across shape' 
% 2. 'Decoding shape generalizing across color'    
% 3. 'Decoding texture generalizing across shape' 
% 4.  Decoding morph generalizing across color  六种形态 对颜色的泛化
% 5. 'Decoding color generalizing across texture' 

% === 1. 参数与基础设置 ===

Target_Feature = 'Decoding morph generalizing across color';  

nIter = 100;                   % 随机重抽样次数 
nStim = 24;                   % 刺激总数
Standard_Trial_Num = 8;       % 统一截取最小试次 
Total_Trials = nStim * Standard_Trial_Num; 
Total_Neurons_ENTO = size(X_ENTO, 2);
Total_Neurons_MVL  = size(X_MVL, 2);
fprintf('✅ 数据就绪: ENTO 有 %d 个细胞, MVL 有 %d 个细胞。\n', Total_Neurons_ENTO, Total_Neurons_MVL);

% --- 构建对应的试次标签 ---
shape_labels   = kron([1, 2, 3]', ones(8, 1));        
texture_labels = repmat(kron([1, 2]', ones(4, 1)), 3, 1); 
color_labels   = repmat([1, 2, 3, 4]', 6, 1);         

% 将 24 个刺激的标签扩展到所有 Single Trials
Y_shape   = reshape(repmat(shape_labels', Standard_Trial_Num, 1), [], 1);
Y_color   = reshape(repmat(color_labels', Standard_Trial_Num, 1), [], 1);
Y_texture = reshape(repmat(texture_labels', Standard_Trial_Num, 1), [], 1); 

% 6个形态 主要是跨颜色泛化
morph_labels = kron((1:6)', ones(4, 1));  
Y_morph = reshape(repmat(morph_labels', Standard_Trial_Num, 1), [], 1);
%% === 3. 设定 CCGP 逻辑与标签字典 ===
if strcmp(Target_Feature, 'Decoding color generalizing across shape')
    Y_target = Y_color;     % 目标：解颜色
    Y_gen    = Y_shape;     % 跨越：形状
    gen_conditions = 3;     % 3 个形状轮换留一
    chance_level = 0.25;    % 4种颜色，几率 25%
    class_names = {'Color 1', 'Color 2', 'Color 3', 'Color 4'};
elseif strcmp(Target_Feature, 'Decoding shape generalizing across color')
    Y_target = Y_shape;     % 目标：解形状
    Y_gen    = Y_color;     % 跨越：颜色
    gen_conditions = 4;     % 4 种颜色轮换留一
    chance_level = 1/3;     % 3种形状，几率 ~33.3%
    class_names = {'Shape 1', 'Shape 2', 'Shape 3'};
elseif strcmp(Target_Feature, 'Decoding texture generalizing across shape')
    Y_target = Y_texture;   % 目标：解纹理
    Y_gen    = Y_shape;     % 跨越：形状
    gen_conditions = 3;     % 3 个形状轮换留一
    chance_level = 0.50;    % 2种纹理，几率 50%
    class_names = {'Texture 1', 'Texture 2'};
elseif strcmp(Target_Feature, 'Decoding texture generalizing across color')
    Y_target = Y_texture;   % 目标：解纹理
    Y_gen    = Y_color;     % 跨越：颜色
    gen_conditions = 4;     % 4 种颜色轮换留一
    chance_level = 0.50;    % 2种纹理，几率 50%
    class_names = {'Texture 1', 'Texture 2'};
elseif strcmp(Target_Feature, 'Decoding color generalizing across texture')
    Y_target = Y_color;     % 目标：解颜色
    Y_gen    = Y_texture;   % 跨越：纹理
    gen_conditions = 2;     % 2 种纹理轮换留一
    chance_level = 0.25;    % 4种颜色，几率 25%
    class_names = {'Color 1', 'Color 2', 'Color 3', 'Color 4'};
elseif strcmp(Target_Feature, 'Decoding morph generalizing across color')
    Y_target = Y_morph;     % 目标：解码 6 种 morph
    Y_gen    = Y_color;     % 泛化：跨颜色
    gen_conditions = 4;     % 4 种颜色，留一颜色测试
    chance_level = 1/6;     % 6 类 morph，随机水平 16.7%
    class_names = {'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'};
end
%% === 4. 执行动态抽样 CCGP 解码 ===
max_plot_neurons = min([127, Total_Neurons_ENTO, Total_Neurons_MVL]); 
neuron_steps = [1, 2, 5, 10:10:max_plot_neurons]; 
num_steps = length(neuron_steps);

acc_ento = nan(num_steps, nIter);
acc_mvl  = nan(num_steps, nIter);

t = templateSVM('Standardize', true, 'KernelFunction', 'linear'); 

disp(['🚀 开始执行 CCGP 动态解码: ', Target_Feature, ' ...']);

for i = 1:num_steps
    n = neuron_steps(i);
    for iter = 1:nIter
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        folds_acc_ento = zeros(gen_conditions, 1);
        folds_acc_mvl  = zeros(gen_conditions, 1);
        
        % 留一条件泛化 (CCGP)
        for gen_idx = 1:gen_conditions
            train_idx = (Y_gen ~= gen_idx); 
            test_idx  = (Y_gen == gen_idx); 
            
            mdl_ento = fitcecoc(sub_X_ento(train_idx, :), Y_target(train_idx), 'Learners', t);
            mdl_mvl  = fitcecoc(sub_X_mvl(train_idx, :),  Y_target(train_idx), 'Learners', t);
            
            folds_acc_ento(gen_idx) = mean(predict(mdl_ento, sub_X_ento(test_idx, :)) == Y_target(test_idx));
            folds_acc_mvl(gen_idx)  = mean(predict(mdl_mvl, sub_X_mvl(test_idx, :))  == Y_target(test_idx));
        end
        
        acc_ento(i, iter) = mean(folds_acc_ento);
        acc_mvl(i, iter)  = mean(folds_acc_mvl);
    end
    fprintf('完成抽样维度: %d 神经元...\n', n);
end
%% === 5.1 严谨版：各节点动态 CCGP 置换检验 ( Permutation Test) ===
disp(['🚀 正在执行 ', Target_Feature, ' 的各节点 CCGP 置换检验 (这可能需要几分钟时间)...']);
nPerm = 500; 

null_acc_ento = zeros(nPerm, num_steps);
null_acc_mvl  = zeros(nPerm, num_steps);

for p = 1:nPerm
 
    shuffled_Y = Y_target(randperm(length(Y_target)));
    
    for i = 1:num_steps
        n = neuron_steps(i);    
     
        idx_ento = randperm(Total_Neurons_ENTO, n);
        idx_mvl  = randperm(Total_Neurons_MVL, n);
        
        sub_X_ento = X_ENTO(:, idx_ento);
        sub_X_mvl  = X_MVL(:, idx_mvl);
        
        folds_acc_ento_null = zeros(gen_conditions, 1);
        folds_acc_mvl_null  = zeros(gen_conditions, 1);
        
        for gen_idx = 1:gen_conditions
            train_idx = (Y_gen ~= gen_idx); 
            test_idx  = (Y_gen == gen_idx); 
            
            mdl_ento_null = fitcecoc(sub_X_ento(train_idx, :), shuffled_Y(train_idx), 'Learners', t);
            mdl_mvl_null  = fitcecoc(sub_X_mvl(train_idx, :),  shuffled_Y(train_idx), 'Learners', t);
            
            folds_acc_ento_null(gen_idx) = mean(predict(mdl_ento_null, sub_X_ento(test_idx, :)) == shuffled_Y(test_idx));
            folds_acc_mvl_null(gen_idx)  = mean(predict(mdl_mvl_null, sub_X_mvl(test_idx, :))  == shuffled_Y(test_idx));
        end
        
        null_acc_ento(p, i) = mean(folds_acc_ento_null);
        null_acc_mvl(p, i)  = mean(folds_acc_mvl_null);
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

fprintf('✅ 严谨版各节点 CCGP 置换检验全部完成！\n');
%% === 5. 绘制随神经元数量的解码曲线 ===
m_ento = mean(acc_ento, 2)' * 100; s_ento = (std(acc_ento, 0, 2)' / sqrt(nIter)) * 100;
m_mvl  = mean(acc_mvl, 2)' * 100;  s_mvl  = (std(acc_mvl, 0, 2)' / sqrt(nIter)) * 100;
%
figure('Position', [100, 100, 650, 500], 'Color', 'w');
set(gcf, 'DefaultAxesFontName' ...
    , 'Arial');
hold on;

yline(chance_level * 100, '--k', sprintf('Chance (%.1f%%)', chance_level*100), 'LineWidth', 2, 'FontSize', 12, 'LabelHorizontalAlignment', 'left');

X_patch = [neuron_steps, fliplr(neuron_steps)];
fill(X_patch, [m_ento + s_ento, fliplr(m_ento - s_ento)], [0.2 0.4 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
fill(X_patch, [m_mvl + s_mvl, fliplr(m_mvl - s_mvl)], [0.8 0.2 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none');

p1 = plot(neuron_steps, m_ento, '-o', 'Color', [0.2 0.4 0.8], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.2 0.4 0.8]);
p2 = plot(neuron_steps, m_mvl, '-s', 'Color', [0.8 0.2 0.2], 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', [0.8 0.2 0.2]);

set(gca, 'Box', 'off', 'TickDir', 'out', 'LineWidth', 1.5, 'FontSize', 14);
xlabel('Number of recording sites', 'FontSize', 18);
ylabel('Accuracy (%)', 'FontSize', 18);
title(strrep(Target_Feature, '_', ' '), 'FontSize', 20);
legend([p1, p2], {'ENTO', 'MVL'}, 'Location', 'northwest', 'FontSize', 14, 'Box', 'off');
%ylim([15, 80]);
 ylim([5, 65]); 
% 将图例统一移动到左上角
legend([p1, p2], {'ENTO', 'MVL'}, 'Location', 'northwest', 'FontSize', 14, 'Box', 'off');

% === 绘制底部的显著性彩色水平线段 (NaN 断点法) ===
% y_pos_ento = 18; 
% y_pos_mvl  = 16.5;
y_pos_ento = 8; 
y_pos_mvl  = 6.5;
% 2. 生成 NaN 断点
y_line_ento = repmat(y_pos_ento, 1, num_steps);
y_line_ento(p_vals_ento >= alpha_level) = NaN;

y_line_mvl = repmat(y_pos_mvl, 1, num_steps);
y_line_mvl(p_vals_mvl >= alpha_level) = NaN;

% 3. 画显著性水平粗线 (隐去图例)
plot(neuron_steps, y_line_ento, '-', 'Color', [0.2 0.4 0.8], 'LineWidth', 1.5, 'HandleVisibility', 'off');
plot(neuron_steps, y_line_mvl, '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'HandleVisibility', 'off');

% text(max(neuron_steps)*0.6, 20, 'Horizontal lines: p < 0.05', ...
%      'FontSize', 12, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');

text(max(neuron_steps)*0.6, 10, 'Horizontal lines: p < 0.05', ...
     'FontSize', 12, 'Color', [0.4 0.4 0.4], 'FontAngle', 'italic');
