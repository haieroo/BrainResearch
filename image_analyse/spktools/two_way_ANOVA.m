function [final_feature_matrix, final_eta2_matrix, all_S_width, all_d_prime,all_sparseness] = two_way_ANOVA(respones_onall, respones_offall)
% =========================================================================
% 视觉特征全维度解码程序 (Hierarchical Nested ANOVA)
% 自动鉴定：宏观形状、颜色、细粒度纹理及其所有非线性交互
% =========================================================================

nTrials = size(respones_onall, 2);
nChannels = size(respones_onall, 3);
nStimuli = size(respones_onall, 4);

% --- 1. 构建上帝标签向量 ---
morph_base = [ones(1,4), 2*ones(1,4), 3*ones(1,4), 4*ones(1,4), 5*ones(1,4), 6*ones(1,4)];
g_morph = repelem(morph_base, nTrials)'; 

shape_base = [ones(1,8), 2*ones(1,8), 3*ones(1,8)];
g_shape = repelem(shape_base, nTrials)';

texture_base = [ones(1,4), 2*ones(1,4), ones(1,4), 2*ones(1,4), ones(1,4), 2*ones(1,4)];
g_texture = repelem(texture_base, nTrials)';

color_base = repmat([1, 2, 3, 4], 1, 6);
g_color = repelem(color_base, nTrials)';

% --- 2. 预分配结果矩阵 ---
p_macro = zeros(nChannels, 3); eta2_macro = zeros(nChannels, 3);
p_micro = zeros(nChannels, 3); eta2_micro = zeros(nChannels, 3);
p_texture_tukey = ones(nChannels, 3);  
p_texture_color_int = ones(nChannels, 3); eta2_texture_color_int = zeros(nChannels, 3);
all_S_width = zeros(nChannels, 1); 
all_d_prime = zeros(nChannels, 1); 
all_sparseness = zeros(nChannels, 1); % ★ 新增：稀疏度矩阵

disp(['🚀 启动全维度双轨解码机... 共计 ', num2str(nChannels), ' 个通道。']);

for ch = 1:nChannels
    resp = squeeze(respones_onall(:,:, ch, :) - respones_offall(:,:, ch, :)); 
    y = resp(:); 
    
    % --- 轨道 A: 宏观视角 ---
    [p_mac, tbl_mac, ~] = anovan(y, {g_shape, g_color}, 'model', 'interaction', 'display', 'off');
    p_macro(ch, :) = p_mac'; 
    SS_error_mac = tbl_mac{size(tbl_mac,1)-1, 2};
    for f = 1:3
        SS_effect = tbl_mac{f+1, 2};
        eta2_macro(ch, f) = SS_effect / (SS_effect + SS_error_mac);
    end  

    % --- 轨道 B: 微观视角 ---
    [p_mic, tbl_mic, stats_mic] = anovan(y, {g_morph, g_color}, 'model', 'interaction', 'display', 'off');
    p_micro(ch, :) = p_mic'; 
    SS_error_mic = tbl_mic{size(tbl_mic,1)-1, 2};
    for f = 1:3
        SS_effect = tbl_mic{f+1, 2};
        eta2_micro(ch, f) = SS_effect / (SS_effect + SS_error_mic);
    end   

    % --- 轨道 C: Tukey 黄金三对 ---   texture 
    if p_mic(1) < 0.05 
        [c_morph, ~, ~, ~] = multcompare(stats_mic, 'Dimension', 1, 'Display', 'off');
        for row = 1:size(c_morph, 1)
            g1 = c_morph(row, 1); g2 = c_morph(row, 2); pval = c_morph(row, 6);
            if (g1 == 1 && g2 == 2), p_texture_tukey(ch, 1) = pval; end
            if (g1 == 3 && g2 == 4), p_texture_tukey(ch, 2) = pval; end
            if (g1 == 5 && g2 == 6), p_texture_tukey(ch, 3) = pval; end
        end
    end
    
    % --- 嵌套子分析: 纹理 x 颜色交互 ---
    for s = 1:3
        idx = (g_shape == s);
        if any(idx)
            [p_sub, tbl_sub] = anovan(y(idx), {g_texture(idx), g_color(idx)}, 'model', 'interaction', 'display', 'off');
            p_texture_color_int(ch, s) = p_sub(3); 
            SS_err_sub = tbl_sub{size(tbl_sub,1)-1, 2};
            SS_int_sub = tbl_sub{4, 2}; 
            eta2_texture_color_int(ch, s) = SS_int_sub / (SS_int_sub + SS_err_sub);
        end
    end
       
    % --- 轨道 D: 辨别力 ---
    mean_resp = max(0, mean(resp, 1)); 
    if max(mean_resp) == 0, all_S_width(ch) = 0;
    else, all_S_width(ch) = (nStimuli - sum(mean_resp / max(mean_resp))) / (nStimuli - 1); end
    
    [mu_pref, idx_pref] = max(mean_resp); 
    [mu_nonpref, idx_nonpref] = min(mean_resp);
    sigma_pooled = sqrt((var(resp(:, idx_pref)) + var(resp(:, idx_nonpref))) / 2);
    if sigma_pooled == 0, all_d_prime(ch) = 0; else, all_d_prime(ch) = (mu_pref - mu_nonpref) / sigma_pooled; end
    
    % ★ Treves-Rolls 生命周期稀疏度 (0 极度泛化 -> 1 极度特异)
    if sum(mean_resp) == 0
        all_sparseness(ch) = 0;
    else
        numerator = (mean(mean_resp))^2;
        denominator = mean(mean_resp.^2);
        a = numerator / denominator;
        all_sparseness(ch) = (1 - a) / (1 - 1/nStimuli); 
    end

end

% --- 第三步：合并与组装矩阵 ---
merged_texture_p = min(p_texture_tukey, [], 2); 
merged_tex_col_int_p = min(p_texture_color_int, [], 2);  %在统计学上，做 3 次独立检验，只要有一次碰巧因为纯噪声导致 $< 0.05$，它就会被判定为显著。这导致假阳性率从 5% 暴涨到了 14.3%！
merged_tex_col_int_eta2 = max(eta2_texture_color_int, [], 2);

final_feature_matrix = zeros(nChannels, 7);
final_eta2_matrix    = zeros(nChannels, 7);

% P值填充
final_feature_matrix(:, 1) = p_macro(:, 1);           % Shape
final_feature_matrix(:, 2) = p_macro(:, 2);           % Color
final_feature_matrix(:, 3) = merged_texture_p;        % Texture
final_feature_matrix(:, 4) = p_macro(:, 3);           % Shape*Color
final_feature_matrix(:, 5) = merged_tex_col_int_p;    % Texture*Color
final_feature_matrix(:, 6) = p_micro(:, 3);           % 终极交互
final_feature_matrix(:, 7) = p_micro(:, 1);           % Morph Identity

% 效应量填充
final_eta2_matrix(:, 1) = eta2_macro(:, 1);   %形状
final_eta2_matrix(:, 2) = eta2_macro(:, 2);   %颜色
final_eta2_matrix(:, 3) = eta2_micro(:, 1); % 纹理强度参考其形态效应
final_eta2_matrix(:, 4) = eta2_macro(:, 3);   %形状*颜色
final_eta2_matrix(:, 5) = merged_tex_col_int_eta2;
final_eta2_matrix(:, 6) = eta2_micro(:, 3);
final_eta2_matrix(:, 7) = eta2_micro(:, 1);

end