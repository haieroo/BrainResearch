function [] = F_morph_bar1(ch, net_mua, bar_sem, texture_data, window_off)
% =========================================================================
% 绘制单通道图表：以【局部形态 (Morph / Local Feature)】为主效应分组
% - 上半部分 (Subplot 1)：PSTH图，6 条曲线 (采用高对比度成对配色 Paired Colors)
% - 下半部分 (Subplot 2)：柱状图，X轴为 3 种形状，每组 2 根柱子 (代表 T1 和 T2，均值化了4种颜色)
% (全局 Arial + 取消加粗 + 图注右上角外部)
% =========================================================================

    % 提取单通道的 24 个刺激的均值和标准误
    ch_mean = net_mua; 
    ch_sem  = bar_sem;
    
    % =====================================================================
    % ★ 6 线/柱 绝佳成对配色 (Paired Colors) ★
    % 深蓝/浅蓝 (S1), 深绿/浅绿 (S2), 深黄/浅黄 (S3)
    % =====================================================================
    paired_colors = [
        0.12  0.47  0.71;   % 1: S1-T1 (深蓝)
        0.65  0.81  0.89;   % 2: S1-T2 (浅蓝)
        0.20  0.63  0.17;   % 3: S2-T1 (深绿)
        0.70  0.87  0.54;   % 4: S2-T2 (浅绿)
        0.88  0.65  0.10;   % 5: S3-T1 琥珀黄 (深色)
        0.96  0.85  0.45    % 6: S3-T2 麦穗黄 (浅色)
    ];

    % 初始化画布
    figure('Position', [100, 100, 700, 600], 'Color', 'w');
    set(gcf, 'DefaultAxesFontName', 'Arial');
    set(gcf, 'DefaultTextFontName', 'Arial');
    % 将画布大小拓展 方便保存成PDF文件
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
   
    % =====================================================================
%     % 1. 绘制 Subplot 1: PSTH 动态图 (上方)
%     % =====================================================================
%     subplot(2, 1, 1);
%     hold on;
%     
%     morph_data = texture_data;
%     nTrials = size(morph_data, 2);
%     nTime   = size(morph_data, 1);
%     
%     mean_ts = zeros(nTime, 6);
%     sem_ts  = zeros(nTime, 6);
%     
%     % 循环 6 次 (6 种局部形态)
%     for c = 1:6
%         temp_data = squeeze(morph_data(:, :, ch, c)); 
%         baseline = mean(temp_data(window_off(1):window_off(2)-1, :), 1); 
%         temp_data_aligned = temp_data - baseline; 
%         temp_data_aligned = temp_data_aligned * 1000; % 转化为 Hz
%         temp_data_aligned = smoothdata(temp_data_aligned, 1, 'gaussian', 30); 
%         
%         mean_ts(:, c) = mean(temp_data_aligned, 2, 'omitnan');
%         sem_ts(:, c)  = std(temp_data_aligned, 0, 2, 'omitnan') / sqrt(nTrials);
%     end
%     
%     load("mt_bin.mat");
%     window = 200:900; % -0.3s - 0.4s
%     time_vec = mt_bin(window) * 1000;
%     
%     % 画 6 个半透明阴影误差带 (透明度降至 0.2)
%     for c = 1:6
%         X_poly = [time_vec, fliplr(time_vec)];
%         Y_poly = [mean_ts(window, c)' + sem_ts(window, c)', fliplr(mean_ts(window, c)' - sem_ts(window, c)')];
%         fill(X_poly, Y_poly, paired_colors(c, :), 'FaceAlpha', 0.20, 'EdgeColor', 'none');
%     end
%    
%     % 画 6 条均值主线 
%     h_lines = zeros(1, 6);
%     for c = 1:6
%         if mod(c, 2) ~= 0 % 奇数 (T1 深色线)
%             line_w = 2; 
%         else              % 偶数 (T2 浅色线)
%             line_w = 2;
%         end
%         h_lines(c) = plot(time_vec, mean_ts(window, c), 'Color', paired_colors(c, :), 'LineWidth', line_w);
%     end
%     
%     % 刺激出现 t=0 线
%     xline(0, 'k--', 'LineWidth', 1.2); 
%     
%     % 美化 PSTH 图
%     set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
%     xlabel('Time (ms)', 'FontSize', 16, 'FontName', 'Arial');
%     ylabel('Net firing rate (Hz)', 'FontSize', 16, 'FontName', 'Arial');
%     xlim([time_vec(1), time_vec(end)]); 
%     
%     % 图例排版：移出右侧边界
%     legend(h_lines, {'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'}, ...
%         'Location', 'northeastoutside', 'Box', 'off', 'FontSize', 11, 'FontName', 'Arial');
%     
%     hold off;
    

    % =====================================================================
    % 1. 绘制 Subplot 1: Raster plot (Morph trials)
    % =====================================================================
    subplot(2, 1, 1);
    hold on;

    morph_data = texture_data;
    morph_names = {'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'};

    load("mt_bin.mat");
    window = 200:900;                 % -0.3 s to 0.4 s
    time_vec = mt_bin(window) * 1000;  % ms

    nTrials = size(morph_data, 2);
    y_offset = 0;
    h_lines = gobjects(1, 6);

    for c = 1:6
        temp_data = squeeze(morph_data(:, :, ch, c));

        % Raster 使用原始 spike/bin 数据，不做 baseline correction 和 smooth
        spike_mat = temp_data(window, :) > 0;

        for tr = 1:nTrials
            spike_idx = find(spike_mat(:, tr));
            spike_t = time_vec(spike_idx);

            y = y_offset + tr;

            for k = 1:numel(spike_t)
                plot([spike_t(k), spike_t(k)], [y - 0.60, y + 0.60], ...
                    'Color', paired_colors(c, :), ...
                    'LineWidth', 1.6);
            end
        end

        h_lines(c) = plot(nan, nan, ...
            'Color', paired_colors(c, :), ...
            'LineWidth', 2);

        if c < 6
            yline(y_offset + nTrials + 0.5, '-', ...
                'Color', [0.75 0.75 0.75], ...
                'LineWidth', 0.8);
        end

        y_offset = y_offset + nTrials;
    end

    xline(0, 'k--', 'LineWidth', 1.2);

    set(gca, ...
        'FontSize', 14, ...
        'LineWidth', 1.2, ...
        'Box', 'off', ...
        'TickDir', 'out', ...
        'FontName', 'Arial');

    xlabel('Time (ms)', 'FontSize', 14, 'FontName', 'Arial');

    xlim([time_vec(1), time_vec(end)]);
    ylim([0.5, y_offset + 0.5]);

    morph_y = [
        nTrials / 2, ...
        nTrials + nTrials / 2, ...
        2 * nTrials + nTrials / 2, ...
        3 * nTrials + nTrials / 2, ...
        4 * nTrials + nTrials / 2, ...
        5 * nTrials + nTrials / 2
    ];

    yticks(morph_y);
    yticklabels([]);

    for c = 1:6
        text(time_vec(1) - 35, morph_y(c), morph_names{c}, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Rotation', 0, ...
            'FontSize', 11, ...
            'FontName', 'Arial', ...
            'Clipping', 'off');
    end
ylabel('Trials', 'FontSize', 16, 'FontName', 'Arial');
set(get(gca, 'YLabel'), 'Position', [time_vec(1) - 60, y_offset / 2, 0]);
%     legend(h_lines, morph_names, ...
%         'Location', 'northeastoutside', ...
%         'Box', 'off', ...
%         'FontSize', 10, ...
%         'FontName', 'Arial');

    hold off;
    % =====================================================================
    % 2. 绘制 Subplot 2: 柱状图 (下方)
    % 数据重组为 [3种形状 x 2种纹理]，将内部的 4 种颜色平均掉
    % =====================================================================
    plot_mean = zeros(3, 2);
    plot_sem  = zeros(3, 2);
    
    for s = 1:3       % 遍历 3 种形状
        for t = 1:2   % 遍历每种形状下的 2 种纹理 (T1, T2)
            % 计算 24 个刺激中对应的 4 个颜色数据的索引
            % 例如 S1-T1 (s=1, t=1) 对应 1~4; S1-T2 (s=1, t=2) 对应 5~8
            idx_start = (s-1)*8 + (t-1)*4 + 1;
            idx_end   = idx_start + 3;
            
            % 求 4 种颜色的平均放电率
            plot_mean(s, t) = mean(ch_mean(idx_start:idx_end));
            % 求 4 种颜色的合并标准误 (假设样本量相同，基于方差传递公式)
            plot_sem(s, t)  = sqrt(sum(ch_sem(idx_start:idx_end).^2)) / 4;
        end
    end
    
    subplot(2, 1, 2);
    hold on;
    
    % 画分组柱状图
    b = bar(plot_mean, 'grouped', 'EdgeColor', 'none');
    
    % =====================================================================
    % ★ 黑科技：使用 CData 让柱子颜色与上方的 PSTH 完全对应 ★
    % 默认 grouped bar 只能给 T1 统一涂一种色，T2 统一涂另一种色
    % 我们通过 Flat CData，强制将不同组（形状）的 T1 和 T2 涂上其专属色彩
    % =====================================================================
    b(1).FaceColor = 'flat';
    b(2).FaceColor = 'flat';
    
    % b(1) 控制所有形状的第一根柱子 (即 S1-T1, S2-T1, S3-T1)
    b(1).CData = [paired_colors(1, :); paired_colors(3, :); paired_colors(5, :)];
    % b(2) 控制所有形状的第二根柱子 (即 S1-T2, S2-T2, S3-T2)
    b(2).CData = [paired_colors(2, :); paired_colors(4, :); paired_colors(6, :)];
    
    % 叠加误差棒
    for s = 1:3
        % 计算 T1 和 T2 柱子中心的 X 坐标
        x_t1 = b(1).XEndPoints(s);
        x_t2 = b(2).XEndPoints(s);
        
        % 分别为 T1 和 T2 画误差棒
        errorbar(x_t1, plot_mean(s, 1), plot_sem(s, 1), 'Color', [0.3 0.3 0.3], ...
            'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
        errorbar(x_t2, plot_mean(s, 2), plot_sem(s, 2), 'Color', [0.3 0.3 0.3], ...
            'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
    end
    
    % 美化柱状图
    set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
    ylabel('Avg net firing rate (Hz)', 'FontSize', 16, 'FontName', 'Arial');
    
    % X 轴标签设为 3 种形状
    xticks(1:3);
    xticklabels({'Shape 1', 'Shape 2', 'Shape 3'}); 
    
    % 下方图表无需图例，上方 PSTH 的图例已经说明了 6 种颜色的含义
    hold off;

end