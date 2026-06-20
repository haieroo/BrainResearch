function [] = F_morph_bar(ch, net_mua, bar_sem, texture_data, window_off)
% =========================================================================
% 绘制单通道图表：以【局部形态 (Morph / Local Feature)】为主效应分组
% - 柱状图：X轴为 6 种形态(S1T1~S3T2)，每组 4 根柱子(4种颜色)
% - PSTH图：6 条曲线 (采用高对比度成对配色 Paired Colors)
% (全局 Arial + 取消加粗 + 图注右上角外部)
% =========================================================================

    % =====================================================================
    % 1. 柱状图 (Bar Chart) 绘制部分
    % =====================================================================
    % 提取单通道的 24 个刺激的均值和标准误
    ch_mean = net_mua; 
    ch_sem  = bar_sem;
    
    % 根据你提供的图片顺序，24 个刺激非常规整：
    % 1-4: S1_T1(蓝绿红黄), 5-8: S1_T2(蓝绿红黄) ... 依此类推
    % 直接重组为 [6种形态 x 4种颜色] 的矩阵
    plot_mean = zeros(6, 4);
    plot_sem  = zeros(6, 4);
    
    for m = 1:6
        idx_start = (m-1)*4 + 1;
        idx_end   = m*4;
        plot_mean(m, :) = ch_mean(idx_start:idx_end);
        plot_sem(m, :)  = ch_sem(idx_start:idx_end);
    end
    
    % 开始画柱状图
%     figure('Position', [100, 100, 1000, 480], 'Color', 'w');
    figure('Position', [100, 100, 700, 600], 'Color', 'w');
    set(gcf, 'DefaultAxesFontName', 'Arial');
    set(gcf, 'DefaultTextFontName', 'Arial');
     % 将画布大小拓展 方便保存成PDF文件
set(gcf, 'Units', 'Inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);
   
    subplot(2, 1,2);

    hold on;
    
    b = bar(plot_mean, 'grouped', 'EdgeColor', 'none');
    
%     % ★ 内部 4 根柱子的物理颜色 (蓝、绿、红、黄) ★
    phys_colors = [
        0.17  0.40  0.68;   % Blue
        0.15  0.55  0.30;   % Green
        0.85  0.20  0.20;   % Red
        0.90  0.70  0.10    % Yellow
    ];
%     phys_colors = [
%     0.17  0.40  0.68;   % Blue 深海蓝
% %    0.15  0.35  0.95;  % Blue (亮宝蓝)
%     0.10  0.75  0.35;  % Green (翠绿)
%     0.95  0.30  0.20;  % Red (亮橙红)
%     0.98  0.85  0.10   % Yellow (亮黄)
%     ];
    
    for i = 1:4
        b(i).FaceColor = phys_colors(i, :);
    end
    
    % 叠加误差棒
    for i = 1:4
        x = b(i).XEndPoints; 
        errorbar(x, plot_mean(:, i), plot_sem(:, i), 'Color', [0.3 0.3 0.3], ...
            'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
    end
    
    % 美化柱状图
    set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
    ylabel('Avg net firing rate (Hz)', 'FontSize', 18, 'FontName', 'Arial');
    
    % X 轴标签设为 6 种形态
    xticks(1:6);
    xticklabels({'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'}); 
    
    % 图例 (表示内部的 4 种颜色)
    legend({'Blue', 'Green', 'Red', 'Yellow'}, ...
        'Location', 'northeastoutside', 'Box', 'off', 'FontSize', 10, 'FontName', 'Arial'); 
    
    % 收缩主图区，给外侧图例留空间
%     set(gca, 'Position', [0.10, 0.15, 0.75, 0.75]);
    hold off;
    

    % =====================================================================
    % 2. PSTH 动态图 (Morph Average) 绘制部分
    % =====================================================================
    morph_data=texture_data;
    nTrials = size(morph_data, 2);
    nTime   = size(morph_data, 1);
    
    mean_ts = zeros(nTime, 6);
    sem_ts  = zeros(nTime, 6);
    
    % 循环 6 次 (6 种局部形态)
    for c = 1:6
        temp_data = squeeze(morph_data(:, :, ch, c)); 
        baseline = mean(temp_data(window_off(1):window_off(2)-1, :), 1); 
        temp_data_aligned = temp_data - baseline; 
        temp_data_aligned = temp_data_aligned*1000;
        temp_data_aligned = smoothdata(temp_data_aligned, 1, 'gaussian', 30); %huo 20
        
        mean_ts(:, c) = mean(temp_data_aligned, 2, 'omitnan');
        sem_ts(:, c)  = std(temp_data_aligned, 0, 2, 'omitnan') / sqrt(nTrials);
    end
    
% %     figure('Position', [100, 100, 750, 480], 'Color', 'w');
% %     set(gcf, 'DefaultAxesFontName', 'Arial');
% %     set(gcf, 'DefaultTextFontName', 'Arial');
% %      hold on;
%     subplot(2, 1,1);
%     hold on;
%     % =====================================================================
%     % ★ 6 线绝佳配色：成对配色 (Paired Colors) ★
%     % 深蓝/浅蓝 (S1), 深绿/浅绿 (S2), 深红/浅红 (S3)
%     % 这样能在 6 条线中清晰地看出“同一形状下不同纹理”的分化！
%     % =====================================================================
%     paired_colors = [
%         0.12  0.47  0.71;   % 1: S1-T1 (深蓝)
%         0.65  0.81  0.89;   % 2: S1-T2 (浅蓝)
%         0.20  0.63  0.17;   % 3: S2-T1 (深绿)
%         0.70  0.87  0.54;   % 4: S2-T2 (浅绿)
% %         0.89  0.10  0.11;   % 5: S3-T1 (深红)
% %         0.98  0.60  0.60    % 6: S3-T2 (浅红)
%         0.88  0.65  0.10;   % S3-T1 琥珀黄 (深色)
%         0.96  0.85  0.45    % S3-T2 麦穗黄 (浅色)
%     ];
%     
% %     time_vec0 = linspace(-500, 800, nTime);
%     load("mt_bin.mat");
%     window=200:900; %-0.3s-0.4s
%     time_vec = mt_bin(window)*1000;
%     
%     % 先画 6 个半透明阴影误差带
%     % (透明度降至 0.2，防止 6 个阴影重叠时颜色过深)
%     for c = 1:6
%         X_poly = [time_vec, fliplr(time_vec)];
%         Y_poly = [mean_ts(window, c)' + sem_ts(window, c)', fliplr(mean_ts(window, c)' - sem_ts(window, c)')];
%         fill(X_poly, Y_poly, paired_colors(c, :), 'FaceAlpha', 0.20, 'EdgeColor', 'none');
%     end
%    
%    
%     % 再画 6 条均值主线 (深色线加粗，浅色线稍细，增加层次感)
%     h_lines = zeros(1, 6);
%     for c = 1:6
%         if mod(c, 2) ~= 0 % 奇数 (T1 深色线)
%             line_w = 2; 
%         else              % 偶数 (T2 浅色线)
%             line_w = 2;
%         end
%         h_lines(c) = plot(time_vec, mean_ts(window, c), 'Color', paired_colors(c, :), 'LineWidth', line_w);
%     end
%      xline(0, 'k--', 'LineWidth', 1.2); % 刺激出现的 t=0 线
%     % 美化 PSTH 图
%     set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
%     xlabel('Time (ms)', 'FontSize', 14, 'FontName', 'Arial');
%     ylabel('Net firing rate (Hz)', 'FontSize', 14, 'FontName', 'Arial');
%     
%     xlim([time_vec(1), time_vec(end)]); 
% %     ylim([-40, 80]);
%     % 终极图例排版：移出右侧边界
%     legend(h_lines, {'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'}, ...
%         'Location', 'northeastoutside', ... 
%         'Box', 'off', 'FontSize', 11, 'FontName', 'Arial');
%     
% %     收缩主图区适应外部图例
% %     set(gca, 'Position', [0.12, 0.15, 0.65, 0.75]);
%     
%     hold off;

    % =====================================================================
    % 2. Raster plot (Morph trials) 绘制部分
    % =====================================================================
    subplot(2, 1, 1);
    hold on;

    morph_data = texture_data;

    paired_colors = [
        0.12  0.47  0.71;   % S1-T1
        0.65  0.81  0.89;   % S1-T2
        0.20  0.63  0.17;   % S2-T1
        0.70  0.87  0.54;   % S2-T2
        0.88  0.65  0.10;   % S3-T1
        0.96  0.85  0.45    % S3-T2
    ];

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

            for s = 1:numel(spike_t)
                plot([spike_t(s), spike_t(s)], [y - 0.80, y + 0.80], ...
                    'Color', paired_colors(c, :), ...
                    'LineWidth', 2.2);
            end
        end

        h_lines(c) = plot(nan, nan, ...
            'Color', paired_colors(c, :), ...
            'LineWidth', 2);

        if c < 6
            yline(y_offset + nTrials + 0.5, '-', ...
                'Color', [0.88 0.88 0.88], ...
                'LineWidth', 0.6);
        end

        y_offset = y_offset + nTrials;
    end

    xline(0, 'k--', 'LineWidth', 1.2);

    set(gca, ...
        'FontSize', 12, ...
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
            'FontSize', 10, ...
            'FontName', 'Arial', ...
            'Clipping', 'off');
    end
ylabel('Trials', 'FontSize', 18, 'FontName', 'Arial');
set(get(gca, 'YLabel'), 'Position', [time_vec(1) - 60, y_offset / 2, 0]);
%     legend(h_lines, morph_names, ...
%         'Location', 'northeastoutside', ...
%         'Box', 'off', ...
%         'FontSize', 10, ...
%         'FontName', 'Arial');

    hold off;
% set(gcf, 'Renderer', 'painters');
% set(findall(gcf, '-property', 'FontName'), 'FontName', 'Arial');
% 
% outname = 'F:\peng\Data_picture_fixed_Analyse_picture\Analyse_chengxu\Spikeanalyse\spktools\F_shape_bar.pdf';
% print(gcf, outname, '-dpdf', '-painters', '-bestfit');

end