function [] = F_shape_bar1(ch, net_mua, bar_sem, shape_data, window_off)
% =========================================================================
% 绘制单通道图表：以【形状 (Shape)】为主效应分组
% - 上半部分 (Subplot 1)：PSTH图，3 条曲线 (Shape 1, Shape 2, Shape 3)
% - 下半部分 (Subplot 2)：柱状图，X轴为 4 种颜色，每组 3 根柱子 (合并T1和T2)
% (全局 Arial + 取消加粗 + 图注右上角外部)
% =========================================================================

    % 提取对应通道的 24 个刺激的均值和标准误
    ch_mean = net_mua; 
    ch_sem  = bar_sem;

    % ★ PSTH 与柱状图统一的高级形状配色 ★
    shape_colors = [
        0.17  0.40  0.68;   % Shape 1: 深海蓝
        0.15  0.55  0.30;   % Shape 2: 森林绿
        0.88  0.65  0.10;   % Shape 3: 琥珀黄 (深色)
    ];

    % 初始化画布
    figure('Position', [100, 100, 700, 600], 'Color', 'w');
    set(gcf, 'DefaultAxesFontName', 'Arial');
    set(gcf, 'DefaultTextFontName', 'Arial');
    % 画布大小自适应 PDF 导出
    set(gcf, 'Units', 'Inches');
    pos = get(gcf, 'Position');
    set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

    % =====================================================================
%     % 1. 绘制 Subplot 1: PSTH 动态图 (上方)
%     % =====================================================================
%     subplot(2, 1, 1);
%     hold on;
%     
%     nTrials = size(shape_data, 2);
%     nTime   = size(shape_data, 1);
%     
%     mean_ts = zeros(nTime, 3);
%     sem_ts  = zeros(nTime, 3);
%     
%     % 循环 3 次 (计算 3 种形状的时间序列)
%     for s = 1:3
%         temp_data = squeeze(shape_data(:, :, ch, s)); 
%         baseline = mean(temp_data(window_off(1):window_off(2)-1, :), 1); 
%         temp_data_aligned = temp_data - baseline; 
%         temp_data_aligned = temp_data_aligned * 1000; % 转换为 Hz
%         temp_data_aligned = smoothdata(temp_data_aligned, 1, 'gaussian', 30);
%         
%         mean_ts(:, s) = mean(temp_data_aligned, 2, 'omitnan');
%         sem_ts(:, s)  = std(temp_data_aligned, 0, 2, 'omitnan') / sqrt(nTrials);
%     end
%     
%     load("mt_bin.mat");
%     window = 200:900; % -0.3s-0.4s
%     time_vec = mt_bin(window) * 1000; % 单位变成 ms
%     
%     % 画 3 个半透明阴影误差带
%     for s = 1:3
%         X_poly = [time_vec, fliplr(time_vec)];
%         Y_poly = [mean_ts(window, s)' + sem_ts(window, s)', fliplr(mean_ts(window, s)' - sem_ts(window, s)')];
%         fill(X_poly, Y_poly, shape_colors(s, :), 'FaceAlpha', 0.35, 'EdgeColor', 'none');
%     end
%     
%     % 画 3 条均值主线
%     h_lines = zeros(1, 3);
%     for s = 1:3
%         h_lines(s) = plot(time_vec, mean_ts(window, s), 'Color', shape_colors(s, :), 'LineWidth', 2);
%     end
%     
%     % 刺激出现 t=0 的基准线
%     xline(0, 'k--', 'LineWidth', 1.2); 
%     
%     % 美化 PSTH 图
%     set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
%     xlabel('Time (ms)', 'FontSize', 16, 'FontName', 'Arial');
%     ylabel('Net firing rate (Hz)', 'FontSize', 16, 'FontName', 'Arial');
%     xlim([time_vec(1), time_vec(end)]); 
%     
%     % 图例排版：移出右侧边界
%     legend(h_lines, {'Shape 1', 'Shape 2', 'Shape 3'}, ...
%         'Location', 'northeastoutside', 'Box', 'off', 'FontSize', 11, 'FontName', 'Arial');
%     
%     hold off;
    % =====================================================================
    % 1. 绘制 Subplot 1: Raster plot (Shape trials)
    % =====================================================================
    subplot(2, 1, 1);
    hold on;

    load("mt_bin.mat");
    window = 200:900;                 % -0.3 s to 0.4 s
    time_vec = mt_bin(window) * 1000;  % ms

    nTrials = size(shape_data, 2);
    y_offset = 0;
    h_lines = gobjects(1, 3);

    for s = 1:3
        temp_data = squeeze(shape_data(:, :, ch, s));

        % Raster 使用原始 spike/bin 数据，不做 baseline correction 和 smooth
        spike_mat = temp_data(window, :) > 0;

        for tr = 1:nTrials
            spike_idx = find(spike_mat(:, tr));
            spike_t = time_vec(spike_idx);

            y = y_offset + tr;

            for k = 1:numel(spike_t)
                plot([spike_t(k), spike_t(k)], [y - 0.60, y + 0.60], ...
                    'Color', shape_colors(s, :), ...
                    'LineWidth', 1.6);
            end
        end

        h_lines(s) = plot(nan, nan, ...
            'Color', shape_colors(s, :), ...
            'LineWidth', 2);

        if s < 3
            yline(y_offset + nTrials + 0.5, '-', ...
                'Color', [0.88 0.88 0.88], ...
                'LineWidth', 0.6);
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

    shape_y = [
        nTrials / 2, ...
        nTrials + nTrials / 2, ...
        2 * nTrials + nTrials / 2
    ];

    yticks(shape_y);
    yticklabels([]);

    shape_names = {'Shape 1', 'Shape 2', 'Shape 3'};

    for s = 1:3
        text(time_vec(1) - 35, shape_y(s), shape_names{s}, ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'Rotation', 90, ...
            'FontSize', 12, ...
            'FontName', 'Arial', ...
            'Clipping', 'off');
    end
ylabel('Trials', 'FontSize', 16, 'FontName', 'Arial');
set(get(gca, 'YLabel'), 'Position', [time_vec(1) - 50, y_offset / 2, 0]);
%     legend(h_lines, shape_names, ...
%         'Location', 'northeastoutside', ...
%         'Box', 'off', ...
%         'FontSize', 11, ...
%         'FontName', 'Arial');

    hold off;




    % =====================================================================
    % 2. 绘制 Subplot 2: 分组柱状图 (下方)
    % 数据重组为 [4种颜色 x 3种形状]，将 T1 和 T2 进行合并平均
    % =====================================================================
    plot_mean = zeros(4, 3);
    plot_sem  = zeros(4, 3);
    
    % 遍历 4 种颜色 (c) 和 3 种形状 (s)
    for c = 1:4
        for s = 1:3
            % 计算原始数据 24 个刺激中的索引
            % 例如 s=1, c=1 (Shape 1, Blue), T1为1, T2为5
            idx_t1 = (s-1)*8 + c;       
            idx_t2 = (s-1)*8 + 4 + c;   
            
            % 合并 T1 和 T2 的均值，填入矩阵的第 c 行，第 s 列
            plot_mean(c, s) = (ch_mean(idx_t1) + ch_mean(idx_t2)) / 2;
            % 合并标准误
            plot_sem(c, s) = sqrt(ch_sem(idx_t1)^2 + ch_sem(idx_t2)^2) / 2;
        end
    end
    
    subplot(2, 1, 2);
    hold on;
    
    % 画分组柱状图 (X轴自然映射为矩阵的行，即 4 种颜色)
    b = bar(plot_mean, 'grouped', 'EdgeColor', 'none');
    
    % 给每组内的 3 根柱子 (代表 3 种形状) 上色，与 PSTH 完全对应
    for s = 1:3
        b(s).FaceColor = shape_colors(s, :);
    end
    
    % 叠加误差棒
    for s = 1:3
        x = b(s).XEndPoints; 
        errorbar(x, plot_mean(:, s), plot_sem(:, s), 'Color', [0.3 0.3 0.3], ...
            'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
    end
    
    % 美化柱状图
    set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
    ylabel('Avg net firing rate (Hz)', 'FontSize', 16, 'FontName', 'Arial');
    
    % 设置 X 轴为 4 种颜色
    xticks(1:4);
    xticklabels({'Blue', 'Green', 'Red', 'Yellow'}); 
    
    % 底部无需图例 (上方 PSTH 的图例已经足够解释 Shape 1, 2, 3，保持画面简洁)
    hold off;

end