function [] = F_shape_bar(ch, net_mua, bar_sem, shape_data, window_off)
% =========================================================================
% 绘制单通道图表：以【形状 (Shape)】为主效应分组
% - 柱状图：X轴为 3 种形状，每组 8 根柱子 (4颜色 x 2纹理)
% - PSTH图：3 条曲线 (Shape 1, Shape 2, Shape 3)
% (全局 Arial + 取消加粗 + 高级半透明阴影 + 图注右上角外部)
% =========================================================================

    % =====================================================================
    % 1. 柱状图 (Bar Chart) 绘制部分
    % =====================================================================
    % 提取对应通道的 24 个刺激的均值和标准误
    ch_mean = net_mua; 
    ch_sem  = bar_sem;
    
    % 根据你的设定定义 3 种形状的索引簇
    idx_s1 = 1:8;   % 形状 1
    idx_s2 = 9:16;  % 形状 2
    idx_s3 = 17:24; % 形状 3
    % 原始顺序: [1:B-T1, 2:G-T1, 3:R-T1, 4:Y-T1, 5:B-T2, 6:G-T2, 7:R-T2, 8:Y-T2]
    % 目标顺序: [B-T1, B-T2, G-T1, G-T2, R-T1, R-T2, Y-T1, Y-T2]
    reorder_idx = [1, 5, 2, 6, 3, 7, 4, 8];
    % 重组为 [3种形状 x 8个颜色纹理组合] 的矩阵
    plot_mean = zeros(3, 8);
    plot_sem  = zeros(3, 8);
    
    % 行代表 X 轴的组 (1: Shape1, 2: Shape2, 3: Shape3)
    plot_mean(1, :) = ch_mean(idx_s1(reorder_idx));
    plot_mean(2, :) = ch_mean(idx_s2(reorder_idx));
    plot_mean(3, :) = ch_mean(idx_s3(reorder_idx));
    
    plot_sem(1, :) = ch_sem(idx_s1(reorder_idx));
    plot_sem(2, :) = ch_sem(idx_s2(reorder_idx));
    plot_sem(3, :) = ch_sem(idx_s3(reorder_idx));
    
    % 开始画柱状图
%     figure('Position', [100, 100, 950, 480], 'Color', 'w');
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
    
    % ★ 8 根柱子的高级配色 (完美对应 4种颜色 x 2种纹理) ★
    % 深蓝/浅蓝, 深绿/浅绿, 深红/浅粉, 深黄/浅黄
    colors_8 = [
        0.12  0.47  0.71;   % 1: Blue-T1
        0.65  0.81  0.89;   % 2: Blue-T2
        0.20  0.63  0.17;   % 3: Green-T1
        0.70  0.87  0.54;   % 4: Green-T2
        0.89  0.10  0.11;   % 5: Red-T1
        0.98  0.60  0.60;   % 6: Red-T2
        0.88  0.65  0.10;   % 7: Yellow-T1
        0.96  0.85  0.45    % 8: Yellow-T2
    ];
    
    for i = 1:8
        b(i).FaceColor = colors_8(i, :);
    end
    
    % 叠加误差棒
    for i = 1:8
        x = b(i).XEndPoints; 
        errorbar(x, plot_mean(:, i), plot_sem(:, i), 'Color', [0.3 0.3 0.3], ...
            'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
    end
    
    % 美化柱状图
    set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
    ylabel('Avg net firing rate (Hz)', 'FontSize', 14, 'FontName', 'Arial');
    
    xticks(1:3);
    xticklabels({'Shape 1', 'Shape 2', 'Shape 3'}); 
    
    legend({'Blue-T1', 'Blue-T2', 'Green-T1', 'Green-T2', 'Red-T1', 'Red-T2', 'Yellow-T1', 'Yellow-T2'}, ...
        'Location', 'northeastoutside', 'Box', 'off', 'FontSize', 10, ...
        'NumColumns', 1, 'FontName', 'Arial'); % 这里图例变多了，单列排在外侧更好看
    
    % 收缩主图区，给外侧图例留空间
%     set(gca, 'Position', [0.10, 0.15, 0.70, 0.75]);
    hold off;


    % =====================================================================
    % 2. PSTH 动态图 (Shape Average) 绘制部分
    % =====================================================================
    nTrials = size(shape_data, 2);
    nTime   = size(shape_data, 1);
    
    mean_ts = zeros(nTime, 3);
    sem_ts  = zeros(nTime, 3);
    
    % 循环 3 次 (因为有 3 种形状)
    for c = 1:3
        temp_data = squeeze(shape_data(:, :, ch, c)); 
        baseline = mean(temp_data(window_off(1):window_off(2)-1, :), 1); 
        temp_data_aligned = temp_data - baseline; 
        temp_data_aligned = temp_data_aligned*1000;
        temp_data_aligned = smoothdata(temp_data_aligned, 1, 'gaussian', 30);
        
        mean_ts(:, c) = mean(temp_data_aligned, 2, 'omitnan');
        sem_ts(:, c)  = std(temp_data_aligned, 0, 2, 'omitnan') / sqrt(nTrials);
    end
    
%     figure('Position', [100, 100, 650, 450], 'Color', 'w');
%     set(gcf, 'DefaultAxesFontName', 'Arial');
%     set(gcf, 'DefaultTextFontName', 'Arial');
%     subplot(2, 1,1);
%     hold on;
%     
%     % ★ PSTH 专用的高级形状色 (避免与蓝绿红黄混淆) ★
% %     shape_colors = [
% %         0.55  0.25  0.80;  % Shape 1: 高级紫 (Purple)
% %         0.10  0.65  0.75;  % Shape 2: 孔雀青 (Teal)
% %         0.90  0.45  0.15   % Shape 3: 亮橙色 (Orange)
% %     ];
%     shape_colors = [
%         0.17  0.40  0.68;   % Shape 1: 深海蓝
%         0.15  0.55  0.30;  % Shape 2: 森林绿
%         0.88  0.65  0.10;  % Shape 3: 琥珀黄 (深色)
%     ];
% %     time_vec0 = linspace(-500, 800, nTime);
%     load("mt_bin.mat");
%     window=200:900; % -0.3s-0.4s
%     time_vec = mt_bin(window)*1000; % 单位变成ms
%     
%     % 画 3 个半透明阴影误差带
%     for c = 1:3
%         X_poly = [time_vec, fliplr(time_vec)];
%         Y_poly = [mean_ts(window, c)' + sem_ts(window, c)', fliplr(mean_ts(window, c)' - sem_ts(window, c)')];
%         fill(X_poly, Y_poly, shape_colors(c, :), 'FaceAlpha', 0.35, 'EdgeColor', 'none');
%     end
%     
%     % 画 3 条均值主线
%     h_lines = zeros(1, 3);
%     for c = 1:3
%         h_lines(c) = plot(time_vec, mean_ts(window, c), 'Color', shape_colors(c, :), 'LineWidth', 2);
%     end
%      xline(0, 'k--', 'LineWidth', 1.2); % 刺激出现的 t=0 线
%     % 美化 PSTH 图
%     set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
%     xlabel('Time (ms)', 'FontSize', 14, 'FontName', 'Arial');
%     ylabel('Net firing rate (Hz)', 'FontSize', 14, 'FontName', 'Arial');
%     
%     xlim([time_vec(1), time_vec(end)]); 
% %     ylim([-20, 160]);
%     % 终极图例排版：彻底移出右侧边界
%     legend(h_lines, {'Shape 1', 'Shape 2', 'Shape 3'}, ...
%         'Location', 'northeastoutside', ... 
%         'Box', 'off', 'FontSize', 11, 'FontName', 'Arial');
%     
%     % 收缩主图区适应外部图例
% %     set(gca, 'Position', [0.15, 0.15, 0.6, 0.75]);
%     
%     hold off;

        % =====================================================================
    % 2. Raster plot (Shape trials) 绘制部分
    % =====================================================================
    subplot(2, 1, 1);
    hold on;

    shape_colors = [
        0.17  0.40  0.68;   % Shape 1
        0.15  0.55  0.30;   % Shape 2
        0.88  0.65  0.10    % Shape 3
    ];

    load("mt_bin.mat");
    window = 200:900;                 % -0.3 s to 0.4 s
    time_vec = mt_bin(window) * 1000;  % ms

    nTrials = size(shape_data, 2);
    y_offset = 0;
    h_lines = gobjects(1, 3);

    for c = 1:3
        temp_data = squeeze(shape_data(:, :, ch, c));

        % Raster 使用原始 spike/bin 数据，不做 baseline correction 和 smooth
        spike_mat = temp_data(window, :) > 0;

        for tr = 1:nTrials
            spike_idx = find(spike_mat(:, tr));
            spike_t = time_vec(spike_idx);

            y = y_offset + tr;

            for s = 1:numel(spike_t)
                plot([spike_t(s), spike_t(s)], [y - 0.60, y + 0.60], ...
                    'Color', shape_colors(c, :), ...
                    'LineWidth', 1.6);
            end
        end

        % 用不可见线生成 legend
        h_lines(c) = plot(nan, nan, ...
            'Color', shape_colors(c, :), ...
            'LineWidth', 2);

        % shape 之间加分隔线
        if c < 3
            yline(y_offset + nTrials + 0.5, '-', ...
                'Color', [0.75 0.75 0.75], ...
                'LineWidth', 0.8);
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
    ylabel('Trials', 'FontSize', 14, 'FontName', 'Arial');

    xlim([time_vec(1), time_vec(end)]);
    ylim([0.5, y_offset + 0.5]);

    yticks([
        nTrials / 2, ...
        nTrials + nTrials / 2, ...
        2 * nTrials + nTrials / 2
    ]);

    
    yticks([
    nTrials / 2, ...
    nTrials + nTrials / 2, ...
    2 * nTrials + nTrials / 2
]);

yticklabels([]);

shape_y = [
    nTrials / 2, ...
    nTrials + nTrials / 2, ...
    2 * nTrials + nTrials / 2
];

shape_names = {'Shape 1', 'Shape 2', 'Shape 3'};

for c = 1:3
    text(time_vec(1) - 35, shape_y(c), shape_names{c}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'Rotation', 90, ...
        'FontSize', 12, ...
        'FontName', 'Arial', ...
        'Clipping', 'off');
end

ylabel('Trials', 'FontSize', 14, 'FontName', 'Arial');
set(get(gca, 'YLabel'), 'Position', [time_vec(1) - 50, y_offset / 2, 0]);

    hold off;



end