function [] = F_color_bar1(ch, net_mua, bar_sem, color_data, window_off)
% =========================================================================
% 绘制单通道视觉响应图 (双排子图)
% 上半部分 (Subplot 1): 4种颜色主效应的 PSTH 曲线
% 下半部分 (Subplot 2): X轴为 3种形状，每组 4根颜色柱子 (合并T1和T2)
% =========================================================================

% 提取对应通道的 24 个刺激的均值和标准误
ch_mean = net_mua; 
ch_sem  = bar_sem;

% 统一的 4 色配色方案
colors_4 = [
    0.17  0.40  0.68;   % Blue 深海蓝
    0.10  0.75  0.35;   % Green 翠绿
    0.95  0.30  0.20;   % Red 亮橙红
    0.98  0.85  0.10    % Yellow 亮黄
];

% 初始化画布
figure('Position', [100, 100, 700, 600], 'Color', 'w');
% 强制全局 Arial 字体
set(gcf, 'DefaultAxesFontName', 'Arial');
set(gcf, 'DefaultTextFontName', 'Arial');
% 画布大小自适应 PDF 导出
set(gcf, 'Units', 'Inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

% =========================================================================
% % 1. 绘制 Subplot 1: PSTH 曲线图 (上方)
% % =========================================================================
% subplot(2, 1, 1);
% hold on;
% 
% nTrials = size(color_data, 2);
% nTime = size(color_data, 1);
% mean_ts = zeros(nTime, 4);
% sem_ts  = zeros(nTime, 4);
% 
% for c = 1:4
%     % 提取单通道、单颜色的数据 [nTime x nTrials]
%     temp_data = squeeze(color_data(:, :, ch, c)); 
%     
%     % 基线校准
%     baseline = mean(temp_data(window_off(1):window_off(2)-1, :), 1); 
%     temp_data_aligned = temp_data - baseline; 
%     temp_data_aligned = temp_data_aligned * 1000; % 转化为 Hz
%     
%     % 平滑数据 (30ms高斯平滑)
%     temp_data_aligned = smoothdata(temp_data_aligned, 1, 'gaussian', 30);  
%     
%     % 计算时间序列上的均值和 SEM
%     mean_ts(:, c) = mean(temp_data_aligned, 2, 'omitnan');
%     sem_ts(:, c)  = std(temp_data_aligned, 0, 2, 'omitnan') / sqrt(nTrials);
% end
% 
% load("mt_bin.mat");
% window = 200:900; % -0.3s 到 0.4s
% time_vec = mt_bin(window) * 1000;
% 
% % 先画 4 个半透明阴影误差带
% for c = 1:4
%     X_poly = [time_vec, fliplr(time_vec)];
%     Y_poly = [mean_ts(window, c)' + sem_ts(window, c)', fliplr(mean_ts(window, c)' - sem_ts(window, c)')];
%     fill(X_poly, Y_poly, colors_4(c, :), 'FaceAlpha', 0.35, 'EdgeColor', 'none');
% end
% 
% % 再画 4 条均值主线
% h_lines = zeros(1, 4);
% for c = 1:4
%     h_lines(c) = plot(time_vec, mean_ts(window, c), 'Color', colors_4(c, :), 'LineWidth', 2);
% end
% 
% % 添加刺激出现 0 时刻基准线
% xline(0, 'k--', 'LineWidth', 1.2); 
% 
% % 坐标轴美化
% set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
% xlabel('Time (ms)', 'FontSize', 16, 'FontName', 'Arial');
% ylabel('Net firing rate (Hz)', 'FontSize', 16, 'FontName', 'Arial');
% xlim([time_vec(1), time_vec(end)]); 
% 
% % 图例排版 (移出右侧边界)
% legend(h_lines, {'Blue', 'Green', 'Red', 'Yellow'}, ...
%     'Location', 'northeastoutside', 'Box', 'off', 'FontSize', 11, 'FontName', 'Arial');
% 
% hold off;

% =========================================================================
% 1. 绘制 Subplot 1: Raster plot (Color trials)
% =========================================================================
subplot(2, 1, 1);
hold on;

color_names = {'Blue', 'Green', 'Red', 'Yellow'};

load("mt_bin.mat");
window = 200:900;                 % -0.3 s to 0.4 s
time_vec = mt_bin(window) * 1000;  % ms

nTrials = size(color_data, 2);
y_offset = 0;
h_lines = gobjects(1, 4);

for c = 1:4
    temp_data = squeeze(color_data(:, :, ch, c));

    % Raster 使用原始 spike/bin 数据，不做 baseline correction 和 smooth
    spike_mat = temp_data(window, :) > 0;

    for tr = 1:nTrials
        spike_idx = find(spike_mat(:, tr));
        spike_t = time_vec(spike_idx);

        y = y_offset + tr;

        for k = 1:numel(spike_t)
            plot([spike_t(k), spike_t(k)], [y - 0.60, y + 0.60], ...
                'Color', colors_4(c, :), ...
                'LineWidth', 1.6);
        end
    end

    h_lines(c) = plot(nan, nan, ...
        'Color', colors_4(c, :), ...
        'LineWidth', 2);

    if c < 4
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

color_y = [
    nTrials / 2, ...
    nTrials + nTrials / 2, ...
    2 * nTrials + nTrials / 2, ...
    3 * nTrials + nTrials / 2
];

yticks(color_y);
yticklabels([]);

for c = 1:4
    text(time_vec(1) - 35, color_y(c), color_names{c}, ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'Rotation', 90, ...
        'FontSize', 12, ...
        'FontName', 'Arial', ...
        'Clipping', 'off');
end
ylabel('Trials', 'FontSize', 16, 'FontName', 'Arial');
set(get(gca, 'YLabel'), 'Position', [time_vec(1) - 50, y_offset / 2, 0]);
% legend(h_lines, color_names, ...
%     'Location', 'northeastoutside', ...
%     'Box', 'off', ...
%     'FontSize', 11, ...
%     'FontName', 'Arial');

hold off;

% =========================================================================
% 2. 绘制 Subplot 2: 分组柱状图 (下方)
% 数据重组为 [3种形状 x 4种颜色]，将 T1 和 T2 进行合并平均
% =========================================================================
plot_mean = zeros(3, 4);
plot_sem  = zeros(3, 4);

% 遍历 3 种形状 (s) 和 4 种颜色 (c)
for s = 1:3
    for c = 1:4
        % 计算原始数据 24 个刺激中的索引
        % 例如 s=1 (Shape1), T1为 1-4, T2为 5-8
        idx_t1 = (s-1)*8 + c;       
        idx_t2 = (s-1)*8 + 4 + c;   
        
        % 合并 T1 和 T2 的均值
        plot_mean(s, c) = (ch_mean(idx_t1) + ch_mean(idx_t2)) / 2;
        % 合并标准误 (假设样本量相同，误差平方和求均值)
        plot_sem(s, c) = sqrt(ch_sem(idx_t1)^2 + ch_sem(idx_t2)^2) / 2;
    end
end

subplot(2, 1, 2);
hold on;

% 画分组柱状图
b = bar(plot_mean, 'grouped', 'EdgeColor', 'none');

% 给 4 根颜色柱子上色
for c = 1:4
    b(c).FaceColor = colors_4(c, :);
end

% 叠加误差棒
for c = 1:4
    x = b(c).XEndPoints; % 获取每组内特定柱子的 X 坐标中心
    errorbar(x, plot_mean(:, c), plot_sem(:, c), 'Color', [0.3 0.3 0.3], ...
        'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
end

% 坐标轴美化
set(gca, 'FontSize', 14, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
ylabel('Avg net firing rate (Hz)', 'FontSize', 16, 'FontName', 'Arial');

% 设置横坐标为 3 种宏观形状
xticks(1:3);
xticklabels({'Shape 1', 'Shape 2', 'Shape 3'}); 

% 底部无需图例 (上方 PSTH 的图例已经足够解释颜色，保持画面简洁)
hold off;

end