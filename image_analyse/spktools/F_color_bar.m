function [] = F_color_bar(ch, net_mua, bar_sem,color_data,window_off)
% =========================================================================
% 绘制单通道柱状图：X轴为颜色(4组)，每组6根柱子(形状-纹理)
% (蓝-绿-黄高级成对配色 + 全局 Arial + 取消加粗 + 图注右上角)
% =========================================================================

% 1. 选择你要画的“精英通道” (例如第 1 个通道)
% ch_idx = 1; 

% 提取对应通道的 24 个刺激的均值和标准误
ch_mean = net_mua; 
ch_sem  = bar_sem;

% 2. 定义 4 种颜色的索引簇
idx_blue   = [1, 5, 9,  13, 17, 21];
idx_green  = [2, 6, 10, 14, 18, 22];
idx_red    = [3, 7, 11, 15, 19, 23];
idx_yellow = [4, 8, 12, 16, 20, 24];

% 3. 重组为 [4种颜色 x 6个形状纹理组合] 的矩阵
plot_mean = zeros(4, 6);
plot_sem  = zeros(4, 6);

% 行代表 X 轴的组 (1:蓝, 2:绿, 3:红, 4:黄)
plot_mean(1, :) = ch_mean(idx_blue);
plot_mean(2, :) = ch_mean(idx_green);
plot_mean(3, :) = ch_mean(idx_red);
plot_mean(4, :) = ch_mean(idx_yellow);

plot_sem(1, :) = ch_sem(idx_blue);
plot_sem(2, :) = ch_sem(idx_green);
plot_sem(3, :) = ch_sem(idx_red);
plot_sem(4, :) = ch_sem(idx_yellow);

% 4. 开始画图
% figure('Position', [100, 100, 950, 480], 'Color', 'w');
figure('Position', [100, 100, 700, 600], 'Color', 'w');
% ★ 强制将当前图窗的默认字体全部设为 Arial ★
set(gcf, 'DefaultAxesFontName', 'Arial');
set(gcf, 'DefaultTextFontName', 'Arial');
    % 将画布大小拓展 方便保存成PDF文件
set(gcf, 'Units', 'Inches');
pos = get(gcf, 'Position');
set(gcf, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)]);

subplot(2, 1,2);
hold on;

% 画分组柱状图
b = bar(plot_mean, 'grouped', 'EdgeColor', 'none');

% =========================================================================
% % 5. ★ 蓝-绿-黄 成对配色方案 (Blue-Green-Yellow Paired Palette) ★
% % 深蓝/浅蓝(形状1), 深绿/浅绿(形状2), 深黄/浅黄(形状3)
% % =========================================================================
colors_6 = [
    0.17  0.40  0.68;   % S1-T1 深海蓝
    0.55  0.71  0.88;   % S1-T2 天际蓝
    0.15  0.55  0.30;   % S2-T1 森林绿
    0.53  0.81  0.60;   % S2-T2 薄荷绿
    0.88  0.65  0.10;   % S3-T1 琥珀黄 (深色)
    0.96  0.85  0.45    % S3-T2 麦穗黄 (浅色)
];
% colors_6 = [
%     0.12  0.47  0.71;   % S1-T1 深海蓝
%     0.65  0.81  0.89;   % S1-T2 天际蓝
%     0.89  0.10  0.11;   % S2-T1 勃艮第红
%     0.98  0.60  0.60;   % S2-T2 珊瑚粉
%     0.20  0.63  0.17;   % S3-T1 森林绿
%     0.70  0.87  0.54    % S3-T2 苹果绿
% ];
for i = 1:6
    b(i).FaceColor = colors_6(i, :);
end

% 6. 叠加误差棒 (Error Bars)
for i = 1:6
    % 获取每组内特定柱子的 X 坐标中心
    x = b(i).XEndPoints; 
    
    % 误差棒统一设为深灰色，线宽调细一点，提升极简轻盈的质感
    errorbar(x, plot_mean(:, i), plot_sem(:, i), 'Color', [0.3 0.3 0.3], ...
        'LineStyle', 'none', 'LineWidth', 1.2, 'CapSize', 3);
end

% 7. 美化图表 (应用 Arial 字体，全面取消加粗 'FontWeight', 'bold', )
set(gca, 'FontSize', 12, 'LineWidth', 1.2, ...
    'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');

% 设置极具专业性的纵坐标标签 (取消加粗)
ylabel('Avg net firing rate (Hz)', 'FontSize', 14, 'FontName', 'Arial');

% 设置 X 轴标签
xticks(1:4);
xticklabels({'Blue', 'Green', 'Red', 'Yellow'}); 

% ★ 添加图例 (移到右上角 'northeast'，两列显示，取消加粗) ★
legend({'S1-T1', 'S1-T2', 'S2-T1', 'S2-T2', 'S3-T1', 'S3-T2'}, ...
    'Location', 'northeastoutside', 'Box', 'off', 'FontSize', 10, ...
    'NumColumns', 1,'FontName', 'Arial');
% set(gca, 'Position', [0.10, 0.15, 0.70, 0.75]);
hold off;
%% 画PSTH图  在主要是颜色影响下 颜色 纹理 形状 带误差阴影的PSTH图
nTrials = size(color_data, 2);
nTime = size(color_data, 1);
for c = 1:4
    % 提取单通道、单颜色的数据 [nTime x nTrials]
    temp_data = squeeze(color_data(:, :, ch, c)); 
    
    % 基线校准 (减去每个 trial 的基线均值)
    baseline = mean(temp_data(window_off(1):window_off(2)-1, :), 1); 
    temp_data_aligned = temp_data - baseline; 
    temp_data_aligned = temp_data_aligned*1000; %单位是HZ *1000转化成s
    temp_data_aligned = smoothdata(temp_data_aligned, 1, 'gaussian', 30); %窗长30ms 步长1ms  
    
    % 计算时间序列上的均值和 SEM （跳过NaN，更稳健）
    mean_ts(:, c) = mean(temp_data_aligned, 2, 'omitnan');
    sem_ts(:, c)  = std(temp_data_aligned, 0, 2, 'omitnan') / sqrt(nTrials);
end

% 3. 开始绘图
% figure('Position', [100, 100, 650, 450], 'Color', 'w');
% 
% % 强制全局 Arial 字体
% set(gcf, 'DefaultAxesFontName', 'Arial');
% set(gcf, 'DefaultTextFontName', 'Arial');
% subplot(2, 1,1);
% hold on;
% 
% colors = [
%     0.17  0.40  0.68;   % Blue 深海蓝
%     0.10  0.75  0.35;  % Green (翠绿)
%     0.95  0.30  0.20;  % Red (亮橙红)
%     0.98  0.85  0.10   % Yellow (亮黄)
% ];
% % =========================================================
% % ★ 画图黑科技：先画所有阴影，再画所有主线 ★
% % 这样可以保证半透明阴影不会覆盖或模糊掉均值主线
% % =========================================================
% % time_vec0 = linspace(-500, 800, nTime);
% load("mt_bin.mat");
% window=200:900; %-0.3s-0.4s
% time_vec = mt_bin(window)*1000;
% 
% % 先画 4 个半透明阴影误差带
% for c = 1:4
%     % 构建多边形的 X 和 Y 坐标
%     X_poly = [time_vec, fliplr(time_vec)];
%     Y_poly = [mean_ts(window, c)' + sem_ts(window, c)', fliplr(mean_ts(window, c)' - sem_ts(window, c)')];
%     
%     % 画阴影 ('FaceAlpha', 0.15 让颜色极其轻透，重叠时依然好看)
%     fill(X_poly, Y_poly, colors(c, :), 'FaceAlpha', 0.35, 'EdgeColor', 'none');
% end
% 
% % 再画 4 条均值主线，并保存句柄 (用于图例)
% h_lines = zeros(1, 4);
% for c = 1:4
%     h_lines(c) = plot(time_vec, mean_ts(window, c), 'Color', colors(c, :), 'LineWidth', 2);
% end
% 
% % % 4. 添加基准线
%  xline(0, 'k--', 'LineWidth', 1.2); % 刺激出现的 t=0 线
% % yline(0, 'k--', 'LineWidth', 1, 'Color', [0.6 0.6 0.6]); % y=0 净放电率基线
% 
% % 5. 美化图表 (完美衔接前图的排版规范)
% set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', 'FontName', 'Arial');
% 
% xlabel('Time (ms)', 'FontSize', 14, 'FontName', 'Arial');
% ylabel('Net firing rate (Hz)', 'FontSize', 14, 'FontName', 'Arial');
% 
% % 设置 X 轴的显示范围 (可以去掉两端多余的空白)
% xlim([time_vec(1), time_vec(end)]); 
% % ylim([-5, 60]);
% % 7. ★ 终极图例排版：彻底移出右侧边界 ★
% % 使用 'northeastoutside' 会将图例放在坐标轴右上方的外部
% % 绝对不会压到任何一根在 400ms 以内的线
% % =========================================================
% legend(h_lines, {'Blue', 'Green', 'Red', 'Yellow'}, ...
%     'Location', 'northeastoutside', ... % 强制放在右上角外部
%     'Box', 'off', ...                   % 去掉图例的丑陋边框
%     'FontSize', 11, ...
%     'FontName', 'Arial');
% 
% % 为了防止图例被切掉，让 MATLAB 自动收缩一点主绘图区来适应外部图例
% % set(gca, 'Position', [0.15, 0.15, 0.6, 0.75]);
% hold off;
%% Raster plot: Color trials
subplot(2, 1, 1);
hold on;

colors = [
    0.17  0.40  0.68;   % Blue
    0.10  0.75  0.35;   % Green
    0.95  0.30  0.20;   % Red
    0.98  0.85  0.10    % Yellow
];

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

        for s = 1:numel(spike_t)
            plot([spike_t(s), spike_t(s)], [y - 0.80, y + 0.80], ...
                'Color', colors(c, :), ...
                'LineWidth', 2.2);
        end
    end

    h_lines(c) = plot(nan, nan, ...
        'Color', colors(c, :), ...
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
    'FontSize', 12, ...
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
ylabel('Trials', 'FontSize', 14, 'FontName', 'Arial');
set(get(gca, 'YLabel'), 'Position', [time_vec(1) - 50, y_offset / 2, 0]);
% legend(h_lines, color_names, ...
%     'Location', 'northeastoutside', ...
%     'Box', 'off', ...
%     'FontSize', 11, ...
%     'FontName', 'Arial');

hold off;





end