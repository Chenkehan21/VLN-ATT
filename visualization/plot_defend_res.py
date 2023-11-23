import matplotlib.pyplot as plt


# physical space
physical_hue = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
physical_hue_sr = [64.11, 64.59, 64.25, 62.9, 60.77, 60.53]

physical_bright = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
physical_bright_sr = [64.11, 63.38, 63.09, 62.27, 62.66, 62.32]

physical_contrast = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
physical_contrast_sr = [64.11, 63.43, 63.57, 63.33, 63.63, 62.17]

physical_saturation = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
physical_saturation_sr = [64.11, 64.35, 63.86, 62.27, 61.4, 61.2]
physical_attack = [100] * 6

physical_info = [(physical_hue, physical_hue_sr, physical_attack),
                 (physical_bright, physical_bright_sr, physical_attack),
                 (physical_contrast, physical_contrast_sr, physical_attack),
                 (physical_saturation, physical_saturation_sr, physical_attack)
                 ]

# digital space
digital_hue = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
digital_hue_sr = [65.62, 63.05, 62.88, 61.09, 60.24, 60.28]

digital_bright = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
digital_bright_sr = [65.62, 63.01, 63.22, 62.92, 62.71, 62.24]

digital_contrast = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
digital_contrast_sr = [65.62, 63.13, 62.54, 61.98, 62.41, 63.39]

digital_saturation = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
digital_saturation_sr = [65.62, 63.09, 63.3, 63.09, 63.13, 63.22]
digital_attack = [100] * 6

digital_info = [(digital_hue, digital_hue_sr, digital_attack),
                 (digital_bright, digital_bright_sr, digital_attack),
                 (digital_contrast, digital_contrast_sr, digital_attack),
                 (digital_saturation, digital_saturation_sr, digital_attack)
                 ]


def plot_results(axes, all_info):
    names = ['Hue', 'Brightness', 'Contrast', 'Saturation']
    for i, info in enumerate(all_info):
        x,y,z = info
        x_label = names[i % 4]
        y_label = "Score(%)"
        ax = axes[i]
        ax.set_box_aspect(1)
        ax.tick_params(labelsize=15)
        ax.set_xlabel(x_label, fontsize=20)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=20)
        ax.plot(x,y, color='b', marker='o', label='SR')
        ax.plot(x,z, color='r', marker='s', label='Att-SR')
        ax.grid(True)
        ax.legend(loc='right')


def plot_all_results(axes, all_info):
    names = ['Hue', 'Brightness', 'Contrast', 'Saturation']
    for i, all_info in enumerate(all_info):
        phy_info, dig_info = all_info
        phy_x,phy_y,phy_z = phy_info
        dig_x,dig_y,dig_z = dig_info
        x_label = names[i % 4]
        y_label = "Score(%)"
        ax = axes[i]
        ax.set_box_aspect(0.6)
        ax.set_yticks(range(60,101,10), fontsize=18)
        ax.tick_params(labelsize=18)
        ax.set_xlabel(x_label, fontsize=18)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=18)
        
        ax.plot(phy_x,phy_y, color='r', marker='o', markersize=10, label='SR in Physical Space',zorder=2)
        ax.plot(phy_x,phy_z, color='b', marker='s', markersize=10, label='Att-SR in Physical Space', zorder=3)
        ax.plot(dig_x,dig_y, color='darkorange', marker='P', markersize=10, label='SR in Digital Space', zorder=1)
        ax.scatter(dig_x,dig_z, color='c', marker='d', s=40, label='Att-SR in Digital Space', zorder=4)
        
    lines, labels = ax.get_legend_handles_labels()
    
    return lines, labels
        

fig, axes = plt.subplots(ncols=4, figsize=(20,4.5), sharey=True)
lines, labels = plot_all_results(axes, zip(physical_info, digital_info))
fig.legend(lines, labels, loc='upper center', ncol=4, fontsize=18)
plt.tight_layout()
plt.savefig("./defend_res.png")