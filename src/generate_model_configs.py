import random
import csv


def generate_config_based_on_classics():
    # 考虑上述模型的常见输入尺寸
    heights_widths = [32, 64, 96, 128, 160, 192, 224, 256, 299, 384, 448, 512]
    input_h = random.choice(heights_widths)
    input_w = input_h  # 我们假设宽度和高度是相同的

    # 对于ImageNet等数据集，通常有3个通道
    input_c = 3

    # 批次大小通常在以下范围内选择
    input_n = random.choice([4, 8, 16, 32, 64])

    # 选择其他参数
    kernel_sizes = [(1, 1), (3, 3), (5, 5), (7, 7)]
    kernel_height, kernel_width = random.choice(kernel_sizes)
    padding_height = (kernel_height - 1) // 2
    padding_width = (kernel_width - 1) // 2

    channels = [16, 32, 64, 128, 256, 512, 1024]
    output_channels = random.choice(channels)

    config = {
        'input_n': input_n,
        'input_c': input_c,
        'input_h': input_h,
        'input_w': input_w,
        'output_c': output_channels,
        'kernel_h': kernel_height,
        'kernel_w': kernel_width,
        'padding_height': padding_height,
        'padding_width': padding_width,
        'stride_height': random.choice([1, 2]),
        'stride_width': random.choice([1, 2]),
        'dilation_h': 1,  # 固定为1
        'dilation_w': 1  # 固定为1
    }
    return config


def generate_multiple_configs(num_configs=30000):
    configs = [generate_config_based_on_classics() for _ in range(num_configs)]
    return configs


if __name__ == "__main__":
    random_configs = generate_multiple_configs()

    # 保存到 CSV 文件
    keys = random_configs[0].keys()
    with open('generate_models_configs.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(random_configs)

    print("Configurations saved to generate_models_configs.csv")
