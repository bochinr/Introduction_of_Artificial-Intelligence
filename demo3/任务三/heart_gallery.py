import itertools


def generate_solid_heart(text):
    """生成实心文字心形图案"""
    # 实心心形模板（数字表示填充顺序）
    heart_matrix = [
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    ]

    # 创建字符循环迭代器
    char_cycle = itertools.cycle(text) if text else itertools.repeat('❤')

    # 生成心形图案
    output = []
    for row in heart_matrix:
        line = []
        for cell in row:
            line.append(next(char_cycle) if cell else ' ')
        output.append(''.join(line))

    return '\033[91m' + '\n'.join(output) + '\033[0m'


def main():
    while True:
        user_input = input("\n请输入组成心形的文字（回车退出）：").strip()
        if not user_input:
            print("程序已退出")
            break

        print(generate_solid_heart(user_input))


if __name__ == "__main__":
    main()