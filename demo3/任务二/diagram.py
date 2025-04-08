def print_right_triangle(n):
    print("\n等腰直角三角形：")
    for i in range(1, n + 1):
        print('*' * i)


def print_hollow_triangle(n):
    print("\n空心等腰三角形：")
    # for i in range(n):
    #     if i == 0:
    #         print('*')
    #     elif i == n - 1:
    #         print('*' * (2 * i + 1))
    #     else:
    #         print('*' + ' ' * (2 * i - 1) + '*')
    for i in range(n):
        for j in range(2 * n - 1):
            # 判断是否为最后一行或左右顶点位置
            if i == n - 1 or j == n - 1 - i or j == n - 1 + i:
                print('*', end='')
            else:
                print(' ', end='')
        print()


def print_hollow_diamond(n):
    print("\n空心菱形：")
    total_lines = 2 * n - 1
    for i in range(total_lines):
        spaces_before = abs(n - 1 - i)
        if i == 0 or i == total_lines - 1:
            print(' ' * spaces_before + '*')
        else:
            middle_space = 2 * (n - 1 - abs(n - 1 - i)) - 1
            print(' ' * spaces_before + '*' + ' ' * middle_space + '*')


def print_solid_square(n):
    print("\n实心正方形：")
    for _ in range(n):
        print('*' * n)


def print_hollow_square(n):
    print("\n空心正方形：")
    for i in range(n):
        if i == 0 or i == n - 1:
            print('*' * n)
        else:
            print('*' + ' ' * (n - 2) + '*')


def show_menu():
    print("\n===== 图形生成器 =====")
    print("1. 等腰直角三角形")
    print("2. 空心等腰三角形")
    print("3. 空心菱形")
    print("4. 实心正方形")
    print("5. 空心正方形")
    print("q. 退出程序")


def main():
    while True:
        show_menu()
        choice = input("请选择要打印的图形（输入数字1-5或q退出）：").strip().lower()

        if choice == 'q':
            break

        options = {
            '1': lambda: print_right_triangle(5),
            '2': lambda: print_hollow_triangle(3),
            '3': lambda: print_hollow_diamond(3),
            '4': lambda: print_solid_square(4),
            '5': lambda: print_hollow_square(5)
        }

        if choice in options:
            options[choice]()
        else:
            print("输入无效，请重新输入1-5之间的数字或q退出。")


if __name__ == "__main__":
    main()