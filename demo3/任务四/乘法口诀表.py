while True:
    try:
        num = int(input("请输入1-9的数字（输入0退出）："))
        if num == 0:
            print("程序已退出")
            break
        if 1 <= num <= 9:
            print(f"\n数字{num}的直角三角形乘法表：")
            for i in range(1, num + 1):
                row = [f"{j}×{i}={j*i:2}" for j in range(1, i + 1)]
                print("  ".join(row))
            print()
        else:
            print("请输入1-9之间的有效数字！")
    except ValueError:
        print("请输入有效的整数！")