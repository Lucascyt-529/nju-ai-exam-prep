"""学生练习：最长无重复子串，输出长度与左闭右开区间。"""
def longest_unique_window(text: str) -> tuple[int, int, int]:
    raise NotImplementedError("请完成 longest_unique_window")
def main() -> None:
    text = input().rstrip("\n")
    length, left, right = longest_unique_window(text)
    print(length); print(left, right)
if __name__ == "__main__":
    main()
