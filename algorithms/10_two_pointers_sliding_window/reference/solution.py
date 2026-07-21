"""参考实现：滑动窗口维护字符最后出现位置。"""
def longest_unique_window(text: str) -> tuple[int, int, int]:
    last = {}; left = 0; best = (0, 0, 0)
    for right, char in enumerate(text):
        if char in last and last[char] >= left:
            left = last[char] + 1
        last[char] = right
        candidate = (right - left + 1, left, right + 1)
        if candidate[0] > best[0]:
            best = candidate
    return best
def main() -> None:
    text = input().rstrip("\n")
    length, left, right = longest_unique_window(text)
    print(length); print(left, right)
if __name__ == "__main__": main()
