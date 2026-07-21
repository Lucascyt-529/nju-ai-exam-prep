"""学生练习：合并闭区间。"""
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    raise NotImplementedError("请完成 merge_intervals")
def main():
    n=int(input()); intervals=[tuple(map(int,input().split())) for _ in range(n)]
    for left,right in merge_intervals(intervals): print(left,right)
if __name__=="__main__": main()
