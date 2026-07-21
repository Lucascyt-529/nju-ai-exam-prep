def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if any(left>right for left,right in intervals): raise ValueError("区间左端不能大于右端")
    ordered=sorted(intervals); merged=[]
    for left,right in ordered:
        if not merged or left>merged[-1][1]: merged.append((left,right))
        else: merged[-1]=(merged[-1][0],max(merged[-1][1],right))
    return merged
def main():
    n=int(input()); intervals=[tuple(map(int,input().split())) for _ in range(n)]
    for left,right in merge_intervals(intervals): print(left,right)
if __name__=="__main__": main()
