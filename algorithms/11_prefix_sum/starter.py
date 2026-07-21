"""学生练习：统计和为 target 的连续子数组数量。"""
def count_subarrays_with_sum(values: list[int], target: int) -> int:
    raise NotImplementedError("请完成 count_subarrays_with_sum")
def main():
    n,target=map(int,input().split()); values=list(map(int,input().split())); assert len(values)==n; print(count_subarrays_with_sum(values,target))
if __name__=="__main__": main()
