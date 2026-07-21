def count_subarrays_with_sum(values: list[int], target: int) -> int:
    counts={0:1}; prefix=0; answer=0
    for value in values:
        prefix+=value; answer+=counts.get(prefix-target,0); counts[prefix]=counts.get(prefix,0)+1
    return answer
def main():
    n,target=map(int,input().split()); values=list(map(int,input().split()))
    if len(values)!=n: raise ValueError("元素数与 n 不一致")
    print(count_subarrays_with_sum(values,target))
if __name__=="__main__": main()
