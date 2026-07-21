from collections import Counter
import heapq
def top_k_frequent(values: list[int], k: int) -> list[int]:
    counts=Counter(values)
    if not 0<=k<=len(counts): raise ValueError("k 超出不同元素数量")
    return [value for _,value in heapq.nsmallest(k, ((-count,value) for value,count in counts.items()))]
def main():
    n,k=map(int,input().split()); values=list(map(int,input().split()))
    if len(values)!=n: raise ValueError("元素数与 n 不一致")
    print(*top_k_frequent(values,k))
if __name__=="__main__": main()
