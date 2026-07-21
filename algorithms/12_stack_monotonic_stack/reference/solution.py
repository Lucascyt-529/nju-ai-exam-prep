def days_until_warmer(temperatures: list[int]) -> list[int]:
    answer=[0]*len(temperatures); stack=[]
    for index,value in enumerate(temperatures):
        while stack and temperatures[stack[-1]]<value:
            previous=stack.pop(); answer[previous]=index-previous
        stack.append(index)
    return answer
def main():
    n=int(input()); values=list(map(int,input().split()))
    if len(values)!=n: raise ValueError("温度数与 n 不一致")
    print(*days_until_warmer(values))
if __name__=="__main__": main()
