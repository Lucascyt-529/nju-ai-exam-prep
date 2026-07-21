def generate_subsets(values: list[int]) -> list[list[int]]:
    result=[]; path=[]
    def search(index):
        if index==len(values): result.append(path.copy()); return
        search(index+1); path.append(values[index]); search(index+1); path.pop()
    search(0); return result
def main():
    n=int(input()); values=list(map(int,input().split())) if n else []
    if len(values)!=n: raise ValueError("元素数与 n 不一致")
    for subset in generate_subsets(values): print(" ".join(map(str,subset)) if subset else "EMPTY")
if __name__=="__main__": main()
