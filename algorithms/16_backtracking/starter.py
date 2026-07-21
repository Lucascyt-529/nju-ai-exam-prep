"""学生练习：按固定顺序生成整数序列的所有子集。"""
def generate_subsets(values: list[int]) -> list[list[int]]:
    raise NotImplementedError("请完成 generate_subsets")
def main():
    n=int(input()); values=list(map(int,input().split())) if n else []; assert len(values)==n
    for subset in generate_subsets(values): print(" ".join(map(str,subset)) if subset else "EMPTY")
if __name__=="__main__": main()
