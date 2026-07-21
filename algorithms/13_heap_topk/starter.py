"""学生练习：输出 Top K 高频整数，频率并列时数值小者优先。"""
def top_k_frequent(values: list[int], k: int) -> list[int]:
    raise NotImplementedError("请完成 top_k_frequent")
def main():
    n,k=map(int,input().split()); values=list(map(int,input().split())); assert len(values)==n; print(*top_k_frequent(values,k))
if __name__=="__main__": main()
