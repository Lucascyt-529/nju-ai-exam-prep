"""学生练习：每日温度，求右侧首个更高温度的距离。"""
def days_until_warmer(temperatures: list[int]) -> list[int]:
    raise NotImplementedError("请完成 days_until_warmer")
def main():
    n=int(input()); values=list(map(int,input().split())); assert len(values)==n; print(*days_until_warmer(values))
if __name__=="__main__": main()
