"""学生练习：课程依赖的拓扑排序；有环返回空列表。"""
def course_order(n_courses: int, prerequisites: list[tuple[int, int]]) -> list[int]:
    raise NotImplementedError("请完成 course_order")
def main():
    n,m=map(int,input().split()); edges=[tuple(map(int,input().split())) for _ in range(m)]; order=course_order(n,edges); print(*order if order else ["IMPOSSIBLE"])
if __name__=="__main__": main()
