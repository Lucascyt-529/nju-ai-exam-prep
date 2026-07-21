import heapq
def course_order(n_courses: int, prerequisites: list[tuple[int, int]]) -> list[int]:
    graph=[[] for _ in range(n_courses)]; indegree=[0]*n_courses
    for course,prerequisite in prerequisites:
        graph[prerequisite].append(course); indegree[course]+=1
    ready=[i for i,d in enumerate(indegree) if d==0]; heapq.heapify(ready); order=[]
    while ready:
        node=heapq.heappop(ready); order.append(node)
        for neighbor in sorted(graph[node]):
            indegree[neighbor]-=1
            if indegree[neighbor]==0: heapq.heappush(ready,neighbor)
    return order if len(order)==n_courses else []
def main():
    n,m=map(int,input().split()); edges=[tuple(map(int,input().split())) for _ in range(m)]; order=course_order(n,edges); print(*order if order else ["IMPOSSIBLE"])
if __name__=="__main__": main()
