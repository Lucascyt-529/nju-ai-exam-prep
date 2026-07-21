import starter
def main():
    edges=[(1,0),(2,0),(3,1),(3,2)]; print("课程数=4, 依赖:",edges,"期望: [0, 1, 2, 3]")
    try: print("实际:",starter.course_order(4,edges))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()
