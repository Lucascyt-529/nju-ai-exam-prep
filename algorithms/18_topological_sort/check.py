import starter
def main():
    cases=[(4,[(1,0),(2,0),(3,1),(3,2)],[0,1,2,3]),(2,[(1,0),(0,1)],[]),(1,[],[0])]; passed=0
    try:
        for n,edges,expected in cases:
            actual=starter.course_order(n,edges); ok=actual==expected; passed+=ok; print(f"n={n}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/3"); return 0 if passed==3 else 1
if __name__=="__main__": raise SystemExit(main())
