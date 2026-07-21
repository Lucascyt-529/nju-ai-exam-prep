import starter
def main():
    cases=[([(1,3),(2,6),(8,10),(10,12)],[(1,6),(8,12)]),([],[]),([(2,2)],[(2,2)])]; passed=0
    try:
        for values,expected in cases:
            actual=starter.merge_intervals(values); ok=actual==expected; passed+=ok; print(f"{values}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/3"); return 0 if passed==3 else 1
if __name__=="__main__": raise SystemExit(main())
