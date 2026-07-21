import starter
def main():
    cases=[([73,74,75,71,69,72,76,73],[1,1,4,2,1,1,0,0]),([3,2,1],[0,0,0]),([],[])]; passed=0
    try:
        for values,expected in cases:
            actual=starter.days_until_warmer(values); ok=actual==expected; passed+=ok; print(f"{values}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/3"); return 0 if passed==3 else 1
if __name__=="__main__": raise SystemExit(main())
