import starter
def main():
    cases=[([1,1,2,2,3],2,[1,2]),([4,4,4,2,2,3],2,[4,2]),([1],0,[])]; passed=0
    try:
        for values,k,expected in cases:
            actual=starter.top_k_frequent(values,k); ok=actual==expected; passed+=ok; print(f"k={k}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/3"); return 0 if passed==3 else 1
if __name__=="__main__": raise SystemExit(main())
