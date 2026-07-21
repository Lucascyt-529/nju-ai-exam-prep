import starter
def main():
    cases=[([1,2],[[],[2],[1],[1,2]]),([], [[]])]; passed=0
    try:
        for values,expected in cases:
            actual=starter.generate_subsets(values); ok=actual==expected; passed+=ok; print(f"{values}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/2"); return 0 if passed==2 else 1
if __name__=="__main__": raise SystemExit(main())
