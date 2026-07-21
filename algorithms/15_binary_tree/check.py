import starter
def main():
    cases=[("3 9 20 # # 15 7".split(),[[3],[9,20],[15,7]]),(["1"],[[1]]),(["#"],[])]; passed=0
    try:
        for tokens,expected in cases:
            actual=starter.level_order(starter.build_tree(tokens)); ok=actual==expected; passed+=ok; print(f"{tokens}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/3"); return 0 if passed==3 else 1
if __name__=="__main__": raise SystemExit(main())
