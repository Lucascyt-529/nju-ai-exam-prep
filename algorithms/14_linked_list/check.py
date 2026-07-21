import starter
def main():
    cases=[([1,2,3],[3,2,1]),([1],[1]),([],[])]; passed=0
    try:
        for values,expected in cases:
            actual=starter.linked_list_values(starter.reverse_linked_list(starter.build_linked_list(values))); ok=actual==expected; passed+=ok; print(f"{values}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/3"); return 0 if passed==3 else 1
if __name__=="__main__": raise SystemExit(main())
