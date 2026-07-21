import starter
def main():
    cases=[("abba",(2,0,2)),("abcabcbb",(3,0,3)),("",(0,0,0))]; passed=0
    try:
        for text, expected in cases:
            actual=starter.longest_unique_window(text); ok=actual==expected; passed+=ok; print(f"{text!r}: 期望={expected}, 实际={actual}, 通过={ok}")
    except NotImplementedError as e: print("停止核对:",e)
    print(f"通过: {passed}/{len(cases)}"); return 0 if passed==len(cases) else 1
if __name__ == "__main__": raise SystemExit(main())
