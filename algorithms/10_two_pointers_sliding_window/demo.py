import starter
def main():
    text = "abba"; print("输入:", text); print("期望: (2, 0, 2)")
    try: print("实际:", starter.longest_unique_window(text))
    except NotImplementedError as e: print("停止展示:", e)
if __name__ == "__main__": main()
