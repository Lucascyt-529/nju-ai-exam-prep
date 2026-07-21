import starter
def main():
    tokens="3 9 20 # # 15 7".split(); print("输入:",tokens,"期望: [[3], [9, 20], [15, 7]]")
    try: print("实际:",starter.level_order(starter.build_tree(tokens)))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()
