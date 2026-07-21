import starter
def main():
    values=[1,2]; expected=[[],[2],[1],[1,2]]; print("输入:",values,"期望:",expected)
    try: print("实际:",starter.generate_subsets(values))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()
