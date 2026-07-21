import starter
def main():
    values=[1,1,2,2,3]; print("输入:",values,"k=2\n期望: [1, 2]")
    try: print("实际:",starter.top_k_frequent(values,2))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()
