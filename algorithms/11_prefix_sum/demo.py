import starter
def main():
    values=[1,1,1]; target=2; print("输入:",values,"target=",target,"期望: 2")
    try: print("实际:",starter.count_subarrays_with_sum(values,target))
    except NotImplementedError as e: print("停止展示:",e)
if __name__=="__main__": main()
